# Training code goes here

from data_loader import FOGDataset
from models import CT_FOG
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from config import cfg
from tqdm import tqdm
import random
from random import sample
import numpy as np

def train_val_test_split(subjects, train=10, val=1, test=1):
    random.seed(10)
    # split subjects into train/val/test
    train_subjects = sample(subjects, train)
    subjects = set(subjects) - set(train_subjects)
    val_subjects = sample(subjects, val)
    subjects = set(subjects) - set(val_subjects)
    test_subjects = sample(subjects, test)
    return train_subjects, val_subjects, test_subjects

def main():

    # data downloaded from https://data.mendeley.com/datasets/r8gmbtv7w2/3 should be in this dir
    # ie there should be a 'data/Filtered Data' folder
    data_dir = "data"
    train_subjects, val_subjects, test_subjects = train_val_test_split(cfg['SUBJECTS'], train=2)

    train_ds = FOGDataset(
        data_dir=data_dir,
        subjects=train_subjects,
        modalities=cfg['MODALITIES'],  # specify which we want from data_loader.MODALITIES
        n_windows=cfg['N_WINDOWS']
    )

    val_ds = FOGDataset(
        data_dir=data_dir,
        subjects=val_subjects,
        modalities=cfg['MODALITIES'],  # specify which we want from data_loader.MODALITIES
        n_windows=cfg['N_WINDOWS']
    )

    test_ds = FOGDataset(
        data_dir=data_dir,
        subjects=test_subjects,
        modalities=cfg['MODALITIES'],  # specify which we want from data_loader.MODALITIES
        n_windows=cfg['N_WINDOWS']
    )

    model = CT_FOG(in_channels=train_ds.num_channels, seq_len=train_ds.n_windows)

    #Added for GPU support 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.to(torch.device("cuda:0"))

    loss = cfg['LOSS']()
    optimizer = Adam(
            model.parameters(), 
            cfg['LR'], 
            betas=cfg['BETAS'],
            eps=cfg['EPSILON'],
            weight_decay=cfg['WEIGHT_DECAY']
        )

    train_loader = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True)

    losses = []
    for epoch in range(cfg['EPOCHS']):
        model.train()
        num_correct = 0
        for batch_i, batch in enumerate(train_loader):
            batch_x, batch_y = batch
            # print(batch_x.shape)  # (batch_size, n_windows, window_length, num_modalities)
            # print(batch_y)  # 0 or 1

            if torch.cuda.is_available():
                batch_x = batch_x.to(torch.device("cuda:0"))
                batch_y = batch_y.to(torch.device("cuda:0"))

            output = model(batch_x)

            batch_loss = loss(output, batch_y.float())

            if torch.cuda.is_available():
                batch_loss = batch_loss.to(torch.device("cpu"))
            
            losses.append(batch_loss.detach().numpy())

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            preds = output.detach().numpy()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            num_correct += np.sum(preds == np.array(batch_y))

        
        print("train acc: ", num_correct / train_ds.num_samples)

        num_correct = 0
        model.eval()
        with torch.no_grad():
            for batch_i, batch in enumerate(val_loader):
                batch_x, batch_y = batch
                
                if torch.cuda.is_available():
                    batch_x = batch_x.to(torch.device("cuda:0"))
                    batch_y = batch_y
                    output = model(batch_x).to(torch.device("cpu"))
                else:
                    output = model(batch_x)

                preds = output.detach().numpy()
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
                num_correct += np.sum(preds == np.array(batch_y))
        print("val acc: ", num_correct / val_ds.num_samples)

if __name__ == "__main__":
    main()
