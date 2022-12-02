# Training code goes here

from data_loader import FOGDataset, prepare_data
from models import CT_FOG
import os
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


def run_training(train_ds, val_ds):
    """Train CT FOG Model"""
    model = CT_FOG(in_channels=train_ds.num_channels, seq_len=train_ds.n_windows)

    #Added for GPU support 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.to(torch.device("cuda:0"))

    # set up loss and optimizer
    loss = cfg['LOSS']()
    optimizer = Adam(
            model.parameters(), 
            cfg['LR'], 
            betas=cfg['BETAS'],
            eps=cfg['EPSILON'],
            weight_decay=cfg['WEIGHT_DECAY']
        )

    # data loaders
    train_loader = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True, drop_last=True)

    val_accs = []
    train_accs = []
    
    # train epochs
    for epoch in range(cfg['EPOCHS']):
        # set model up for training
        model.train()
        num_correct = 0

        # train on training batches
        for batch_i, batch in enumerate(train_loader):
            batch_x, batch_y = batch

            if torch.cuda.is_available():
                batch_x = batch_x.to(torch.device("cuda:0"))
                batch_y = batch_y.to(torch.device("cuda:0"))

            output = model(batch_x)

            batch_loss = loss(output, batch_y.float())

            if torch.cuda.is_available():
                batch_loss = batch_loss.to(torch.device("cpu"))

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            preds = output.detach().numpy()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            num_correct += np.sum(preds == np.array(batch_y))

        epoch_train_acc = num_correct / train_ds.num_samples
        print("train acc: ", epoch_train_acc)
        train_accs.append(epoch_train_acc)

        num_correct = 0

        # set model up for eval mode
        model.eval()

        # Validate on validation batches
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
        epoch_val_acc = num_correct / val_ds.num_samples
        print("val acc: ", epoch_val_acc)
        val_accs.append(epoch_val_acc)

    # get best epoch with max val accuracy
    best_epoch = np.argmax(val_accs)
    return train_accs[best_epoch], val_accs[best_epoch]


def train_loso(subjects, data_dir="data"):
    """Leave One Subject Out Cross Validation Experiment"""
    
    data_dict = prepare_data(subjects, data_dir)
    
    all_val_acc = []
    all_train_acc = []

    for i, sub in enumerate(subjects):
        # current subject is held out for validation
        val_subs = [sub]

        # remianing subjects used for training
        train_subs = list(set(subjects) - set(val_subs))

        # construct datasets
        train_ds = FOGDataset(
            train_subs,
            data_dict
        )

        val_ds = FOGDataset(
            val_subs,
            data_dict
        )

        # run training for current subs
        try:
            run_train_acc, run_val_acc = run_training(train_ds, val_ds)
        except Exception as e:
            print(e)
            continue

        print("Run {0} done!".format(i + 1))
        print("Best epoch train acc: ", run_train_acc)
        print("Best epoch val acc: ", run_val_acc)

        # store results
        all_train_acc.append(run_train_acc)
        all_val_acc.append(run_val_acc)

    return np.mean(all_train_acc), np.mean(all_val_acc)


def main():
    # data downloaded from https://data.mendeley.com/datasets/r8gmbtv7w2/3 should be in this dir
    # ie there should be a 'data/Filtered Data' folder
    
    train_acc, val_acc = train_loso(cfg['SUBJECTS'])
    print("mean train acc: ", train_acc)
    print("mean val acc", val_acc)


if __name__ == "__main__":
    main()
