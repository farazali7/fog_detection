# Training code goes here

from data_loader import FOGDataset, prepare_data
from models import CNN_FOG
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
import neptune.new as neptune
import sklearn.metrics
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

RESULTS_DICT = {}
RUN_COUNT = 0

# run = neptune.init(
#     project="SYDE599",
# 	api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZWUyZjYwYi1lZGE3LTQ2NGUtYTliOC03OWEzZTRlYjRiMDYifQ=="
# )

def train_val_test_split(subjects, train=10, val=1, test=1):
    random.seed(10)
    # split subjects into train/val/test
    train_subjects = sample(subjects, train)
    subjects = set(subjects) - set(train_subjects)
    val_subjects = sample(subjects, val)
    subjects = set(subjects) - set(val_subjects)
    test_subjects = sample(subjects, test)
    return train_subjects, val_subjects, test_subjects


def run_training(train_ds, val_ds, num_head, num_ec_layers, num_filters, run):
    """Train CT FOG Model"""
    model = CNN_FOG(
        in_channels=train_ds.num_channels, 
        seq_len=train_ds.n_windows, 
        n_heads=num_head, 
        n_enc_layers=num_ec_layers, 
        max_conv_filters=num_filters
    )

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
    train_auc = []
    val_auc = []
    

    # for early stopping
    max_val_auc = 0
    epochs_without_improvement = 0

    all_preds = []
    all_y = []
    
    # train epochs
    for epoch in range(cfg['EPOCHS']):
        # set model up for training
        model.train()
        num_correct = 0
        preds_list = []
        batch_y_list = []
        val_preds_list = []
        val_batch_y_list = []

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
            batch_y_list += list(batch_y)
            preds_list += list(preds)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            num_correct += np.sum(preds == np.array(batch_y))
            
            
        epoch_train_acc = num_correct / train_ds.num_samples
        train_auc_score = sklearn.metrics.roc_auc_score(batch_y_list, preds_list)
        
        print("train acc: ", epoch_train_acc)
        print("train roc: ", train_auc_score)

        # Neptune Logging
        if run is not None:
            run["train/accuracy"].log(epoch_train_acc)
            run["train/auc"].log(train_auc_score)
        
        train_accs.append(epoch_train_acc)
        train_auc.append(train_auc_score)

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
                val_batch_y_list += list(batch_y.detach().numpy())
                val_preds_list += list(preds)
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
                num_correct += np.sum(preds == np.array(batch_y))
                

        epoch_val_acc = num_correct / val_ds.num_samples
        val_auc_score = sklearn.metrics.roc_auc_score(val_batch_y_list, val_preds_list)

        # store validation predictions and labels
        all_preds.append(val_preds_list)
        all_y.append(val_batch_y_list)
        print("val acc: ", epoch_val_acc)
        print("val auc: ", val_auc_score)

        if run is not None:
            run["validation/auc"].log(val_auc_score)
            run["validation/accuracy"].log(epoch_val_acc)

        val_accs.append(epoch_val_acc)
        val_auc.append(val_auc_score)

        # check for early stopping
        if val_auc_score > max_val_auc:
            max_val_auc = val_auc_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= cfg['EARLY_STOPPING']:
            break


    # get best epoch with max val accuracy
    best_epoch = np.argmax(val_auc)

    # Store results
    RESULTS_DICT['run_{0}_y'.format(RUN_COUNT)] = all_y[best_epoch]
    RESULTS_DICT['run_{0}_preds'.format(RUN_COUNT)] = all_preds[best_epoch]

    return train_accs[best_epoch], val_accs[best_epoch], train_auc[best_epoch], val_auc[best_epoch]


def split_subjects(subjects, n_folds=cfg['CROSS_VAL_FOLDS']):
    """Split Subjects Into N Folds"""
    n = len(subjects) // n_folds
    for i in range(0, len(subjects), n):
        yield subjects[i: i + n]


def train_loso(subjects, modalities, sample_rate, win_len, overlap, n_windows, num_head, num_ec_layers, num_filters, locations_drop, run, data_dir="data"):
    """Leave One Subject Out Cross Validation Experiment"""
    
    data_dict = prepare_data(subjects, data_dir, modalities, locations_drop, sample_rate, win_len)
    
    all_val_acc = []
    all_train_acc = []
    all_val_auc = []
    all_train_auc = []
    subject_folds = list(split_subjects(subjects))

    for i, val_subs in enumerate(subject_folds):
        global RUN_COUNT
        RUN_COUNT = i + 1

        # remianing subjects used for training
        train_subs = list(set(subjects) - set(val_subs))

        # construct datasets
        train_ds = FOGDataset(
            subjects=train_subs,
            data_dict=data_dict, 
            overlap=overlap, 
            n_windows=n_windows, 
            sample_rate=sample_rate, 
            win_len=win_len
        )

        val_ds = FOGDataset(
            subjects=val_subs,
            data_dict=data_dict, 
            overlap=overlap, 
            n_windows=n_windows, 
            sample_rate=sample_rate, 
            win_len=win_len
        )

        # run training for current subs
        try:
            run_train_acc, run_val_acc, run_train_auc, run_val_auc = run_training(train_ds, val_ds, num_head, num_ec_layers, num_filters, run)
            print("Run {0} done!".format(i + 1))
            print("Best epoch train acc: ", run_train_acc)
            print("Best epoch val acc: ", run_val_acc)
            print("Best epoch train auc: ", run_train_auc)
            print("Best epoch val auc: ", run_val_auc)

            # store results
            all_train_acc.append(run_train_acc)
            all_val_acc.append(run_val_acc)
            all_train_auc.append(run_train_auc)
            all_val_auc.append(run_val_auc)
        except Exception as e:
            print(e)
            all_train_acc.append(np.nan)
            all_val_acc.append(np.nan)
            all_train_auc.append(np.nan)
            all_val_auc.append(np.nan)
            continue

        

    return np.nanmean(all_train_acc), np.nanmean(all_val_acc), np.nanmean(all_train_auc), np.nanmean(all_val_auc)


def main():
    # data downloaded from https://data.mendeley.com/datasets/r8gmbtv7w2/3 should be in this dir
    # ie there should be a 'data/Filtered Data' folder
    
    train_acc, val_acc, train_auc, val_auc = train_loso(
        subjects=cfg['SUBJECTS'], 
        modalities=cfg['MODALITIES'], 
        sample_rate=cfg['SAMPLE_RATE'], 
        win_len=cfg['WIN_LENGTH'], 
        overlap=cfg['OVERLAP'], 
        n_windows=cfg['N_WINDOWS'], 
        num_head=cfg['N_HEADS'], 
        num_ec_layers=cfg['N_ENC_LAYERS'], 
        num_filters=cfg['N_MAX_CONV_FILTERS'],
        locations_drop=[],
        run=None
    )

    df = pd.DataFrame(RESULTS_DICT)
    df.to_csv('results.csv', index=False)
    print("mean train acc: ", train_acc)
    print("mean val acc", val_acc)
    print("mean train auc: ", train_auc)
    print("mean val auc", val_auc)


if __name__ == "__main__":
    main()
