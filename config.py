from torch import nn, optim
from torchvision import transforms

cfg = {
    'EPOCHS': 25,
    'BATCH_SIZE': 16,
    'N_WINDOWS': 1,
    'N_HEADS': 8,
    'N_ENC_LAYERS': 3,
    'N_MAX_CONV_FILTERS': 128,
    'WIN_LENGTH': 5.12,
    'OVERLAP': 0.75,
    'SAMPLE_RATE': 25,
    'SUBJECTS': [
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010",
            "011",
            "012"
        ],
    'MODALITIES': [
            "EMG",
            "ECG",
            "ACC", 
            "GYR", 
            "NC/SC"
        ],
    'LOSS': nn.BCELoss,
    'OPTIMIZER': 'adam',
    'LR': 0.0001,
    'WEIGHT_DECAY': 0,
    'BETAS': [0.9, 0.999],
    'EPSILON': 1e-8,
    'MOMENTUM': 0.1,
    'DROPOUT': 0.1,
    'EARLY_STOPPING': 7,
    'CROSS_VAL_FOLDS': 4
}
