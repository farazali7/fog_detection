from torch import nn, optim
from torchvision import transforms

cfg = {
    'EPOCHS': 50,
    'BATCH_SIZE': 32,
    'N_WINDOWS': 3,
    'N_HEADS': 8,
    'N_ENC_LAYERS': 3,
    'N_MAX_CONV_FILTERS': 512,
    'WIN_LENGTH': 5.12,
    'OVERLAP': 0.75,
    'SAMPLE_RATE': 40,
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
    'LR': 0.0001156394379977959,
    'WEIGHT_DECAY': 0,
    'BETAS': [0.9, 0.999],
    'EPSILON': 1e-8,
    'MOMENTUM': 0.5,
    'DROPOUT': 0.5,
    'EARLY_STOPPING': 7,
    'CROSS_VAL_FOLDS': 3
}
