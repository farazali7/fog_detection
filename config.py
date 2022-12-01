from torch import nn, optim
from torchvision import transforms

cfg = {
    'EPOCHS': 30,
    'BATCH_SIZE': 16,
    'N_WINDOWS': 1,
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
    'MODALITIES': {
            # "EMG": [
            #     "EMG-LTA",
            #     "EMG-RTA",
            #     "EMG-RGS"
            #     ],
            # "ECG": [
            #     "ECG"
            # ],
            "ACC": [
                "A-ACCX",
                "A-ACCY",
                "A-ACCZ",
                "LS-ACCX",
                "LS-ACCY",
                "LS-ACCZ",
                "RS-ACCX",
                "RS-ACCY",
                "RS-ACCZ",
                "W-ACCX",
                "W-ACCY",
                "W-ACCZ",
            ],
            # "GYR": [
            #     "A-GYRX",
            #     "A-GYRY",
            #     "A-GYRZ",
            #     "LS-GYRX",
            #     "LS-GYRY",
            #     "LS-GYRZ",
            #     "RS-GYRX",
            #     "RS-GYRY",
            #     "RS-GYRZ",
            #     "W-GYRX",
            #     "W-GYRY",
            #     "W-GYRZ",
                
            # ],
            # "NC/SC": [
            #     "A-NC/SC",
            #     "LS-NC/SC",
            #     "RS-NC/SC",
            #     "W-NC/SC",
            # ],
            # "IO": [
            #     "IO"
            # ],
    },
    'LOSS': nn.BCELoss,
    'OPTIMIZER': 'adam',
    'LR': 0.3,
    'WEIGHT_DECAY': 0,
    'BETAS': [0.9, 0.999],
    'EPSILON': 1e-8,
    'MOMENTUM': 0.3,
    'ROTATION_AUG': 10,
    'SHEAR_AUG': 10,
    'TRANSLATION_AUG': 0.2,
    'CONTRAST_AUG': 0,
    'FILTER_NUMS': [19, 38, 76, 152, 152, 152, 152],
    'FILTER_SIZES': [5, 5, 5, 5, 5, 5, 5],
    'STRIDES': [1, 1, 1, 1, 1, 1, 1],
    'POOLINGS': [0, 0, 0, 0, 0, 0, 0],
    'BATCH_NORM': [1, 1, 1, 1, 1, 1, 1],
    'NUM_EPOCHS': 10,
    'FC_SIZES': [40, 10],
    'DROPOUT': [0, 0, 0, 0, 0, 0, 0],
    'L2_REG': False,
    'L1_REG': False
}