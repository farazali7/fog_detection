# Training code goes here

from data_loader import FOGDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def main():

    # data downloaded from https://data.mendeley.com/datasets/r8gmbtv7w2/3 should be in this dir
    # ie there should be a 'data/Filtered Data' folder
    data_dir = "data"

    fog_ds = FOGDataset(
        data_dir=data_dir,
        subjects=[
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
        ],
        modalities=[
            "EMG",
            "ECG",
            "ACC",
            "GYR",
            "NC/SC",
            "IO",
        ],  # specify which we want from data_loader.MODALITIES
        n_windows=1,
        transforms=transforms.ToTensor,
    )

    loader = DataLoader(fog_ds, batch_size=16, shuffle=True)

    for batch_i, batch in enumerate(loader):
        x, y = batch
        print(x.shape)  # (batch_size, n_windows, window_length, num_modalities)
        print(y)  # 0 or 1

        break


if __name__ == "__main__":
    main()
