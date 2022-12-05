import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import cfg
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


MODALITIES = {"EMG", "ECG", "ACC", "GYR", "NC/SC", "IO"}

# not used for now but could be useful later
MODALITIES_DETAILED = {
            "EMG": [
                "EMG-LTA",
                "EMG-RTA",
                "EMG-RGS"
                ],
            "ECG": [
                "ECG"
            ],
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
            "GYR": [
                "A-GYRX",
                "A-GYRY",
                "A-GYRZ",
                "LS-GYRX",
                "LS-GYRY",
                "LS-GYRZ",
                "RS-GYRX",
                "RS-GYRY",
                "RS-GYRZ",
                "W-GYRX",
                "W-GYRY",
                "W-GYRZ", 
            ],
            "NC/SC": [
                "A-NC/SC",
                "LS-NC/SC",
                "RS-NC/SC",
                "W-NC/SC",
            ],
            "IO": [
                "IO"
            ],
    }

def perform_col_swap(df, cols):
    """
    Perform swap of two columns in a dataframe.
    :param df: DataFrame, initial unswapped data
    :param cols: Tuple of Strings, names of columns to swap
    :return: DataFrame with columns swapped
    """
    temp = df[cols[0]]
    df[cols[0]] = df[cols[1]]
    df[cols[1]] = temp

    return df


def create_subject_df(data_path, subject_num_str):
    """
    Load data of all subjects into one dataframe with consistent column ordering.
    :param data_path: String, path to directory containing all subject data files
    :param subject_num_str: String, 3-digit subject number string to load data of (from '001' to '012')
    :return: DataFrame containing requested subject's data
    """
    # Create dictionary of required column swaps based on ordering in data's README to maintain consistency
    # Done for each subject as list of (col_name_1, col_name_2)
    subjects_swaps = {
        "001": [("EMG-LTA", "EMG-RTA")],
        "002": [("EMG-LTA", "EMG-RTA")],
        "003": [],
        "004": [],
        "005": [],
        "006": [("EMG-LTA", "EMG-RTA")],
        "007": [("EMG-LTA", "EMG-RTA")],
        "008": [("EMG-LTA", "EMG-RTA")],
        "008_2": [("EMG-LTA", "EMG-RGS"), ("EMG-RTA", "EMG-RGS")],
        "009": [("ECG", "EMG-RGS")],
        "010": [],
        "011": [],
        "012": [],
    }
    full_data_paths = []

    # Handle subject 8 separately
    if subject_num_str == "008":
        full_data_paths.append(os.path.join(data_path, subject_num_str, "OFF_1"))
        full_data_paths.append(os.path.join(data_path, subject_num_str, "OFF_2"))
    else:
        full_data_paths.append(os.path.join(data_path, subject_num_str))

    subject_df_per_task = []
    for i, full_data_path in enumerate(full_data_paths):
        if i == 1:
            subject_num_str = subject_num_str + "_2"
        for file in os.listdir(full_data_path):
            if file[0] == ".":
                continue
            path = os.path.join(full_data_path, file)
            task_num = int(path.split("/")[-1][-5])
            data = pd.read_csv(path, header=None)
            data = data.drop(data.iloc[:, 2:27], axis=1)
            df_col_names = [
                "Index",
                "Time",
                "EMG-LTA",
                "EMG-RTA",
                "IO",
                "ECG",
                "EMG-RGS",
                "LS-ACCX",
                "LS-ACCY",
                "LS-ACCZ",
                "LS-GYRX",
                "LS-GYRY",
                "LS-GYRZ",
                "LS-NC/SC",
                "RS-ACCX",
                "RS-ACCY",
                "RS-ACCZ",
                "RS-GYRX",
                "RS-GYRY",
                "RS-GYRZ",
                "RS-NC/SC",
                "W-ACCX",
                "W-ACCY",
                "W-ACCZ",
                "W-GYRX",
                "W-GYRY",
                "W-GYRZ",
                "W-NC/SC",
                "A-ACCX",
                "A-ACCY",
                "A-ACCZ",
                "A-GYRX",
                "A-GYRY",
                "A-GYRZ",
                "A-NC/SC",
                "FOG",
            ]
            data.columns = df_col_names
            for swap in subjects_swaps[subject_num_str]:
                data = perform_col_swap(data, swap)
            data["Subject"] = subject_num_str
            data["Task"] = task_num
            subject_df_per_task.append(data)

    subject_df = pd.concat(subject_df_per_task, axis=0, ignore_index=True)
    cols = subject_df.columns.tolist()
    cols = [cols[0]] + cols[-2:] + cols[1:-2]
    subject_df = subject_df[cols]

    return subject_df


class FOGDataset(Dataset):
    def __init__(
        self,
        subjects,
        data_dict,
        overlap=cfg['OVERLAP'],
        n_windows=cfg['N_WINDOWS'],
        sample_rate=cfg['SAMPLE_RATE'],
        win_len=cfg['WIN_LENGTH']
    ):

        """
        save all data to one numpy file and access windows with memmap
        """
        self.n_windows = n_windows
        self.win_len = int(win_len * sample_rate)

        all_data = []
        for subject in subjects:
            all_data.append(data_dict[subject])
        
        all_data = np.concatenate(all_data)

        self.num_timesteps = len(all_data)
        self.data = all_data

        self.num_channels = all_data.shape[-1] - 1 # - 1 because 1 channel is for label

        # load all samples from file
        self.win_step = int((1 - overlap) * self.win_len)
        self.num_samples = int(
            (len(all_data) - (overlap) * self.win_len) // self.win_step
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        windows = []
        for i in range(self.n_windows):
            win_start = max(0, (idx - i)) * self.win_step
            win_end = win_start + self.win_len
            windows.append(self.data[win_start:win_end])
        windows = np.stack(windows)
        fog = windows[0, :, -1]  # only consider labels of latest window
        y = 1 if np.mean(fog) > 0.5 else 0

        windows = torch.tensor(windows, dtype=torch.float32)

        return windows[..., :-1], y


def prepare_data(subjects, data_dir, modalities=cfg['MODALITIES'], locations_drop=[], sample_rate=cfg['SAMPLE_RATE'], win_len=cfg['WIN_LENGTH']):
    all_data = {}
    win_len = int(win_len * sample_rate)
    for subject in subjects:
        subject_df = create_subject_df(
            os.path.join(data_dir, "Filtered Data"), subject
        )
        # drop unneeded
        subject_df = subject_df.drop(
            ["Index", "Time", "Subject", "Task", "Time"], axis=1
        )
        for mode in MODALITIES:
            if mode not in modalities:
                cols_to_drop = [col for col in subject_df.columns if mode in col]
                subject_df = subject_df.drop(cols_to_drop, axis=1)

        for location in locations_drop:
            cols_to_drop = [col for col in subject_df.columns if location in col]
            subject_df = subject_df.drop(cols_to_drop, axis=1)

        # TODO: putting signal filtering here

        # downsample to sample_rate
        sample_period = int(500 / sample_rate)
        subject_df = subject_df.iloc[::sample_period]

        # normalize data
        subject_df = (subject_df-subject_df.min())/(subject_df.max()-subject_df.min())
        subject_df = subject_df.fillna(0)

        # pad rows to subject data so that we don't have overlapping windows between subjects
        # need at least a windows worth of dummy rows
        subject_df = subject_df.append(
            pd.DataFrame(
                [[0] * subject_df.shape[1]] * win_len,
                columns=subject_df.columns,
            )
        )

        all_data[subject] = subject_df.to_numpy()

    return all_data


if __name__ == "__main__":
    data_path = "fog_detection/data/"

    fog_ds = FOGDataset(
        data_dir=data_path,
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
        ],  # specify from MODALITIES
        n_windows=1,
    )

    print("done")
