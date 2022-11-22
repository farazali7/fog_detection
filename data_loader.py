# Code to load (and partition) dataset
import pandas as pd

data_path = 'data/Filtered Data/001/task_1.txt'
data = pd.read_csv(data_path, header=None)
data = data.drop(data.iloc[:, 2:27], axis=1)
# TODO: Ordering of EMG data is switched for some patients... view data README
df_col_names = ['Index', 'Time', 'EMG-LTA', 'EMG-RTA', 'IO', 'ECG', 'EMG-GS',
                'LS-ACCX', 'LS-ACCY', 'LS-ACCZ', 'LS-GYRX', 'LS-GYRY', 'LS-GYRZ', 'LS-NC/SC',
                'RS-ACCX', 'RS-ACCY', 'RS-ACCZ', 'RS-GYRX', 'RS-GYRY', 'RS-GYRZ', 'RS-NC/SC',
                'W-ACCX', 'W-ACCY', 'W-ACCZ', 'W-GYRX', 'W-GYRY', 'W-GYRZ', 'W-NC/SC',
                'A-ACCX', 'A-ACCY', 'A-ACCZ', 'A-GYRX', 'A-GYRY', 'A-GYRZ', 'A-NC/SC',
                'FOG']
data.columns = df_col_names
print('done')