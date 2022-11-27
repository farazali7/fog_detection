import pandas as pd
import os


def perform_col_swap(df, cols):
    '''
    Perform swap of two columns in a dataframe.
    :param df: DataFrame, initial unswapped data
    :param cols: Tuple of Strings, names of columns to swap
    :return: DataFrame with columns swapped
    '''
    temp = df[cols[0]]
    df[cols[0]] = df[cols[1]]
    df[cols[1]] = temp

    return df


def create_subject_df(data_path, subject_num_str):
    '''
    Load data of all subjects into one dataframe with consistent column ordering.
    :param data_path: String, path to directory containing all subject data files
    :param subject_num_str: String, 3-digit subject number string to load data of (from '001' to '012')
    :return: DataFrame containing requested subject's data
    '''
    # Create dictionary of required column swaps based on ordering in data's README to maintain consistency
    # Done for each subject as list of (col_name_1, col_name_2)
    subjects_swaps = {'001': [('EMG-LTA', 'EMG-RTA')], '002': [('EMG-LTA', 'EMG-RTA')], '003': [], '004': [], '005': [],
                      '006': [('EMG-LTA', 'EMG-RTA')], '007': [('EMG-LTA', 'EMG-RTA')], '008': [('EMG-LTA', 'EMG-RTA')],
                      '008_2': [('EMG-LTA', 'EMG-RGS'), ('EMG-RTA', 'EMG-RGS')], '009': [('ECG', 'EMG-RGS')],
                      '010': [], '011': [], '012': []}
    full_data_paths = []

    # Handle subject 8 separately
    if subject_num_str == '008':
        full_data_paths.append(os.path.join(data_path, subject_num_str, 'OFF_1'))
        full_data_paths.append(os.path.join(data_path, subject_num_str, 'OFF_2'))
    else:
        full_data_paths.append(os.path.join(data_path, subject_num_str))

    subject_df_per_task = []
    for i, full_data_path in enumerate(full_data_paths):
        if i == 1:
            subject_num_str = subject_num_str + '_2'
        for file in os.listdir(full_data_path):
            if file[0] == '.':
                continue
            path = os.path.join(full_data_path, file)
            task_num = int(path.split('/')[-1][-5])
            data = pd.read_csv(path, header=None)
            data = data.drop(data.iloc[:, 2:27], axis=1)
            df_col_names = ['Index', 'Time', 'EMG-LTA', 'EMG-RTA', 'IO', 'ECG', 'EMG-RGS',
                            'LS-ACCX', 'LS-ACCY', 'LS-ACCZ', 'LS-GYRX', 'LS-GYRY', 'LS-GYRZ', 'LS-NC/SC',
                            'RS-ACCX', 'RS-ACCY', 'RS-ACCZ', 'RS-GYRX', 'RS-GYRY', 'RS-GYRZ', 'RS-NC/SC',
                            'W-ACCX', 'W-ACCY', 'W-ACCZ', 'W-GYRX', 'W-GYRY', 'W-GYRZ', 'W-NC/SC',
                            'A-ACCX', 'A-ACCY', 'A-ACCZ', 'A-GYRX', 'A-GYRY', 'A-GYRZ', 'A-NC/SC',
                            'FOG']
            data.columns = df_col_names
            for swap in subjects_swaps[subject_num_str]:
                data = perform_col_swap(data, swap)
            data['Subject'] = subject_num_str
            data['Task'] = task_num
            subject_df_per_task.append(data)

    subject_df = pd.concat(subject_df_per_task, axis=0, ignore_index=True)
    cols = subject_df.columns.tolist()
    cols = [cols[0]] + cols[-2:] + cols[1:-2]
    subject_df = subject_df[cols]

    return subject_df


if __name__ == '__main__':
    data_path = 'data/Filtered Data'

    # Example: load subject 1 dataframe
    subject1_df = create_subject_df(data_path, '001')

    # Example: load subject 8 dataframe
    subject8_df = create_subject_df(data_path, '008')

    print('done')
