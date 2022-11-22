import pandas as pd
import os
import pickle
import bz2


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


def create_full_subject_df(data_path):
    '''
    Load data of all subjects into one dataframe with consistent column ordering.
    :param data_path: String, path to directory containing subject data files
    :return: DataFrame containing all subjects' data
    '''
    subjects_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [],
                     7: [], 81: [], 82: [], 9: [], 10: [], 11: [], 12: []}
    for path, subdirs, files in os.walk(data_path):
        subdir_name = path.split('/')[-1]
        if '008/OFF_1' in path:
            subdir_name = 81
        elif '008/OFF_2' in path:
            subdir_name = 82
        elif not subdir_name[0] == '0':
            continue
        subject_num = int(subdir_name)
        for name in files:
            if name[0] == '.':
                continue
            subjects_dict[subject_num].append(os.path.join(path, name))
    subject_datasets = []
    # Create dictionary of required column swaps based on ordering in data's README to maintain consistency
    # Done for each subject as list of (col_name_1, col_name_2)
    subjects_swaps = {1: [('EMG-LTA', 'EMG-RTA')], 2: [('EMG-LTA', 'EMG-RTA')], 3: [], 4: [], 5: [],
                      6: [('EMG-LTA', 'EMG-RTA')], 7: [('EMG-LTA', 'EMG-RTA')], 81: [('EMG-LTA', 'EMG-RTA')],
                      82: [('EMG-LTA', 'EMG-RGS'), ('EMG-RTA', 'EMG-RGS')], 9: [('ECG', 'EMG-RGS')],
                      10: [], 11: [], 12: []}
    for subject_num in subjects_dict.keys():
        subject_paths = subjects_dict[subject_num]
        for path in subject_paths:
            task_num = int(path.split('/')[-1][-5])
            data = pd.read_csv(path, header=None)
            data = data.drop(data.iloc[:, 2:27], axis=1)
            # TODO: Ordering of EMG data is switched for some patients... view data README
            df_col_names = ['Index', 'Time', 'EMG-LTA', 'EMG-RTA', 'IO', 'ECG', 'EMG-RGS',
                            'LS-ACCX', 'LS-ACCY', 'LS-ACCZ', 'LS-GYRX', 'LS-GYRY', 'LS-GYRZ', 'LS-NC/SC',
                            'RS-ACCX', 'RS-ACCY', 'RS-ACCZ', 'RS-GYRX', 'RS-GYRY', 'RS-GYRZ', 'RS-NC/SC',
                            'W-ACCX', 'W-ACCY', 'W-ACCZ', 'W-GYRX', 'W-GYRY', 'W-GYRZ', 'W-NC/SC',
                            'A-ACCX', 'A-ACCY', 'A-ACCZ', 'A-GYRX', 'A-GYRY', 'A-GYRZ', 'A-NC/SC',
                            'FOG']
            data.columns = df_col_names
            for swap in subjects_swaps[subject_num]:
                data = perform_col_swap(data, swap)
            data['Subject'] = subject_num
            data['Task'] = task_num
            subject_datasets.append(data)

    all_data = pd.concat(subject_datasets, axis=0, ignore_index=True)
    cols = all_data.columns.tolist()
    cols = [cols[0]] + cols[-2:] + cols[1:-2]
    all_data = all_data[cols]

    return all_data


def load_data(path):
    '''
    Load BZ2 compressed data file.
    :param path: String, path to data
    :returns: DataFrame
    '''
    data = bz2.BZ2File(path, 'rb')
    data = pickle.load(data)

    return data


if __name__ == '__main__':
    # data_path = 'data/Filtered Data'
    # data = create_full_subject_df(data_path)
    #
    # # Compress and save
    # sfile = bz2.BZ2File('all_subject_data', 'w')
    # pickle.dump(data, sfile)

    all_data_path = 'all_subject_data.pbz2'
    data = load_data(all_data_path)

    print('done')
