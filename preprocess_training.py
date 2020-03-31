import os
import time
import argparse
import numpy as np
import pickle

# Custom Classes
import preprocess

import functools
print = functools.partial(print, flush=True)

def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_for_training(train_A_dir, train_B_dir, cache_folder, ignore_pat=None):
    num_mcep = 36
    sampling_rate = 22000
    frame_period = 5.0
    n_frames = 128

    print("Starting to prepocess data.......")
    start_time = time.time()

    # wavs_A = preprocess.load_wavs(wav_dir=train_A_dir, sr=sampling_rate, ignore=ignore_pat)
    # wavs_B = preprocess.load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    # f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
    #     wave=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    # f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = preprocess.world_encode_data(
    #     wave=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    wave_file_list_A = list()
    for file in os.listdir(train_A_dir):
        file_path = os.path.join(train_A_dir, file)
        if ignore_pat is not None:
            if ignore_pat in file_path: 
                print("ignore file: ", file_path)
                continue
        wave_file_list_A.append(file_path)
    if len(wave_file_list_A) > 3000 * 4:
        f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = list(), list(), list(), list(), list()
        for sub in range(int(len(wave_file_list_A) / 3000)):
            sub_start = sub * 3000
            sub_end = min(sub_start + 3000, len(wave_file_list_A))
            tmp_f0s_A, tmp_timeaxes_A, tmp_sps_A, tmp_aps_A, tmp_coded_sps_A = preprocess.world_encode_data(
                    wav_dir=wave_file_list_A[sub_start:sub_end], fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep, ignore=ignore_pat)
            f0s_A.extend(tmp_f0s_A)
            timeaxes_A.extend(tmp_timeaxes_A)
            sps_A.extend(tmp_sps_A)
            aps_A.extend(tmp_aps_A)
            coded_sps_A .extend(tmp_coded_sps_A)

    else:
        f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
            wav_dir=wave_file_list_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep, ignore=ignore_pat)
    print("world_encode_data ok. f0s_A: %d" % (len(f0s_A)))
    
    wave_file_list_B = list()
    for file in os.listdir(train_B_dir):
        file_path = os.path.join(train_B_dir, file)
        wave_file_list_B.append(file_path)
    if len(wave_file_list_B) > 3000 * 4:
        f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = list(), list(), list(), list(), list()
        for sub in range(int(len(wave_file_list_B) / 3000)):
            sub_start = sub * 3000
            sub_end = sub * 3000
            tmp_f0s_B, tmp_timeaxes_B, tmp_sps_B, tmp_aps_B, tmp_coded_sps_B = preprocess.world_encode_data(
                wav_dir=wave_file_list_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
            f0s_B.extend(tmp_f0s_B)
            timeaxes_B.extend(tmp_timeaxes_B)
            sps_B.extend(tmp_sps_B)
            aps_B.extend(tmp_aps_B)
            coded_sps_B.extend(tmp_coded_sps_B)
    else:
        f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = preprocess.world_encode_data(
            wav_dir=wave_file_list_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean_A, log_f0s_std_A = preprocess.logf0_statistics(f0s=f0s_A)
    log_f0s_mean_B, log_f0s_std_B = preprocess.logf0_statistics(f0s=f0s_B)

    print("Log Pitch A")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))
    print("Log Pitch B")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_B, log_f0s_std_B))

    coded_sps_A_transposed = preprocess.transpose_in_list(lst=coded_sps_A)
    coded_sps_B_transposed = preprocess.transpose_in_list(lst=coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_B_transposed)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
             mean_A=log_f0s_mean_A,
             std_A=log_f0s_std_A,
             mean_B=log_f0s_mean_B,
             std_B=log_f0s_std_B)

    np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
             mean_A=coded_sps_A_mean,
             std_A=coded_sps_A_std,
             mean_B=coded_sps_B_mean,
             std_B=coded_sps_B_std)

    save_pickle(variable=coded_sps_A_norm,
                fileName=os.path.join(cache_folder, "coded_sps_A_norm.pickle"))
    save_pickle(variable=coded_sps_B_norm,
                fileName=os.path.join(cache_folder, "coded_sps_B_norm.pickle"))

    end_time = time.time()
    print("Preprocessing finsihed!! see your directory ../cache for cached preprocessed data")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for training Cycle GAN using PyTorch')
    train_A_dir_default = 'data/vcc2018_training/VCC2SF1/'
    train_B_dir_default = 'data/vcc2018_training/VCC2TM1/'
    cache_folder_default = 'data/cache_check.2/'

    parser.add_argument('--train_A_dir', type=str,
                        help="Directory for source voice sample", default=train_A_dir_default)
    parser.add_argument('--train_B_dir', type=str,
                        help="Directory for target voice sample", default=train_B_dir_default)
    parser.add_argument('--cache_folder', type=str,
                        help="Store preprocessed data in cache folders", default=cache_folder_default)
    parser.add_argument("--ignore_pat", type=str, default=None, help="pattern to ignore in train_A_dir")
    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    cache_folder = argv.cache_folder

    preprocess_for_training(train_A_dir, train_B_dir, cache_folder, argv.ignore_pat)
