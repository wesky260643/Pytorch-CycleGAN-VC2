import librosa
import numpy as np
import os
import pyworld
from pprint import pprint
import librosa.display
import time
import tqdm
import queue
import threading
from multiprocessing import Process, Manager, Queue

import functools
print = functools.partial(print, flush=True)

def wav_producer(file_list, wave_queue, sr):
    while True:
        file_path = file_list.get()
        if file_path is None:
            break
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        wave_queue.put(wav)
        # print("produce a wave file: %s, wave_queue size: %d" % (file_path, wave_queue.qsize()), file_list.qsize())
    wave_queue.put(None)

def wav_comsumer(wave_queue, wav_res_list, num_producer):
    while True:
        wav = wave_queue.get()
        if wav is None:
            num_producer -= 1
        else:
            wav_res_list.append(wav)
            # print("comsume from wave_queue and put into wav_res_list size: ", len(wav_res_list))
        if num_producer == 0:
            break

def load_wavs(wav_dir, sr, ignore=None, num_thread=128):
    file_list = queue.Queue()
    wave_queue = queue.Queue()
    wav_res_list = list()

    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        if ignore is not None:
            if ignore in file_path: 
                print("ignore file: ", file_path)
                continue
        file_list.put(file_path)
    print("num of wave files: %d" % (file_list.qsize()))
    for i in range(num_thread):
        file_list.put(None)
    thread_list = list()
    for i in range(num_thread):
        producer = threading.Thread(target=wav_producer, args=(file_list, wave_queue, sr, ))
        thread_list.append(producer)
        producer.start()
    consumer = threading.Thread(target=wav_comsumer, args=(wave_queue, wav_res_list, num_thread))
    thread_list.append(consumer)
    consumer.start()
    for t in thread_list:
        t.join()
    print("load wave data done: %d wave files in %s" % (len(wav_res_list), wav_dir))
    return wav_res_list

    
def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(
        wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)

    # Finding Spectogram
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)

    # Finding aperiodicity
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    # Use this in Ipython to see plot
    # librosa.display.specshow(np.log(sp).T,
    #                          sr=fs,
    #                          hop_length=int(0.001 * fs * frame_period),
    #                          x_axis="time",
    #                          y_axis="linear",
    #                          cmap="magma")
    # colorbar()
    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-Cepstral coefficients (MCEPs)
    # sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def world_encode_producer(file_list, encoded_queue, fs, frame_period=5.0, coded_dim=24):
    while True:
        file_path = file_list.get()
        if file_path is None:
            break
        wav, _ = librosa.load(file_path, sr=fs, mono=True)
        f0, timeaxis, sp, ap = world_decompose(wav=wav,
                                               fs=fs,
                                               frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        # encoded_queue.put((f0, timeaxis, sp, ap))
        encoded_queue.put((f0, timeaxis, sp, ap, coded_sp))
        print("world producer: put data into encoded_queue, size of encoded_queue: %d, size of file_list: %d" % (encoded_queue.qsize(), file_list.qsize()))
    encoded_queue.put(None)

def world_encode_comsumer(encoded_queue, encoded_list, num_producer, fs, coded_dim=24):
    while True:
        d_encoded = encoded_queue.get()
        if d_encoded is None:
            num_producer -= 1
        else:
            # f0, timeaxis, sp, ap = d_encoded[0], d_encoded[1], d_encoded[2], d_encoded[3]
            # coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
            # encoded_list.append((f0, timeaxis, sp, ap, coded_sp))
            encoded_list.append(d_encoded)
            print("world consumer: get item from encoded_queue and put into encoded_list, encoded_list size", len(encoded_list))
            # encoded_list.put(d_encoded)
            # print("world consumer: get item from encoded_queue and put into encoded_list, encoded_list size", encoded_list.qsize())
        if num_producer == 0:
            break

def world_encode_data(wav_dir, fs, frame_period=5.0, coded_dim=24, ignore=None, num_producer=64):
    # file_list = queue.Queue()
    # encoded_queue = queue.Queue()
    stime = time.time()
    file_list = Manager().Queue()
    encoded_queue = Manager().Queue()
    encoded_list = Manager().list()
    
    for file_path in wav_dir:
        file_list.put(file_path)
    print("num of wave files: %d" % (file_list.qsize()))
    for i in range(num_producer):
        file_list.put(None)

    thread_list = list()
    for i in range(num_producer):
        # producer = threading.Thread(target=world_encode_producer, args=(file_list, encoded_queue, fs, frame_period, coded_dim, ))
        producer = Process(target=world_encode_producer, args=(file_list, encoded_queue, fs, frame_period, coded_dim, ))
        thread_list.append(producer)
        producer.start()

    # num_consumer = 20 
    # for i in range(num_consumer):
    # consumer = threading.Thread(target=world_encode_comsumer, args=(encoded_queue, encoded_list, num_producer, fs, coded_dim, ))
    consumer = Process(target=world_encode_comsumer, args=(encoded_queue, encoded_list, num_producer, fs, coded_dim, ))
    thread_list.append(consumer)
    consumer.start()

    for t in thread_list:
        t.join()
    for t in thread_list:
        t.terminate()

    # world_encode_comsumer(encoded_queue, encoded_list, num_producer, fs, coded_dim)
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()
    # for item in encoded_list:
    print("num of encoded data from wav files: ", len(encoded_list))
    while len(encoded_list) > 0:
        item = encoded_list.pop()
        f0, timeaxis, sp, ap, coded_sp = item[0], item[1], item[2], item[3], item[4]
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)
    etime = time.time()
    print("world encode time cost: ", etime-stime )
    return f0s, timeaxes, sps, aps, coded_sps

    # f0s = list()
    # timeaxes = list()
    # sps = list()
    # aps = list()
    # coded_sps = list()
    # print("world encoding data....")
    # for wav in tqdm.tqdm(wav_dir):
    #     f0, timeaxis, sp, ap = world_decompose(wav=wav,
    #                                            fs=fs,
    #                                            frame_period=frame_period)
    #     coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
    #     f0s.append(f0)
    #     timeaxes.append(timeaxis)
    #     sps.append(sp)
    #     aps.append(ap)
    #     coded_sps.append(coded_sp)
    # return f0s, timeaxes, sps, aps, coded_sps


def logf0_statistics(f0s):
    # Note: np.ma.log() calculating log on masked array (for incomplete or invalid entries in array)
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()
    return log_f0s_mean, log_f0s_std


def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def coded_sps_normalization_fit_transform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        tmp = (coded_sp - coded_sps_mean) / coded_sps_std
        coded_sps_normalized.append( tmp )
    return coded_sps_normalized, coded_sps_mean, coded_sps_std


def wav_padding(wav, sr, frame_period, multiple=4):

    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) +
                                      1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right),
                        'constant', constant_values=0)

    return wav_padded


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian Normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) /
                          std_log_src * std_log_target + mean_log_target)
    return f0_converted


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    wav = wav.astype(np.float32)
    return wav


def sample_train_data(dataset_A, dataset_B, n_frames=128):
    # Created Pytorch custom dataset instead
    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:, start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B


if __name__ == '__main__':
    start_time = time.time()
    # wavs = load_wavs("data/vcc2018_training.speakers/VCC2SF2/", 16000)
    # # wavs = load_wavs("data/vctk_vcc2018_peppapig/", 16000)
    # print("load wave data done, num of waves: ", len(wavs))

    # f0, timeaxis, sp, ap = world_decompose(wavs[0], 16000, 5.0)
    # print(f0.shape, timeaxis.shape, sp.shape, ap.shape)

    # coded_sp = world_encode_spectral_envelop(sp, 16000, 24)
    # print(coded_sp.shape)

    f0s, timeaxes, sps, aps, coded_sps = world_encode_data("data/tmp/", 16000, 5, 24)
    # f0s, timeaxes, sps, aps, coded_sps = world_encode_data("data/vcc2018_training.speakers/VCC2SF2/", 16000, 5, 24)
    # f0s, timeaxes, sps, aps, coded_sps = world_encode_data(wavs, 16000, 5, 24)
    print("size of sps: ", len(sps))
    exit(0)
    # print(f0s)

    log_f0_mean, log_f0_std = logf0_statistics(f0s)
    # print(log_f0_mean)

    coded_sps_transposed = transpose_in_list(lst=coded_sps)
    # print(coded_sps_transposed)

    coded_sps_norm, coded_sps_mean, coded_sps_std = coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_transposed)
    print(
        "Total time for preprcessing-> {:.4f}".format(time.time() - start_time))

    print(len(coded_sps_norm), coded_sps_norm[0].shape)
    temp_A = np.random.randn(162, 24, 550)
    temp_B = np.random.randn(158, 24, 550)

    a, b = sample_train_data(temp_A, temp_B)
    print(a.shape, b.shape)
