from torch.utils.data.dataset import Dataset
import torch
import numpy as np

import functools
print = functools.partial(print, flush=True)

class trainingDataset(Dataset):
    def __init__(self, datasetA, datasetB, n_frames=128):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n_frames = n_frames
        self.dataset_length = 0
        self.load_to_mem()

    def load_to_mem(self):
        dataset_A = self.datasetA
        dataset_B = self.datasetB
        n_frames = self.n_frames

        len_A, len_B = len(dataset_A), len(dataset_B)
        self.length = min(len_A, len_B)

        num_samples = min(len_A, len_B)
        train_data_A_idx = np.arange(len_A)
        train_data_B_idx = np.arange(len_B)
        np.random.shuffle(train_data_A_idx)
        np.random.shuffle(train_data_B_idx)
        if len_A > len_B:
            num_rpt = int(len_A / len_B)
            train_data_A_idx_subset = train_data_A_idx
            train_data_B_idx_subset = np.append(np.tile(train_data_B_idx, num_rpt), train_data_B_idx[:len_A-len_B*num_rpt])
        else:
            num_rpt = int(len_B / len_A)
            train_data_A_idx_subset = np.append(np.tile(train_data_A_idx, num_rpt), train_data_A_idx[:len_B-len_A*num_rpt])
            train_data_B_idx_subset = train_data_B_idx
        # train_data_A_idx_subset = train_data_A_idx[:num_samples]
        # train_data_B_idx_subset = train_data_B_idx[:num_samples]

        self.train_data_A = list()
        self.train_data_B = list()

        for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
            data_A = dataset_A[idx_A]
            frames_A_total = data_A.shape[1]
            # assert frames_A_total >= n_frames
            if frames_A_total < n_frames:
                print("warning: frames_B_total < n_frames")
                continue
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            self.train_data_A.append(data_A[:, start_A:end_A])

            data_B = dataset_B[idx_B]
            frames_B_total = data_B.shape[1]
            # assert frames_B_total >= n_frames
            if frames_B_total < n_frames:
                print("warning: frames_B_total < n_frames")
                continue
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            self.train_data_B.append(data_B[:, start_B:end_B])
            self.dataset_length += 1

        self.train_data_A = np.array(self.train_data_A)
        self.train_data_B = np.array(self.train_data_B)
        print("--- dataset length --", self.train_data_A.shape, self.train_data_B.shape)

    def __getitem__(self, index):
        return self.train_data_A[index], self.train_data_B[index]

    def __len__(self):
        # return min(len(self.datasetA), len(self.datasetB))
        # return max(len(self.datasetA), len(self.datasetB))
        return self.dataset_length


if __name__ == '__main__':
    trainA = np.random.randn(162, 24, 554)
    trainB = np.random.randn(158, 24, 554)
    dataset = trainingDataset(trainA, trainB)
    trainLoader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2,
                                              shuffle=True)
    for epoch in range(10):
        for i, (trainA, trainB) in enumerate(trainLoader):
            print(trainA.shape, trainB.shape)
