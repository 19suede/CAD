import h5py
import numpy as np
import torch
import glob
from scipy.io import loadmat
import os

from torch.utils.data import Dataset, DataLoader


# Define the dataset
class PansharpeningDataset(Dataset):
    def __init__(self, h5_file_path, norm, dr):
        self.h5_file_path = h5_file_path
        with h5py.File(h5_file_path, 'r') as file:
            lms = file['lms'][:]
            lms = np.array(lms, dtype=np.float32)
            self.lms_data = torch.from_numpy(lms)

            ms = file['ms'][:]
            ms = np.array(ms, dtype=np.float32)
            self.ms_data = torch.from_numpy(ms).to(torch.float32)

            pan = file['pan'][:]
            pan = np.array(pan, dtype=np.float32)
            self.pan_data = torch.from_numpy(pan).to(torch.float32)

            if norm == True:
                self.lms_data = self.lms_data / dr
                self.ms_data = self.ms_data / dr
                self.pan_data = self.pan_data / dr

            if file.__contains__('gt'):
                gt = file['gt'][:]
                gt = np.array(gt, dtype=np.float32)
                self.gt_data = torch.from_numpy(gt).to(torch.float32)
                if norm == True:
                    self.gt_data = self.gt_data / dr

    def __len__(self):
        return self.pan_data.shape[0]

    def __getitem__(self, idx):
        if hasattr(self, 'gt_data'):
            gt = self.gt_data[idx].float()
        else:
            gt = torch.tensor(0.0)

        pan = self.pan_data[idx].float()
        lms = self.lms_data[idx].float()
        ms = self.ms_data[idx].float()
        

        sample = {'lms': lms, 'ms': ms, 'pan': pan, 'gt': gt}

        return sample

