import os
import re
import glob

from natsort import natsorted
from itertools import product

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from .km_config import KMFlow2D_DatasetConfig

from physics.pde import compute_time_derivative

class KMFlow2D_Dataset(Dataset):
    def __init__(self, config: KMFlow2D_DatasetConfig):

        self.root_dir = config.root_dir
        self.stats_dir = config.stat_dir

        self.list_re = config.list_re
        self.list_fn = config.list_fn

        self.using_vorticity = config.using_vorticity
        self.using_velocity = config.using_velocity
        
        self.size = config.size

        self.prepare_data()
        self.compute_stats()
        
    def prepare_data(self):

        assert self.using_velocity ^ self.using_vorticity, "Only one of using_vorticity or using_velocity should be True"
        file_name = "vorticity" if self.using_vorticity else "velocity"

        self.dataset = []
        self.meta_info = []

        for re, fn in product(self.list_re, self.list_fn):
            pattern = os.path.join(self.root_dir, f"size{self.size}", f"kf_2d_re{int(re)}", f"fn{int(fn)}", "seed*", file_name, "sol_t*_step*.npy")
            matched = natsorted(glob.glob(pattern))

            if len(matched) == 0:
                print(f"[Warning] No files found for pattern: {pattern}")
                continue
            
            self.dataset.append(matched)
            self.meta_info.append({'re': re, 'fn': fn})
        
        print(f"Size of dataset: {len(self.dataset)} cases.")
        print(f"Each case has {len(self.dataset[0])} samples.")

    def compute_stats(self):
        
        w_stats_dir = self.stats_dir if self.stats_dir else os.path.join(self.root_dir, f"size{self.size}", "stats.npz")
        dw_stats_dir = w_stats_dir.replace(".npz", "_deriv.npz")
        
        if os.path.exists(w_stats_dir) and os.path.exists(dw_stats_dir):
            w_stats = np.load(w_stats_dir)
            self.w_mean = w_stats['mean']
            self.w_std = w_stats['std']
            print(f"Loaded vorticity mean: {self.w_mean}, vorticity std: {self.w_std}")

            dw_stats = np.load(dw_stats_dir)
            self.dw_mean = dw_stats['mean']
            self.dw_std = dw_stats['std']
            print(f"Loaded vorticity derivative mean: {self.dw_mean}, vorticity derivative std: {self.dw_std}")
            return

        # Input data should be normalized to zero mean and unit variance.
        self.w_scaler = StandardScaler()
        self.dw_scaler = StandardScaler()
        length = len(self.dataset[0])

        for j, lst in enumerate(self.dataset):
            for i in range(1, length - 1, 4): # no need to be too precise,
                data = np.load(lst[i])
                deriv = compute_time_derivative(torch.from_numpy(data).float().unsqueeze(1), 
                                                torch.full((data.shape[0], 1), self.meta_info[j]['re'], dtype=torch.float32), 
                                                torch.full((data.shape[0], 1), self.meta_info[j]['fn'], dtype=torch.float32)
                                                ) # derivative computed on HR grid
                data = data.reshape(-1, 1)
                self.w_scaler.partial_fit(data)
                deriv = deriv.reshape(-1, 1)
                self.dw_scaler.partial_fit(deriv)

                del data, deriv
        
        self.w_mean = self.w_scaler.mean_
        self.w_std = self.w_scaler.scale_

        self.dw_mean = self.dw_scaler.mean_
        self.dw_std = self.dw_scaler.scale_

        print(f"Mean: {self.w_mean}, Std: {self.w_std}")
        print(f"Derivative Mean: {self.dw_mean}, Derivative Std: {self.dw_std}")

        np.savez(w_stats_dir, mean=self.w_mean, std=self.w_std)
        np.savez(dw_stats_dir, mean=self.dw_mean, std=self.dw_std)

    def __len__(self):
        return len(self.dataset[0]) * len(self.dataset)
    
    def __getitem__(self, index):
        dataset_idx = index // len(self.dataset[0])
        sample_idx = index % len(self.dataset[0])

        file_path = self.dataset[dataset_idx][sample_idx]
        data = np.load(file_path)

        data = torch.from_numpy(data).float()

        re_val = float(self.meta_info[dataset_idx]["re"])
        fn_val = float(self.meta_info[dataset_idx]["fn"])

        meta = {
            "re": torch.tensor([re_val], dtype=torch.float32),  # [1]
            "fn": torch.tensor([fn_val], dtype=torch.float32),  # [1]
        }

        return {
            "data": data,
            "meta": meta
        }