import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

def load_agevecimg(path):
    x = np.load(path)
    return x

def get_label(path):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 1].astype(int).values

def make_dataset(npy_path, meta_path, device):
    x = torch.from_numpy(load_agevecimg(npy_path)).float().cuda(device)
    if meta_path is not None:
        y = torch.from_numpy(get_label(meta_path)).cuda(device)
    else:
        y = torch.full((x.size(0),), -1, dtype=torch.long).cuda(device)
    return TensorDataset(x, y)