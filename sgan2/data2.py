import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 新しく追加する「省メモリ版」のデータ読み込みクラス
class MmapDataset(Dataset):
    def __init__(self, npy_path, meta_path=None):
        # mmap_mode='r' を使って、ディスク上のデータにインデックスだけ貼る（メモリを節約）
        self.data_x = np.load(npy_path, mmap_mode='r')
        
        if meta_path is not None:
            df = pd.read_csv(meta_path, header=None)
            self.data_y = df.iloc[:, 1].astype(int).values
        else:
            # ラベルなしデータの場合はすべて -1 を入れる
            self.data_y = np.full((self.data_x.shape[0],), -1, dtype=np.int64)

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        # バッチサイズ分だけ呼ばれた時に、Tensorに変換して返す
        x = torch.from_numpy(self.data_x[idx].copy()).float()
        y = torch.tensor(self.data_y[idx], dtype=torch.long)
        return x, y

def make_dataset(npy_path, meta_path, device):
    # 省メモリなデータセットを返す
    return MmapDataset(npy_path, meta_path)