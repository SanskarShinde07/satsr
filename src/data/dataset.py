import numpy as np
import torch
import rasterio as rio
import tacoreader
from torch.utils.data import Dataset


class Sen2NaipDataset(Dataset):
    def __init__(self, n_samples=4000, divisor=3000.0):
        self.ds = tacoreader.load("tacofoundation:sen2naipv2-unet")
        self.n = min(n_samples, len(self.ds))
        self.div = divisor

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        try:
            row = self.ds.read(i)
            with rio.open(row.read(0)) as s:
                lr = s.read([1, 2, 3]).astype(np.float32)
            with rio.open(row.read(1)) as s:
                hr = s.read([1, 2, 3]).astype(np.float32)
        except Exception:
            return self[(i + 1) % self.n]      # skip bad samples

        lr = np.clip(lr / self.div, 0, 1)
        hr = np.clip(hr / self.div, 0, 1)
        return torch.from_numpy(lr), torch.from_numpy(hr)