import numpy as np
from torch.utils.data import Dataset
import torch

class PDEPairDataset(Dataset):
    def __init__(
        self,
        low_data: np.ndarray,
        high_data: np.ndarray,
        mask_high_data: np.ndarray = None,
        x_dtype=torch.float32,
        mask_dtype="bool", 
    ):
        assert (high_data.ndim == 4 and low_data.ndim == 4) or (high_data.ndim == 5 and low_data.ndim == 5)

        self.low  = low_data
        self.high = high_data
        self.x_dtype = x_dtype

        if mask_high_data is not None:
            assert mask_high_data.shape == high_data.shape
            
        self.mh = mask_high_data
        self.mask_dtype = mask_dtype

    def __len__(self):
        return self.high.shape[0]

    def _to_tensor_x(self, arr):
        return torch.from_numpy(np.asarray(arr)).to(self.x_dtype)

    def _to_tensor_m(self, arr):
        if arr is not None:
            m = torch.from_numpy(np.asarray(arr))
            return m.bool()
        else:
            return None

    def __getitem__(self, idx):        
        xl = self._to_tensor_x(self.low[idx])
        xh = self._to_tensor_x(self.high[idx])
        mh = self._to_tensor_m(self.mh[idx]) 
        
        return xl, xh, mh
