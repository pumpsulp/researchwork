from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from objects.SampleUnit import SampleUnit
from transform.Transform import Transform


@dataclass
class DataSample(Dataset):
    data: list[SampleUnit] = field(default_factory=list)
    transform: Transform = field(default=None)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
