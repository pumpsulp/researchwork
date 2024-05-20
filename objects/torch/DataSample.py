from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from objects.torch.SampleUnit import SampleUnit
from torchvision.transforms import Compose


@dataclass
class DataSample(Dataset):
    data: list[SampleUnit] = field(default_factory=list)
    transform: Compose = field(default=None)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        sample = self.data[idx]
        img, label = sample.image, sample.label
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
