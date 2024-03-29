from collections import namedtuple
from dataclasses import dataclass, field

from torch.utils.data import Dataset

DatasetElement = namedtuple(typename='DatasetElement', field_names=['Image', 'Label'])


@dataclass
class ObjectDataset(Dataset):
    data: list[DatasetElement] = field(default_factory=list)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
