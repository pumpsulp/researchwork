from dataclasses import dataclass

from objects.DatasetUnit import DatasetUnit
from objects.DataSample import DataSample


@dataclass
class Dataset:
    train: DatasetUnit
    test: DatasetUnit
    valid: DatasetUnit = None
    
    def __len__(self) -> int:
        return sum(len(unit.data) for unit in (self.train, self.test, self.valid))
    