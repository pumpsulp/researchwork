from dataclasses import dataclass

from torch.utils.data import DataLoader

from objects.DataSample import DataSample


@dataclass
class DataLoaderCreator:
    """Создание Dataloader из DataSample"""
    @classmethod
    def create(cls, data: DataSample, batch_size: int) -> DataLoader:
        return DataLoader(dataset=data, batch_size=batch_size)