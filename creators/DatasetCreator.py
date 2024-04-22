from dataclasses import dataclass

from torch.utils.data.dataloader import DataLoader

from creators.DataSampleCreator import DataSampleCreator
from objects.Dataset import Dataset
from objects.DatasetUnit import DatasetUnit
from objects.ObjectStorage import ObjectStorage
from transform.Transform import Transform
from utils.DataSplit import ObjectDataSplit


# @dataclass
# class DatasetCreator(ABC):
#     """Реализует различные варианты создания набора данных (датасета)"""
#
#     @abstractmethod
#     def create_dataset(self, *args, **kwargs) -> Dataset:
#         raise NotImplementedError("Subclasses of TypeDataset should implement method create_dataset.")


@dataclass
class DatasetCreator:
    """Реализует создание набора данных из ObjectStorage"""
    
    @classmethod
    def create(cls, data: ObjectStorage,
               batch_size: int,
               test_size: float,
               valid_size: float = None,
               transform: Transform = None) -> Dataset:
        
        data_sample = DataSampleCreator.create(data, transform=transform)
        
        splitter = ObjectDataSplit(test_size=test_size, valid_size=valid_size)
        
        # Разделение выборки на train, val (optional), test
        split_data = splitter.split(data_sample)
        
        units = [DatasetUnit(data, DataLoader(data, batch_size=batch_size)) for data in split_data]
        
        return Dataset(*units)
