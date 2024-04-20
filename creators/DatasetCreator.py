from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from objects.ObjectDataset import DatasetElement, ObjectDataset
from objects.ObjectStorage import ObjectStorage
from utils.DataSplit import DataSplit, ObjectDataSplit

DatasetWithLoader = namedtuple('DatasetWithLoader', ['dataset', 'dataloader'])


@dataclass
class DatasetCreator(ABC):
    """Реализует различные варианты создания набора данных (датасета)"""
    
    @abstractmethod
    def create_dataset(self, data) -> Dataset:
        raise NotImplementedError("Subclasses of TypeDataset should implement method create_dataset.")


@dataclass
class ObjectDatasetCreator(DatasetCreator):
    """Реализует создание набора данных из ObjectStorage"""
    batch_size: int
    splitter: ObjectDataSplit
    
    def create_dataset(self, object_storage: ObjectStorage) -> dict[str: DatasetWithLoader]:
        data = []
        
        labels = {cls: label for label, cls in enumerate(object_storage.get_objects_classes())}
        
        for obj in object_storage.objects:
            for photo in obj.photos:
                image = photo.get_image()
                data.append(DatasetElement(Image=image, Label=labels[obj.name]))
        
        # Разделение выборки на train, val (optional), test
        split_dataset = self.splitter.split(ObjectDataset(data=data))
        
        if self.splitter.valid_size is not None:
            abbr = ['train', 'val', 'test']
        else:
            abbr = ['train', 'test']
        
        dataset = {abbr: DatasetWithLoader(data, DataLoader(data, batch_size=self.batch_size))
                   for abbr, data in zip(abbr, split_dataset)}
        
        return dataset
