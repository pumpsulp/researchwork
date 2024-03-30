from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch.utils.data import Dataset

from objects.DataSplit import ObjectDataSplit, DataSplit
from objects.ObjectStorage import ObjectStorage
from objects.ObjectDataset import DatasetElement, ObjectDataset


@dataclass
class DatasetCreator(ABC):
    """Реализует различные варианты создания набора данных (датасета)"""
    
    @abstractmethod
    def create_dataset(self, data) -> Dataset:
        raise NotImplementedError("Subclasses of TypeDataset should implement method create_dataset.")


@dataclass
class ObjectDatasetCreator(DatasetCreator):
    """Реализует создание набора данных из ObjectStorage"""
    splitter: DataSplit = field(default=None)
    
    def create_dataset(self, object_storage: ObjectStorage) -> Dataset | tuple[Dataset]:
        data = []
        
        labels = {cls: label for label, cls in enumerate(object_storage.get_objects_classes())}
        
        for obj in object_storage.objects:
            for photo in obj.photos:
                image = photo.get_image()
                data.append(DatasetElement(Image=image, Label=labels[obj.name]))
        
        dataset = ObjectDataset(data=data)
        
        if self.splitter is not None:
            dataset = self.splitter.split(dataset)
        
        return dataset
