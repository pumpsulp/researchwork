from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from creators.torch.SampleCreator import SampleCreator
from creators.ObjectsCreator import ObjectsCreator, FromCsv


@dataclass
class Loader:
    train: DataLoader
    test: DataLoader
    valid: DataLoader = None


@dataclass
class LoaderCreator:
    object_creator: ObjectsCreator = FromCsv
    
    @classmethod
    def create(self, path: Path,
               test_size: float,
               valid_size: float | None,
               batch_size: int,
               transform: Compose | None,
               train_shuffle: bool = True) -> Loader:
        
        objs = self.object_creator.create_objects(path)
        
        data = SampleCreator.create(objs, transform)
        
        train, test = train_test_split(data, test_size=test_size)
        
        if valid_size is not None:
            train, valid = train_test_split(train, test_size=valid_size)
            return Loader(train=DataLoader(train, batch_size, train_shuffle),
                          test=DataLoader(test, batch_size),
                          valid=DataLoader(valid, batch_size))
        
        return Loader(train=DataLoader(train, batch_size, train_shuffle),
                      test=DataLoader(test, batch_size))
