from dataclasses import dataclass

from torchvision.transforms import Compose

from objects.torch.DataSample import DataSample
from objects.ObjectStorage import ObjectStorage
from objects.torch.SampleUnit import SampleUnit


@dataclass
class SampleCreator:
    """Реализует создание набора данных из ObjectStorage"""
    
    @classmethod
    def create(cls, object_storage: ObjectStorage,
               transform: Compose | None = None) -> DataSample:
        
        data = []
        
        labels = {cls: label for label, cls in enumerate(object_storage.get_objects_classes())}
        
        for obj in object_storage.objects:
            for photo in obj.photos:
                image = photo.get_image()
                data.append(SampleUnit(image, labels[obj.name]))
        
        data_sample = DataSample(data=data, transform=transform)
        
        return data_sample
