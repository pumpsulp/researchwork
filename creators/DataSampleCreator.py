from dataclasses import dataclass

from objects.DataSample import DataSample
from objects.ObjectStorage import ObjectStorage
from objects.SampleUnit import SampleUnit
from transform.Transform import Transform


@dataclass
class DataSampleCreator:
    """Реализует создание набора данных из ObjectStorage"""
    
    @classmethod
    def create(cls, object_storage: ObjectStorage,
               transform: Transform = None) -> DataSample:
        
        data = []
        
        labels = {cls: label for label, cls in enumerate(object_storage.get_objects_classes())}
        
        for obj in object_storage.objects:
            for photo in obj.photos:
                image = photo.get_image()
                data.append(SampleUnit(image, labels[obj.name]))
        
        data_sample = DataSample(data=data, transform=transform)
        
        return data_sample
