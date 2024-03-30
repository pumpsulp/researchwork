from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from objects.ObjectStorage import ObjectStorage, Photo, Object
from objects.TypeImageLoader import TypeImageLoader


class ObjectsCreator(ABC):
    """Различные реализации создания объектов"""
    
    @abstractmethod
    def create_objects(self, path: Path):
        raise NotImplementedError("Subclasses of ObjectCreator should implement method create_object.")


@dataclass
class CsvObjectsCreator(ObjectsCreator):
    """Один из способов создания объектов для интерполяции для следующей файловой структуры:
    
    ../путь к данным
    ________|-first_annotation.csv
    ________|...
    ________|-last_annotation.csv
    ________|-first_image_folder
    ____________|...
    ________|...
    ________|-last_image_folder
    ____________|...
    
    CSV аннотация должна содержать 2 обязательных столбца: Image, Class
    Image должен содержать имена файлов в папках (str),
    Class должен содержать имена классов (аннотация)
    """
    
    image_loader: TypeImageLoader
    
    @staticmethod
    def load_csv(path: Path) -> list[pd.DataFrame]:
        return [pd.read_csv(csv_path) for csv_path in path.rglob('*.csv')]
    
    def create_objects(self, path: Path) -> ObjectStorage:
        # TODO: нужен рефактор
        
        object_storage = ObjectStorage()
        
        cls_images = {}
        
        for csv in path.glob('*.csv'):
            df = pd.read_csv(csv)
            
            for index, row in df.iterrows():
                filename = row['Image']
                # загрузка изображения
                image = self.image_loader.load_image(Path(path / csv.stem / filename))
                # получаем объект Photo
                photo = Photo(matrix=np.array(image))
                cls = row['Class']
                
                if cls not in cls_images.keys():
                    cls_images[cls] = [photo]
                    # cls_images[cls] = cls_images[cls].add(photo)
                
                else:
                    cls_images[cls].append(photo)
        
        for obj in [Object(name=cls, photos=photos) for cls, photos in cls_images.items()]:
            object_storage.objects.add(obj)
        
        return object_storage
    
    @dataclass
    class FolderObjectsCreator(ObjectsCreator):
        """Один из способов создания объектов для интерполяции для следующей файловой структуры:
            ../путь к данным
            ________|-first_class_folder
            ____________|...
            ________|...
            ________|-last_class_folder
            ____________|...

            Названия папок являются именами классов, в каждой из папок лежат изображения соответствующих классов.
            """
        
        image_loader: TypeImageLoader
        
        def create_objects(self, path: Path) -> ObjectStorage:
            
            object_storage = ObjectStorage()
            
            for directory in path.iterdir():
                if directory.is_dir():
                    continue
                
                cls_name = directory.stem
                photos = []
                images = self.image_loader.load_images(directory)
                
                for image in images:
                    photos.append(Photo(matrix=np.array(image)))
                
                obj = Object(name=cls_name, photos=photos)
                
                object_storage.objects.add(obj)
            
            return object_storage
        