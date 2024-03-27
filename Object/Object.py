from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from sortedcontainers import SortedSet


@dataclass(order=True, unsafe_hash=True)
class Photo:
    """Представляет собой фото"""
    matrix: np.ndarray = field(compare=False, hash=False, repr=False)
    
    def get_matrix(self):
        return self.matrix
    
    def get_image(self):
        return Image.fromarray(self.matrix)


@dataclass(order=True, unsafe_hash=True)
class Object:
    """Представляет собой объект"""
    name: str = field(default_factory=str, repr=True)
    photos: SortedSet[Photo] = field(default_factory=SortedSet, compare=False)
    
    def __len__(self):
        return len(self.photos)


@dataclass(order=True, unsafe_hash=True)
class ObjectStorage:
    """Представляет собой хранилище объектов"""
    
    objects: SortedSet[Object] = field(default_factory=SortedSet, compare=False, repr=False)
    
    def __len__(self):
        return len(self.objects)
