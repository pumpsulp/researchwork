from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision.models.detection import transform

from objects.SampleUnit import SampleUnit


class Transform(ABC):
    """Представляет собой абстрактный класс трансформации изображения"""
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of Transform should implement (redefine) dunder method __call__")


class ToTensor(Transform):
    def __call__(self, sample: SampleUnit):
        image, label = sample.image, sample.label
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        
        image = np.asarray(image).transpose((2, 0, 1))
        
        return SampleUnit(torch.tensor(image), torch.tensor(label))


class Normalize(Transform):
    def __call__(self, sample: SampleUnit):
        image = sample.image
        image = np.asarray(image) / 255
        return SampleUnit(image, sample.label)
        
    
        

# class Rescale(Transform):
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample.image, sample.label
#
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#
#         new_h, new_w = int(new_h), int(new_w)
#
#         img = transform.resize(image, (new_h, new_w))
#
#         return SampleUnit(img, label)
    
    
