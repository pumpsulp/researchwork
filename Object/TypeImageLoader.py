from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class TypeImageLoader(ABC):
    """Реализует загрузку изображений разных форматов"""
    
    @abstractmethod
    def load_image(self, path: Path) -> Image.Image:
        raise NotImplementedError("Subclasses of TypeImageLoader should implement method load_image.")
    
    @abstractmethod
    def load_images(self, path: Path) -> list[Image.Image]:
        raise NotImplementedError("Subclasses of TypeImageLoader should implement method load_images.")


class PngImageLoader(TypeImageLoader):
    def load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert('RGB')
    
    def load_images(self, path: Path) -> list[Image.Image]:
        return [self.load_image(image) for image in path.glob('*.png')]
