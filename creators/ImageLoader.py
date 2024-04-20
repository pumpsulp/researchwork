from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class ImageLoader:
    """Реализует загрузку изображений разных форматов"""
    
    @classmethod
    def load_image(cls, path: Path) -> Image.Image:
        return Image.open(path).convert('RGB')
    
    @classmethod
    def load_images(cls, path: Path, img_format: str) -> list[Image.Image]:
        return [cls.load_image(image) for image in path.rglob(f'*.{img_format}')]
