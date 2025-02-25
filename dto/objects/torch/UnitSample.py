from dataclasses import dataclass

from PIL.Image import Image


@dataclass
class SampleUnit:
    image: Image
    label: int
    