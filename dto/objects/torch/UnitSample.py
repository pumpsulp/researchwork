from dataclasses import dataclass, field

from PIL.Image import Image


@dataclass
class UnitSample:
    image: Image = field(default_factory=Image)
    label: int = field(default_factory=int)
    