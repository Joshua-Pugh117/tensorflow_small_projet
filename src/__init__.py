# my_object_detector/__init__.py

from .detecte import load_model, detect
from .my_data import prepare_image, draw_on_image

__version__ = "0.1.0"
__all__ = [
    "load_model",
    "detect",
    "prepare_image",
    "draw_on_image",
    "detect_yolo"
]