"""Selective Few-Random-Shot Augmentation."""

from .generation import generate_few_random_shot
from .selection import SelectionResult, select_augmented_texts

__all__ = [
    "SelectionResult",
    "generate_few_random_shot",
    "select_augmented_texts",
]

__version__ = "0.1.0"
