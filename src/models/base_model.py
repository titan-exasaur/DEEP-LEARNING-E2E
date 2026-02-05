from abc import ABC, abstractmethod
from typing import Any
import tensorflow as tf

class BaseModel(ABC):
    """
    Abstract base class defining the contract for all models.

    All concrete model implementations must:
    - build a tf.keras.Model
    - compile it
    - support training and inference
    """

    def __init__(self):
        self.model: tf.keras.Model | None = None

    @abstractmethod
    def build(self, input_shape: tuple[int, int, int], num_classes: int) -> None:
        """Builds the keras model architecture."""

    @abstractmethod
    def compile(self, **kwargs) -> None:
        """Compiles the keras model."""

    def fit(self, *args, **kwargs):
        """Trains the model."""
        if self.model is None:
            raise RuntimeError("Model must be built before training")
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Runs inference."""
        if self.model is None:
            raise RuntimeError("Model must be built before prediction")
        return self.model.predict(*args, **kwargs)

    def save(self, path: str) -> None:
        """Saves the model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> tf.keras.Model:
        """Loads a saved model."""
        return tf.keras.models.load_model(path)
