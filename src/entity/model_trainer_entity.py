# src/entity/model_trainer_entity.py
import tensorflow as tf
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelTrainerArtifact:
    model: tf.keras.Model
    history: object
    model_path: Optional[str] = None
