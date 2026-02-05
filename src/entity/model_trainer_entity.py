# src/entity/model_trainer_entity.py
from dataclasses import dataclass
import tensorflow as tf

@dataclass
class ModelTrainerArtifact:
    model: tf.keras.Model
    history: object
