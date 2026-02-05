# src/entity/data_ingestion_entity.py
from dataclasses import dataclass
import tensorflow as tf
from typing import List

@dataclass
class DataIngestionArtifact:
    train_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: List[str]
    num_classes: int
