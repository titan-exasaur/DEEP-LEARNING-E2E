import tensorflow as tf
from tensorflow.keras import layers
from src.utilities.logger import app_logger
from src.utilities.utils import load_config

logger = app_logger(__name__)

class DataLoader:
    """
    Handles loading and preprocessing of image datasets for training and evaluation.

    Responsibilities:
    - Read dataset configuration from YAML
    - Load training and test image datasets from directory structure
    - Apply normalization and performance optimizations
    - Expose class metadata required for model construction

    Expected directory structure:
        TRAIN_DIR/
            class_1/
            class_2/
            ...
        TEST_DIR/
            class_1/
            class_2/
            ...
    """

    def __init__(self):
        """
        Initializes the DataLoader by reading dataset-related parameters
        from `model_parameters.yaml`.

        Configuration keys used:
        - TRAIN_DIR: Path to training data directory
        - TEST_DIR: Path to test data directory
        - IMG_SIZE: Target image size (height, width)
        - BATCH_SIZE: Number of samples per batch
        - SEED: Random seed for reproducibility
        """
        self.config = load_config("configs/model_parameters.yaml")
        self.data_config = self.config["data_config"]

        self.TRAIN_DIR = self.data_config["TRAIN_DIR"]
        self.TEST_DIR = self.data_config["TEST_DIR"]
        self.IMG_SIZE = tuple(self.data_config["IMG_SIZE"])
        self.BATCH_SIZE = self.data_config["BATCH_SIZE"]
        self.SEED = self.data_config["SEED"]

        self.normalization = layers.Rescaling(1.0 / 255.0)

    def load_data(self):
        """
        Loads, preprocesses, and optimizes the training and test datasets.

        Processing steps:
        - Load images from directory using TensorFlow utilities
        - Resize images to configured IMG_SIZE
        - Normalize pixel values to [0, 1]
        - Enable parallel mapping and prefetching for performance

        Returns:
            train_ds (tf.data.Dataset):
                Batched and normalized training dataset.

            test_ds (tf.data.Dataset):
                Batched and normalized test dataset (no shuffling).

            class_names (List[str]):
                Ordered list of class labels inferred from directory names.

            num_classes (int):
                Total number of target classes.
        """
        logger.info("Loading training dataset")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.TRAIN_DIR,
            image_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            seed=self.SEED
        )
        class_names = train_ds.class_names
        num_classes = len(class_names)
        logger.info(f"Classes detected: {class_names}")

        logger.info("Loading test dataset")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.TEST_DIR,
            image_size=self.IMG_SIZE,
            batch_size=self.BATCH_SIZE,
            shuffle=False
        )

        train_ds = train_ds.map(
            lambda x, y: (self.normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        test_ds = test_ds.map(
            lambda x, y: (self.normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        logger.info("Data loaded successfully")

        return train_ds, test_ds, class_names, num_classes
