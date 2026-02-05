import tensorflow as tf
from tensorflow.keras import layers
from src.utilities.logger import app_logger
from src.utilities.utils import load_config

logger = app_logger(__name__)

class DataPreprocessing:
    """
    Applies preprocessing and performance optimizations to image datasets.

    Responsibilities:
    - Normalize pixel values
    - Apply optional data augmentation
    - Optimize input pipelines using parallel mapping and prefetching

    All preprocessing behavior is controlled via `model_parameters.yaml`.
    """

    def __init__(self):
        """
        Initializes preprocessing components based on configuration.

        Configuration keys used:
        - augmentation.enabled: Toggle data augmentation
        - augmentation.horizontal_flip
        - augmentation.rotation
        - augmentation.zoom
        """
        config = load_config("configs/model_parameters.yaml")
        aug_config = config.get("augmentation", {})

        self.normalization = layers.Rescaling(1.0 / 255.0)

        self.augmentation = None
        if aug_config.get("enabled", False):
            self.augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(aug_config.get("rotation", 0.1)),
                layers.RandomZoom(aug_config.get("zoom", 0.1))
            ])

    def process(self, train_ds, test_ds):
        """
        Applies preprocessing transformations to training and test datasets.

        Processing steps:
        - Normalize all images
        - Apply data augmentation to training data only
        - Enable parallel execution and prefetching

        Args:
            train_ds (tf.data.Dataset):
                Raw training dataset.

            test_ds (tf.data.Dataset):
                Raw test dataset.

        Returns:
            train_ds (tf.data.Dataset):
                Preprocessed and optimized training dataset.

            test_ds (tf.data.Dataset):
                Preprocessed and optimized test dataset.
        """
        logger.info("Applying data preprocessing")

        train_ds = train_ds.map(
            self._train_map,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        test_ds = test_ds.map(
            lambda x, y: (self.normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        logger.info("Data preprocessing complete")

        return train_ds, test_ds

    def _train_map(self, x, y):
        """
        Applies preprocessing to a single training batch.

        Augmentation is applied only if enabled.
        """
        x = self.normalization(x)
        if self.augmentation:
            x = self.augmentation(x)
        return x, y
