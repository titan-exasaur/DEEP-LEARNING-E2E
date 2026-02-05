import tensorflow as tf
from src.utilities.logger import app_logger
from src.utilities.utils import load_config

logger = app_logger(__name__)

class DataIngestion:
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
        Initializes the DataIngestion by reading dataset-related parameters
        from `model_parameters.yaml`.

        Configuration keys used:
        - TRAIN_DIR: Path to training data directory
        - TEST_DIR: Path to test data directory
        - IMG_SIZE: Target image size (height, width)
        - BATCH_SIZE: Number of samples per batch
        - SEED: Random seed for reproducibility
        """
        config = load_config("configs/model_parameters.yaml")
        data_config = config["data_config"]

        self.TRAIN_DIR = data_config["TRAIN_DIR"]
        self.TEST_DIR = data_config["TEST_DIR"]
        self.IMG_SIZE = tuple(data_config["IMG_SIZE"])
        self.BATCH_SIZE = data_config["BATCH_SIZE"]
        self.SEED = data_config["SEED"]

    def load(self):
        """
        Loads raw training and test datasets from disk and extracts
        dataset-level metadata.

        No transformations or preprocessing are applied at this stage.

        Returns:
            train_ds (tf.data.Dataset):
                Raw training dataset with images resized to IMG_SIZE.

            test_ds (tf.data.Dataset):
                Raw test dataset with images resized to IMG_SIZE.

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

        return train_ds, test_ds, class_names, num_classes
