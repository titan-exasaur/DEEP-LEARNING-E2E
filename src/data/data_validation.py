from pathlib import Path
from src.utilities.logger import app_logger
from src.utilities.utils import load_config, get_logger

logger = get_logger(__name__)

class DataValidation:
    """
    Performs lightweight structural validation on image datasets.

    This module intentionally avoids heavy data validation and focuses on:
    - directory presence
    - class consistency
    - non-empty datasets
    """

    def __init__(self):
        config = load_config("configs/model_parameters.yaml")
        data_cfg = config["data_config"]

        self.train_dir = Path(data_cfg["TRAIN_DIR"])
        self.test_dir = Path(data_cfg["TEST_DIR"])

    def validate(self):
        logger.info("Running data validation checks")

        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train dir not found: {self.train_dir}")

        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test dir not found: {self.test_dir}")

        train_classes = {d.name for d in self.train_dir.iterdir() if d.is_dir()}
        test_classes = {d.name for d in self.test_dir.iterdir() if d.is_dir()}

        if train_classes != test_classes:
            raise ValueError(
                f"Train/Test class mismatch: "
                f"{train_classes} vs {test_classes}"
            )

        for cls in train_classes:
            if not any((self.train_dir / cls).iterdir()):
                raise ValueError(f"Empty class folder: {cls}")

        logger.info("Data validation passed")
