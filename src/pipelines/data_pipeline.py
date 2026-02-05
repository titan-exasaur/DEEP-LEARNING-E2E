from src.utilities.logger import app_logger, get_logger
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.data.data_validation import DataValidation

logger = get_logger(__name__)

class DataPipeline:
    """
    Orchestrates data ingestion and preprocessing workflow.
    """

    def run(self):
        logger.info("Starting data pipeline")

        data_validator = DataValidation()
        data_validator.validate()

        data_loader = DataIngestion()
        train_ds, test_ds, class_names, num_classes = data_loader.load()

        logger.info(f"Train batches: {len(train_ds)}")
        logger.info(f"Test batches: {len(test_ds)}")
        logger.info(f"Classes: {class_names}")
        logger.info(f"Num classes: {num_classes}")

        data_preprocessor = DataPreprocessing()
        train_ds, test_ds = data_preprocessor.process(train_ds, test_ds)

        logger.info("Data pipeline completed successfully")

        return train_ds, test_ds, class_names, num_classes
