from src.utilities.logger import app_logger
from src.data.data_ingestion import DataLoader

logger = app_logger(__name__)

class DataPipeline:
    """
    Orchestrates the data ingestion workflow.
    """

    def run(self):
        logger.info("Starting data pipeline")

        data_loader = DataLoader()
        train_ds, test_ds, class_names, num_classes = data_loader.load_data()

        logger.info(f"Train batches: {len(train_ds)}")
        logger.info(f"Test batches: {len(test_ds)}")
        logger.info(f"Classes: {class_names}")
        logger.info(f"Num classes: {num_classes}")

        logger.info("Data pipeline completed successfully")

        return train_ds, test_ds, class_names, num_classes
