from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing
from src.data.data_validation import DataValidation
from src.entity.data_ingestion_entity import DataIngestionArtifact
from src.utilities.utils import get_logger

logger = get_logger(__name__)

class DataPipeline:
    """
    Orchestrates data ingestion and preprocessing workflow.
    """

    def run(self) -> DataIngestionArtifact:
        logger.info("Starting data pipeline")

        DataValidation().validate()

        ingestion = DataIngestion()
        train_ds, test_ds, class_names, num_classes = ingestion.load()

        logger.info(f"Train batches: {len(train_ds)}")
        logger.info(f"Test batches: {len(test_ds)}")
        logger.info(f"Classes: {class_names}")
        logger.info(f"Num classes: {num_classes}")

        data_preprocessor = DataPreprocessing()
        train_ds, test_ds = data_preprocessor.process(train_ds, test_ds)

        logger.info("Data pipeline completed successfully")

        return DataIngestionArtifact(
            train_ds=train_ds,
            test_ds=test_ds,
            class_names=class_names,
            num_classes=num_classes
        )