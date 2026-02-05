from src.utilities.logger import app_logger
from src.pipelines.data_pipeline import DataPipeline

logger = app_logger(__name__)

def main():
    logger.info("Application started")

    data_pipeline = DataPipeline()
    data_pipeline.run()

    logger.info("Application finished successfully")

if __name__ == "__main__":
    main()