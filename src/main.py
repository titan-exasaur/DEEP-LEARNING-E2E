import os, uuid

# set RUN_ID first
os.environ["RUN_ID"] = uuid.uuid4().hex[:8]

from src.utilities.logger import app_logger
from src.pipelines.data_pipeline import DataPipeline

logger = app_logger(__name__)
logger.info(f"Logger initialized | run_id={os.getenv('RUN_ID')}")

def main():
    logger.info("Application started")

    data_pipeline = DataPipeline()
    data_pipeline.run()

    logger.info("Application finished successfully")

if __name__ == "__main__":
    main()
