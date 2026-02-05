import os
from datetime import datetime
os.environ["RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

from src.utilities.utils import get_logger
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.train_pipeline import TrainingPipeline

logger = get_logger(__name__)

def main():
    logger.info("Application started")

    data_pipeline = DataPipeline()
    data_artifact = data_pipeline.run()

    training_pipeline = TrainingPipeline()
    trainer_artifact = training_pipeline.run(data_artifact)

    logger.info("Application finished successfully")

if __name__ == "__main__":
    main()
