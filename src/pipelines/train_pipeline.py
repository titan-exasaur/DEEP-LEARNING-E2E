import os
from src.utilities.utils import get_logger
from src.training.model_trainer import ModelTrainer
from src.training.evaluation import ModelEvaluator
from src.entity.data_ingestion_entity import DataIngestionArtifact
from src.entity.model_trainer_entity import ModelTrainerArtifact

logger = get_logger(__name__)

class TrainingPipeline:
    """
    Orchestrates the training and evaluation workflow.
    """

    def run(
        self,
        data_artifact: DataIngestionArtifact
    ) -> ModelTrainerArtifact:

        logger.info("Starting training pipeline")

        run_id = os.getenv("RUN_ID")
        if not run_id:
            raise RuntimeError("RUN_ID not set in environment")

        trainer = ModelTrainer()
        trainer_artifact = trainer.train(data_artifact)

        evaluator = ModelEvaluator()
        evaluator.plot_training_curves(trainer_artifact, run_id)
        evaluator.evaluate(data_artifact, trainer_artifact, run_id)

        logger.info("Training pipeline completed")

        return trainer_artifact
