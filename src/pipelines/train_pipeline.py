from src.utilities.utils import load_config, get_logger
from src.models.simple_cnn import SimpleCNN
from src.entity.data_ingestion_entity import DataIngestionArtifact
from src.entity.model_trainer_entity import ModelTrainerArtifact

logger = get_logger(__name__)

class TrainingPipeline:
    """
    Orchestrates model training.
    """

    def run(
        self,
        data_artifact: DataIngestionArtifact
    ) -> ModelTrainerArtifact:

        logger.info("Starting training pipeline")

        config = load_config("configs/model_parameters.yaml")
        model_cfg = config["model_config"]

        model = SimpleCNN()
        model.build(
            input_shape=(224, 224, 3),
            num_classes=data_artifact.num_classes
        )

        model.compile(
            optimizer=model_cfg["optimizer"],
            loss=model_cfg["loss"],
            metrics=model_cfg["metrics"]
        )

        history = model.fit(
            data_artifact.train_ds,
            validation_data=data_artifact.test_ds,
            epochs=model_cfg["epochs"]
        )

        logger.info("Training completed successfully")

        return ModelTrainerArtifact(
            model=model.model,
            history=history
        )
