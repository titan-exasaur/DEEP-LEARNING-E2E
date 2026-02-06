from src.utilities.utils import load_config, get_logger
from src.models.simple_cnn import SimpleCNN
from src.entity.data_ingestion_entity import DataIngestionArtifact
from src.entity.model_trainer_entity import ModelTrainerArtifact
from src.constants.paths import MODEL_PARAMS_FILE
from src.constants.config_keys import MODEL_CONFIG
from src.constants.training import DEFAULT_INPUT_SHAPE

logger = get_logger(__name__)

class ModelTrainer:
    """
    Handles model construction, compilation, and training.
    """

    def train(
        self,
        data_artifact: DataIngestionArtifact
    ) -> ModelTrainerArtifact:

        logger.info("Starting model training")

        config = load_config(MODEL_PARAMS_FILE)
        model_cfg = config[MODEL_CONFIG]

        model = SimpleCNN()
        model.build(
            input_shape=DEFAULT_INPUT_SHAPE,
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

        logger.info("Model training finished")

        return ModelTrainerArtifact(
            model=model.model,
            history=history
        )
