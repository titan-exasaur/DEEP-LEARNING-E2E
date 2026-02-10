import os
from pathlib import Path
from src.utilities.utils import load_config, get_logger
from src.models.simple_cnn import SimpleCNN
from src.entity.data_ingestion_entity import DataIngestionArtifact
from src.entity.model_trainer_entity import ModelTrainerArtifact
from src.constants.paths import MODEL_PARAMS_FILE, MODEL_DIR
from src.constants.config_keys import MODEL_CONFIG
from src.constants.training import DEFAULT_INPUT_SHAPE

import mlflow
import mlflow.tensorflow
mlflow.set_experiment("brain_tumor_classification")

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = get_logger(__name__)

class ModelTrainer:
    """
    Handles model construction, compilation, training, and persistence.
    """

    def __init__(self):
        self.config = load_config(MODEL_PARAMS_FILE)

    def train(
        self,
        data_artifact: DataIngestionArtifact
    ) -> ModelTrainerArtifact:

        logger.info("Starting model training")

        model_cfg = self.config[MODEL_CONFIG]

        run_id = os.getenv("RUN_ID")
        if not run_id:
            raise EnvironmentError("RUN_ID not found")
        
        mlflow.start_run(run_name=run_id)

        callbacks, best_model_path = self._get_callbacks(run_id)

        model_type = self.config.get("model_type", "simple_cnn")

        if model_type == "vgg16":
            from src.models.vgg16_model import VGG16Model
            model = VGG16Model()
        else:
            model = SimpleCNN()

        model.build(
            input_shape=DEFAULT_INPUT_SHAPE,
            num_classes=data_artifact.num_classes
        )

        # model parameters
        mlflow.log_params({
            "optimizer": model_cfg["optimizer"],
            "loss": model_cfg["loss"],
            "epochs": model_cfg["epochs"]
        })

        # data parameters
        mlflow.log_params({
            "batch_size": self.config["data_config"]["BATCH_SIZE"],
            "img_size": self.config["data_config"]["IMG_SIZE"]
        })


        model.compile(
            optimizer=model_cfg["optimizer"],
            loss=model_cfg["loss"],
            metrics=model_cfg["metrics"]
        )

        history = model.fit(
            data_artifact.train_ds,
            validation_data=data_artifact.test_ds,
            epochs=model_cfg["epochs"],
            callbacks=callbacks
        )

        for epoch, (acc, val_acc, loss, val_loss) in enumerate(
            zip(
                history.history["accuracy"],
                history.history["val_accuracy"],
                history.history["loss"],
                history.history["val_loss"]
            )
        ):
            mlflow.log_metrics({
                "train_accuracy": acc,
                "val_accuracy": val_acc,
                "train_loss": loss,
                "val_loss": val_loss
            }, step=epoch)

        mlflow.tensorflow.log_model(
            model=model.model,
            artifact_path="model"
        )


        logger.info(f"Model saved at {best_model_path}")

        mlflow.end_run()

        return ModelTrainerArtifact(
            model=model.model,
            history=history,
            model_path=str(best_model_path)
        )

    def _get_callbacks(self, run_id: str):
        callbacks = []

        cb_cfg = self.config.get("callbacks", {})

        if "early_stopping" in cb_cfg:
            es_cfg = cb_cfg["early_stopping"]
            callbacks.append(
                EarlyStopping(
                    monitor=es_cfg["monitor"],
                    patience=es_cfg["patience"],
                    restore_best_weights=True
                )
            )

        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        best_model_path = model_dir / f"{run_id}.keras"

        callbacks.append(
            ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False
            )
        )

        return callbacks, best_model_path
