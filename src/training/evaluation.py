import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from src.utilities.utils import load_config, get_logger
from src.entity.model_trainer_entity import ModelTrainerArtifact
from src.entity.data_ingestion_entity import DataIngestionArtifact
from src.constants.paths import MODEL_PARAMS_FILE, ARTIFACTS_DIR
from src.constants.config_keys import MODEL_CONFIG

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Handles model evaluation and metric persistence.
    """

    def evaluate(
        self,
        data_artifact: DataIngestionArtifact,
        model_artifact: ModelTrainerArtifact,
        run_id: str
    ) -> Path:

        logger.info("Model evaluation started")

        y_true, y_pred = [], []

        for images, labels in data_artifact.test_ds:
            preds = model_artifact.model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

        report = classification_report(
            y_true,
            y_pred,
            target_names=data_artifact.class_names,
            output_dict=True
        )

        df = pd.DataFrame(report).transpose()

        report_dir = Path(ARTIFACTS_DIR) / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / f"classification_report_{run_id}.csv"
        df.to_csv(report_path)

        logger.info(f"Classification report saved at {report_path}")

        return report_path

    def plot_training_curves(
        self,
        model_artifact: ModelTrainerArtifact,
        run_id: str
    ) -> Path:

        plots_dir = Path(ARTIFACTS_DIR) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plots_dir / f"training_curves_{run_id}.png"

        plt.figure(figsize=(10, 5))

        plt.plot(
            model_artifact.history.history["accuracy"],
            label="Train Accuracy"
        )
        plt.plot(
            model_artifact.history.history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.plot(
            model_artifact.history.history["loss"],
            label="Train Loss"
        )
        plt.plot(
            model_artifact.history.history["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Training Performance")
        plt.legend()
        plt.grid(True)

        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Training curves saved at {plot_path}")

        return plot_path
