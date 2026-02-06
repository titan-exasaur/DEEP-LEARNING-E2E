import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.constants.training import DEFAULT_INPUT_SHAPE
from src.entity.model_trainer_entity import ModelTrainerArtifact


class ModelInference:
    """
    Handles single-image inference using a trained model.
    """

    def predict(
        self,
        model_artifact: ModelTrainerArtifact,
        class_names: list[str],
        img_path: Path
    ) -> tuple[str, float]:
        """
        Takes an image path and returns predicted class and confidence.
        """

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        if model_artifact.model is None:
            raise RuntimeError("Model not loaded")

        img = load_img(
            img_path,
            target_size=DEFAULT_INPUT_SHAPE[:2]
        )
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model_artifact.model.predict(img)
        idx = int(np.argmax(preds))

        return class_names[idx], float(preds[0][idx])
