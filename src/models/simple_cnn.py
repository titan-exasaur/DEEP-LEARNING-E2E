from tensorflow.keras import layers, models
from src.models.base_model import BaseModel
from src.utilities.utils import get_logger

logger = get_logger(__name__)

class SimpleCNN(BaseModel):
    """
    Simple Convolutional Neural Network for image classification.
    """

    def __init__(self):
        super().__init__()


    def build(self, input_shape: tuple[int, int, int], num_classes: int) -> None:
        """
        Builds the CNN architecture.
        """
        self.model = models.Sequential([
            layers.Conv2D(32, 3, activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax")
        ])

        logger.info("SimpleCNN architecture built")

    def compile(self, **kwargs) -> None:
        """
        Compiles the CNN model.
        """
        self.model.compile(**kwargs)
        logger.info("SimpleCNN compiled successfully")


