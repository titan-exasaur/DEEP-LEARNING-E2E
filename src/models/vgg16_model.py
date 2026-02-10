from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from src.models.base_model import BaseModel
from src.utilities.utils import load_config, get_logger
from src.constants.paths import MODEL_PARAMS_FILE

logger = get_logger(__name__)


class VGG16Model(BaseModel):
    """
    Transfer Learning using pretrained VGG16 backbone.
    """

    def __init__(self):
        super().__init__()
        self.config = load_config(MODEL_PARAMS_FILE)
        self.tl_config = self.config.get("transfer_learning", {})

    def build(self, input_shape: tuple[int, int, int],
              num_classes: int) -> None:
        """
        Builds VGG-16 based transfer learning model
        """

        # load pretrained VGG16 without top
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

        # freeze entire base model
        base_model.trainable = not self.tl_config.get("freeze_base_model", True)

        unfreeze_n = self.tl_config.get("unfreeze_last_n_layers", 0)
        if unfreeze_n > 0:
            for layer in base_model.layers[-unfreeze_n:]:
                layer.trainable = True
            
        # custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs=base_model.input,
                                  outputs=output)
        
        logger.info("VGG16 transfer learning model built")

    def compile(self, **kwargs) -> None:
        """
        Compiles the VGG16 model
        """
        self.model.compile(**kwargs)
        logger.info("VGG16 model compiled successfully")