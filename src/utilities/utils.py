import yaml
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from src.utilities.logger import app_logger

from src.constants.paths import GENERAL_CONFIG_FILE
from src.constants.config_keys import DATA_CONFIG

logger = app_logger(__name__)
load_dotenv()

def get_logger(name: str):
    """
    Returns a module-specific logger.
    Logger creation is deferred to runtime to ensure
    RUN_ID is already set.
    """
    return app_logger(name)


def data_downloader(kaggle_uri: str) -> None:
    config = load_config(GENERAL_CONFIG_FILE)
    RAW_DATA_PATH = Path(config[DATA_CONFIG]["raw_data_dir"])
    if not RAW_DATA_PATH.exists():
        raise EnvironmentError(
            f"Raw data directory {RAW_DATA_PATH} does not exist"
        )
    
    logger.info("Downloading data from kaggle")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_uri, "-p", str(RAW_DATA_PATH)],
        check=True
    )

    zip_files = list(RAW_DATA_PATH.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("No zip file found after download")

    zip_path = zip_files[0]

    logger.info("Unzipping data")
    subprocess.run(["unzip", "-o", str(zip_path), "-d", str(RAW_DATA_PATH)], check=True)

    logger.info("Removing zip file")
    zip_path.unlink()

    logger.info("Data downloaded successfully")


def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")

    return config