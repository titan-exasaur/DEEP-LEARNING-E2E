import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from src.utilities.logger import app_logger

logger = app_logger(__name__)
load_dotenv()

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", ""))
if not RAW_DATA_PATH:
    raise EnvironmentError("RAW_DATA_PATH is not set")

RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

def data_downloader(kaggle_uri: str) -> None:
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
