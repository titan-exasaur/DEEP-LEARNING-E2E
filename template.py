import os

project_structure = [
    "artifacts/",
    "assets/",
    "configs/",
    "docs/",
    "notebooks/",
    "tests/unit/",
    "tests/integration/",

    "requirements.txt",
    "README.md",
    ".gitignore",
    "Dockerfile",
    ".env",
    "schema.yaml",

    "src/",
    "src/__init__.py",

    "src/config/",
    "src/config/__init__.py",
    "src/config/configuration.py",

    "src/constants/",
    "src/constants/__init__.py",
    "src/entity/",
    "src/entity/__init__.py",

    "src/data/",
    "src/data/__init__.py",
    "src/data/data_ingestion.py",
    "src/data/data_validation.py",
    "src/data/data_preprocessing.py",

    "src/models/",
    "src/models/__init__.py",
    "src/models/base_model.py",

    "src/pipelines/",
    "src/pipelines/__init__.py",
    "src/pipelines/train_pipeline.py",
    "src/pipelines/predict_pipeline.py",

    "src/training/",
    "src/training/__init__.py",
    "src/training/model_trainer.py",
    "src/training/evaluation.py",

    "src/experiment/",
    "src/experiment/__init__.py",
    "src/experiment/mlflow_tracking.py",

    "src/inference/",
    "src/inference/__init__.py",
    "src/inference/predictor.py",

    "src/logger.py",
    "src/utils.py"
]

for path in project_structure:
    if path.endswith("/"):
        os.makedirs(path, exist_ok=True)
    else:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()
