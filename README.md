# End-to-End Deep Learning System

## Project Objective

This project aims to build a **production-grade, end-to-end deep learning system** that goes far beyond a notebook-based ML workflow. The goal is to design, implement, and deploy a **scalable, experiment-driven ML application** using industry best practices in software engineering, MLOps, and deployment.

The system is designed to:
- Train and evaluate deep learning models for medical image classification
- Support rapid experimentation using MLflow
- Be modular, testable, and extensible using OOP principles
- Be deployable as a real-world application (API + UI)

---

## Key Design Principles

- **Separation of Concerns**: Data, models, pipelines, and orchestration are strictly separated
- **Config-driven**: All parameters are controlled via YAML (no hardcoding)
- **Pipeline-based architecture**: Clear stages for data, training, evaluation, and inference
- **Reproducibility**: Deterministic runs using shared RUN_IDs and MLflow tracking
- **Production-first mindset**: Logging, artifacts, CI/CD, and deployment are first-class citizens

---

## Current Status (So Far)

### ✅ Implemented

#### 1. Project Structure (Production-Grade)
- `data/` – ingestion, validation, preprocessing
- `models/` – abstract base model + concrete implementations
- `pipelines/` – orchestration logic (data & training)
- `entity/` – strongly typed artifacts between pipelines
- `constants/` – shared static values and config keys
- `configs/` – YAML-based configuration
- `utilities/` – logging, config loading, helpers

#### 2. Object-Oriented Design
- Abstract `BaseModel`
- Concrete models (Simple CNN implemented)
- Extensible for transfer learning models (VGG16, EfficientNet, etc.)

#### 3. Data Pipeline
- Lightweight data validation (structure & sanity checks)
- TensorFlow dataset ingestion
- Configurable preprocessing & augmentation
- Optimized pipelines (map, prefetch)

#### 4. Training Pipeline
- Clean separation from data pipeline
- Model building, compilation, training
- Artifact-based outputs (model + history)

#### 5. Logging System
- Centralized, per-run logging
- Shared RUN_ID across all modules
- Single log file per execution
- Production-safe (no duplicate handlers)

---

## Planned / In-Progress Features

### 🔄 MLflow Integration
- MLflow tracking with **backend database** (SQLite / PostgreSQL)
- Log:
  - parameters
  - metrics
  - artifacts (models, plots)
- RUN_ID ↔ MLflow run_id alignment

---

### 📊 Experimentation & A/B Testing
- Support multiple model variants
- Compare:
  - architectures
  - preprocessing strategies
  - hyperparameters
- Route traffic between models for A/B testing

---

### 📐 Diagrams (To Be Added)
The following diagrams will be created and maintained:

1. **Architecture Diagram**
   - High-level system overview

2. **DFD (Data Flow Diagram)**
   - Data movement through ML pipeline

3. **Component Diagram**
   - Code/module-level structure

4. **Sequence Diagram**
   - Runtime execution flow

5. **Deployment Diagram**
   - Docker → AWS EC2 → Future services

---

### 🚀 Deployment Plan

#### Backend
- FastAPI-based inference service
- REST endpoints for prediction
- MLflow model loading

#### Frontend
- HTML + CSS + JavaScript UI
- Image upload and prediction display

#### Containerization
- Dockerize the full application
- Push image to Docker Hub
- Pull and deploy on AWS EC2

---

### 🔁 CI/CD

- Automated testing
- Linting & formatting
- Docker image build on commit
- Automated deployment pipeline

---

## Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **Experiment Tracking**: MLflow (backend DB)
- **Backend API**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker
- **Cloud**: AWS EC2
- **CI/CD**: GitHub Actions (planned)

---

## Final Goal

By the end of this project, the system will:
- Train multiple deep learning models
- Track experiments cleanly with MLflow
- Support A/B testing
- Be fully dockerized
- Be deployed on AWS
- Serve predictions via an API and UI

This repository is intentionally structured to reflect **real-world ML systems**, not tutorial code.

---

## Notes

This project is being developed incrementally with a strong emphasis on:
- correctness over speed
- architecture over shortcuts
- maintainability over convenience

Each stage builds toward a **true end-to-end ML product**, not just a model.

