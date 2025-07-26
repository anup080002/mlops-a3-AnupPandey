<!--‑‑ README.md – Project root ‑‑-->

# End‑to‑End MLOps Pipeline with Automated CI/CD and Post‑Training Quantization

![CI status](https://github.com/anup080002/mlops-a3-AnupPandey/actions/workflows/ci.yml/badge.svg)

---

## Project Overview

This repository demonstrates a **complete MLOps lifecycle**:

* **Model training** – `src/train.py` fits a scikit‑learn LinearRegression on the California‑housing dataset and exports both a `.joblib` artefact and an identical PyTorch state_dict.  
* **Containerisation** – `Dockerfile` bundles the training artefacts with a tiny inference script (`src/predict.py`) for instant “smoke” predictions.
* **CI/CD** – Every push to branch **docker_ci** triggers the GitHub Actions workflow `ci.yml` that  
  1. trains the model fresh,  
  2. builds the Docker image,  
  3. runs an on‑the‑fly prediction test,  
  4. pushes versioned tags to Docker Hub (`anup080002/mlops-a3`).
* **Quantization** – `src/quantize.py` applies uniform‑affine UINT8 compression, shrinking the weight file by **> 50 %** while maintaining R² parity.

---

## Repository Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="High‑level architecture diagram" width="650">
</p>

* **Four‑branch model** keeps concerns isolated  
  * `main` — documentation only  
  * `dev` — training code + requirements  
  * `docker_ci` — Dockerfile + CI workflow  
  * `quantization` — INT8 post‑training script
* **Immutable artefacts** are stored under `models/` and copied into the container.
* **CI runner** builds, tests, and publishes a deterministic Docker image on every commit.

---

## CI/CD Workflow

<p align="center">
  <img src="docs/workflow.svg" alt="GitHub Actions workflow sequence" width="650">
</p>

1. **Developer push** → GitHub Actions runner fires.  
2. Runner **re‑trains** (`train.py`) and deposits artefacts in `models/`.  
3. `docker build .` tags the image as  
   `anup080002/mlops-a3:<commit‑SHA>` and `latest`.  
4. Runner performs a **smoke‑test** (`docker run … predict.py`).  
5. Authenticated push to Docker Hub completes the pipeline.

---

## Quick Start

> **Prerequisites:** Python 3.10+, Docker Desktop, Git

```bash
# clone the repo
git clone https://github.com/anup080002/mlops-a3-AnupPandey.git
cd mlops-a3-AnupPandey

# local training test (creates models/*)
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python src/quantize.py
deactivate

# run the published container
docker run --rm anup080002/mlops-a3:latest
