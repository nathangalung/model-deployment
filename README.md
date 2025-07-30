# ML Model Deployment Pipeline

## Overview

This project provides an end-to-end machine learning pipeline for model deployment using **Kubeflow Pipelines**. The pipeline covers data collection, preprocessing, model training, evaluation, and deployment to a prediction endpoint using KServe. All configuration is managed via `config.yaml`.

---

## Project Structure

```
├── kubeflow/
│   ├── pipeline.py           # Main Kubeflow pipeline definition
│   ├── upload.py             # Script to upload and run pipeline on Kubeflow
│   ├── pipeline.yaml         # Compiled pipeline (generated)
│   ├── requirements.txt      # Python dependencies for Kubeflow image
│   ├── Dockerfile            # Docker image for all Kubeflow components
│   └── components/
│       ├── collect_data.py
│       ├── preprocess_data.py
│       ├── train_model.py
│       ├── evaluate_model.py
│       └── deploy_model.py
├── data/
│   └── dataset_all_sample.parquet
├── config.yaml               # Central pipeline and resource configuration
└── README.md                 # This documentation
```

---

## Configuration (`config.yaml`)

All pipeline settings are defined in `config.yaml`.  
**Do not hardcode endpoints or resource limits; these are set automatically.**

Example:
```yaml
PROJECT_NAME: "DummyData"
EXPERIMENT_NAME: "DummyDataModel"
TARGET_COL: "label"
ID_COLUMNS: ["risk_id"]
IGNORED_FEATURES: []
DATE_COL: "partition_date"
TRAIN_START_DATE: "2024-01-01"
TRAIN_END_DATE: "2024-08-31"
OOT_START_DATE: "2024-09-01"
OOT_END_DATE: "2024-09-31"

MINIO:
  ENDPOINT: "minio-service.kubeflow:9000"
  ACCESS_KEY: "minio"
  SECRET_KEY: "minio123"
  BUCKET: "mlpipeline"
  DATASET_OBJECT: "dataset_all_sample.parquet"

MLFLOW:
  EXPERIMENT_NAME: "kubeflow-ml-pipeline"

H2O_CONFIG:
  MAX_MODELS: 1
  MAX_RUNTIME_SECS: 1200
  AUTOML_SEED: 42
  CROSS_VALIDATION: "YES"

KUBERNETES:
  NAMESPACE: "kubeflow"
  SERVICE_ACCOUNT: "kserve-service-account"

KSERVE:
  DEPLOYMENT_MODE: "Serverless"

MODEL_NAME: ""      # Optional, auto-generated if empty
MODEL_VERSION: ""   # Optional, auto-generated if empty
PIPELINE_RESOURCES: {}  # Optional, auto-set to max available if empty
```

---

## Pipeline Steps

1. **Data Collection (`collect_data`)**  
   Loads dataset from MinIO (parquet file).

2. **Data Preprocessing (`preprocess_data`)**  
   Cleans, filters, and selects features.

3. **Model Training (`train_model`)**  
   Trains H2O AutoML model, logs to MLflow, registers model.

4. **Model Evaluation (`evaluate_model`)**  
   Evaluates model performance, generates summary.

5. **Model Deployment (`deploy_model`)**  
   Builds Docker image, deploys with KServe, exposes prediction endpoint.

---

## Usage

### 1. Build the Docker Image

```bash
cd kubeflow
docker build -t nathangalung246/kubeflow_dummy:latest .
```

### 2. Compile the Pipeline

```bash
uv run pipeline.py
```
This generates `pipeline.yaml`.

### 3. Upload and Run the Pipeline

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888
```
Then, in another terminal:
```bash
export KUBEFLOW_ENDPOINT=http://localhost:8888
uv run upload.py
```
This uploads the pipeline, creates an experiment, and starts a run.

### 4. Monitor Pipeline Progress

- **Kubeflow UI:**  
  ```bash
  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
  ```
  Open [http://localhost:8080](http://localhost:8080) and monitor your run under "Runs".

- **MinIO Console:**  
  ```bash
  kubectl port-forward -n kubeflow svc/minio-service 9000:9000
  ```
  Open [http://localhost:9000](http://localhost:9000) to browse datasets and artifacts.

- **MLflow UI:**  
  ```bash
  kubectl port-forward -n kubeflow svc/mlflow-service 5000:5000
  ```
  Open [http://localhost:5000](http://localhost:5000) to view experiment runs and models.

- **KServe Endpoint:**  
  After deployment, the API endpoint will be shown in the pipeline logs and Kubeflow UI.

---

## Troubleshooting

- **Dataset Not Found:**  
  Ensure `data/dataset_all_sample.parquet` exists and is uploaded to MinIO.

- **Kubeflow Connection:**  
  Check `KUBEFLOW_ENDPOINT` and port-forwarding.

- **Resource Issues:**  
  The pipeline auto-detects and uses all available CPU and memory.

- **Dependencies:**  
  All dependencies are managed in `kubeflow/requirements.txt` and installed via Docker.

---

## Testing

Run tests with:
```bash
uv run -m pytest tests/
```

---

## Summary

- All configuration is in `config.yaml`
- All pipeline logic is in `kubeflow/`
- Build the Docker image, compile, upload, and monitor via UI
- End-to-end ML pipeline: data → model → endpoint

For more details, see comments in each pipeline component and the Kubeflow