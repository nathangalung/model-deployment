# Kubeflow Pipeline Deployment with KServe, MinIO, and MLflow

This deployment provides a complete machine learning platform with:
- **Kubeflow Pipelines** for ML workflow orchestration
- **KServe** for model serving and inference
- **MinIO** for object storage (S3-compatible)
- **MLflow** for experiment tracking and model registry
- **Istio** for service mesh
- **Knative Serving** for serverless deployments

## Prerequisites

1. **Docker** - Container runtime
2. **Kind** - Kubernetes in Docker
   ```bash
   curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
   chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind
   ```
3. **kubectl** - Kubernetes CLI
   ```bash
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   chmod +x kubectl && sudo mv kubectl /usr/local/bin/
   ```

## Quick Start

### Sequential Deployment Steps

**⚠️ Important: Run these scripts in order - each step depends on the previous one.**

#### Step 1: Create Kind Cluster
```bash
./deployment/scripts/kind-cluster-setup.sh
```

This creates a 3-node Kind cluster with proper port mappings for ingress. **Wait for this to complete before proceeding.**

#### Step 2: Deploy All Components
```bash
./deployment/scripts/setup.sh
```

This comprehensive script installs (in order):
1. Namespaces and storage configuration
2. Istio service mesh
3. Knative Serving
4. Cert Manager
5. KServe
6. Kubeflow Pipelines
7. MinIO storage
8. MLflow tracking server

**Note:** This process takes 15-20 minutes. Do not interrupt the process.

#### Step 3: Access Services

After deployment, access services via port-forwarding:

```bash
# Kubeflow Pipelines UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8088:80

# MLflow UI
kubectl port-forward -n kubeflow svc/mlflow-service 5555:5000

# MinIO Console
kubectl port-forward -n kubeflow svc/minio-service 9009:9001
```

**Service URLs:**
- Kubeflow Pipelines: http://localhost:8088
- MLflow: http://localhost:5555
- MinIO Console: http://localhost:9009

**MinIO Credentials:**
- Username: `minio`
- Password: `minio123`

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kubeflow      │    │     KServe      │    │     MLflow      │
│   Pipelines     │────│   (Serving)     │────│  (Registry)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │      MinIO      │
                    │   (Storage)     │
                    └─────────────────┘
```

## Components

### Kubeflow Pipelines
- **Namespace**: `kubeflow`
- **Purpose**: ML workflow orchestration
- **Components**: Pipeline UI, API server, Metadata store, Cache

### KServe
- **Namespace**: `kserve`
- **Purpose**: Model serving and inference
- **Features**: Serverless scaling, Multi-framework support, Canary deployments

### MinIO
- **Namespace**: `kubeflow`
- **Purpose**: S3-compatible object storage
- **Buckets**: `mlpipeline`, `mlflow-artifacts`, `models`

### MLflow
- **Namespace**: `kubeflow`
- **Purpose**: Experiment tracking and model registry
- **Storage**: MinIO for artifacts, SQLite for metadata

## Verification

Check component status:

```bash
# All Kubeflow components
kubectl get pods -n kubeflow

# KServe components
kubectl get pods -n kserve

# Knative Serving
kubectl get pods -n knative-serving

# Istio
kubectl get pods -n istio-system

# Cert Manager
kubectl get pods -n cert-manager
```

## Example: Deploy a Model with KServe

1. Create an InferenceService:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
  namespace: kubeflow
spec:
  predictor:
    sklearn:
      storageUri: s3://models/sklearn/iris
```

2. Apply the configuration:
```bash
kubectl apply -f inference-service.yaml
```

3. Test the model:
```bash
kubectl get inferenceservice sklearn-iris -n kubeflow
```

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending**: Check node resources
   ```bash
   kubectl describe nodes
   kubectl get pods -o wide
   ```

2. **MinIO connection issues**: Verify service endpoints
   ```bash
   kubectl get svc -n kubeflow | grep minio
   ```

3. **KServe inference issues**: Check Knative and Istio
   ```bash
   kubectl get pods -n knative-serving
   kubectl get pods -n istio-system
   ```

### Logs

View component logs:
```bash
# Kubeflow Pipeline logs
kubectl logs -n kubeflow deployment/ml-pipeline

# KServe controller logs
kubectl logs -n kserve deployment/kserve-controller-manager

# MLflow logs
kubectl logs -n kubeflow deployment/mlflow-server
```

## Complete Workflow

### Full Deployment Sequence
```bash
# 1. Create Kind cluster (2-3 minutes)
./deployment/scripts/kind-cluster-setup.sh

# 2. Deploy all components (15-20 minutes)
./deployment/scripts/setup.sh

# 3. Access services (run in separate terminals)
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8088:80
kubectl port-forward -n kubeflow svc/mlflow-service 5555:5000
kubectl port-forward -n kubeflow svc/minio-service 9009:9001
```

### Cleanup

To completely remove the deployment:

```bash
./deployment/scripts/cleanup.sh
```

This will:
- Delete all Kubernetes resources
- Remove the Kind cluster
- Clean up temporary files

**⚠️ Warning: This will destroy all data and cannot be undone.**

## Advanced Configuration

### Resource Limits

Modify resource limits in the manifest files:
- `deployment/manifests/resource-patches.yaml`
- Individual deployment manifests

### Storage Configuration

Configure persistent storage:
- `deployment/manifests/storage-class.yaml`
- Update PVC sizes as needed

### Network Configuration

Modify ingress and service configurations:
- `deployment/manifests/ingress-config.yaml`
- `deployment/manifests/port-forward-services.yaml`

## Support

For issues and questions:
1. Check component logs
2. Verify resource status
3. Review Kubernetes events: `kubectl get events --sort-by=.metadata.creationTimestamp`