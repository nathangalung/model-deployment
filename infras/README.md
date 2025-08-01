# Kubeflow Deployment

This deployment provides a complete ML platform with Kubeflow Pipelines, Jupyter Notebooks, MLflow for model registry, KServe for model deployment, and MinIO for storage.

## Architecture

- **Kubeflow Pipelines**: ML workflow orchestration
- **Jupyter Notebooks**: Interactive development environment  
- **MLflow**: Model registry and experiment tracking
- **KServe**: Model serving and inference
- **MinIO**: Object storage for artifacts
- **MySQL**: Metadata and pipeline storage

## Environments

### Local Development
- Minimal resource allocation
- Port-forward access
- Kind cluster support
- Local storage

### Staging
- Higher resource allocation
- Ingress with staging domain
- Persistent storage
- SSL certificates

### Production
- High availability setup
- Production domains
- External managed databases
- Monitoring and security

## Quick Start

### Prerequisites
- Docker
- kubectl
- Kind (for local)
- kustomize

### Local Deployment
```bash
# Create Kind cluster
./scripts/kind-cluster-setup.sh

# Deploy Kubeflow with all infrastructure (defaults to local)
./scripts/setup-simple.sh

# Or explicitly specify environment
./scripts/setup-simple.sh local
```

**Note:** Local environment includes all infrastructure components (Istio, cert-manager, Knative, KServe) from the manifests directory - no external downloads needed!

### Staging Deployment
```bash
# Ensure you have a staging cluster
./scripts/setup-simple.sh staging
```

### Production Deployment
```bash
# Ensure you have a production cluster
./scripts/setup-simple.sh production
```

## Access Services

### Local (Port Forward)
```bash
# Kubeflow Pipelines
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8088:80
# Open: http://localhost:8088

# MLflow
kubectl port-forward -n kubeflow svc/mlflow-service 5555:5000
# Open: http://localhost:5555

# MinIO Console
kubectl port-forward -n kubeflow svc/minio 9009:9001
# Open: http://localhost:9009 (minio/minio123)
```

### Staging/Production (Ingress)
Services are available at configured domain endpoints.

## Directory Structure

```
deployment/
├── environments/           # Environment-specific configurations
│   ├── local/             # Local development
│   ├── staging/           # Staging environment
│   └── production/        # Production environment
├── manifests/             # Base manifests
├── scripts/               # Deployment scripts
└── README.md
```

## Environment Configuration

Each environment has its own kustomization with:
- Resource limits and requests
- Replica counts  
- Storage configuration
- Ingress rules (staging/production)
- Security policies (production)

## Cleanup

```bash
# Clean everything including Kind cluster
./scripts/cleanup-simple.sh

# Then setup again
./scripts/setup-simple.sh local
```

## Customization

### Adding New Environments
1. Create directory in `environments/`
2. Add `kustomization.yaml` with environment-specific patches
3. Update setup script if needed

### Modifying Resources
Edit the appropriate environment's patch files:
- `resource-limits-{env}.yaml` - CPU/memory limits
- `storage-{env}.yaml` - Storage configuration
- `ingress-{env}.yaml` - External access (staging/production)

### Security
Production environment includes additional security patches:
- Pod security policies
- Network policies
- Resource quotas
- RBAC restrictions

## Troubleshooting

### Common Issues
1. **Pods not starting**: Check resource limits and node capacity
2. **Storage issues**: Verify StorageClass exists
3. **Ingress not working**: Check DNS and certificate configuration
4. **Services unavailable**: Verify port-forward or ingress setup

### Debugging Commands
```bash
# Check pod status
kubectl get pods -n kubeflow

# Check pod logs
kubectl logs -n kubeflow <pod-name>

# Describe problematic pods
kubectl describe pod -n kubeflow <pod-name>

# Check events
kubectl get events -n kubeflow --sort-by='.lastTimestamp'
```

## Monitoring

For production environments, consider adding:
- Prometheus and Grafana
- Jaeger for tracing
- ELK stack for logging
- External monitoring solutions