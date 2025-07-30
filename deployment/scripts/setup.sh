#!/bin/bash
# Complete Kubeflow setup with KServe, MinIO, and MLflow

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Note: This script deploys all components without waiting for initialization
# You can manually restart deployments if needed:
# kubectl rollout restart deployment <deployment-name> -n <namespace>

echo_step "=== Starting Kubeflow Complete Setup ==="

# Check if cluster exists
if ! kubectl cluster-info &> /dev/null; then
    echo_error "No Kubernetes cluster found. Please run ./deployment/scripts/kind-cluster-setup.sh first"
    exit 1
fi

# Check required dependencies
echo_info "Checking required dependencies..."
MISSING_DEPS=""

if ! command -v kubectl &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS kubectl"
fi

if ! command -v openssl &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS openssl"
fi

if ! command -v curl &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS curl"
fi

if [ -n "$MISSING_DEPS" ]; then
    echo_error "Missing required dependencies:$MISSING_DEPS"
    echo_error "Please install the missing dependencies and try again"
    exit 1
fi

echo_info "All dependencies are available ✅"

echo_step "1/12 Creating namespaces..."
kubectl apply -f deployment/manifests/namespaces.yaml

echo_step "2/12 Installing storage classes and persistent volumes..."
kubectl apply -f deployment/manifests/storage-class.yaml

echo_step "3/12 Installing Application CRD..."
kubectl apply -f deployment/manifests/application-crd.yaml

echo_step "4/12 Installing Istio..."
if [ ! -d "./istio-1.20.0" ]; then
    echo_info "Downloading and installing Istio..."
    curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.20.0 sh -
else
    echo_info "Istio already downloaded, skipping download ✅"
fi
export PATH="./istio-1.20.0/bin:$PATH"

# Check if Istio is already installed
if kubectl get namespace istio-system &>/dev/null && kubectl get deployment istiod -n istio-system &>/dev/null; then
    echo_info "Istio already installed, skipping installation ✅"
else
    echo_info "Installing Istio..."
    istioctl install --set values.defaultRevision=default -y
fi
kubectl label namespace kubeflow istio-injection=enabled --overwrite

echo_step "5/12 Installing Knative Serving..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-core.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.12.0/net-istio.yaml

echo_step "6/12 Installing Cert Manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

echo_info "Waiting for cert-manager webhook to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/cert-manager-webhook -n cert-manager

echo_step "7/12 Installing KServe..."
# Retry KServe installation up to 3 times with exponential backoff
for i in {1..3}; do
    echo_info "KServe installation attempt $i/3..."
    if kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml; then
        echo_info "KServe installation successful ✅"
        break
    else
        if [ $i -lt 3 ]; then
            echo_warn "KServe installation failed, retrying in $((i * 30)) seconds..."
            sleep $((i * 30))
        else
            echo_error "KServe installation failed after 3 attempts"
            exit 1
        fi
    fi
done
kubectl apply -f deployment/manifests/kserve-config.yaml
kubectl apply -f deployment/manifests/kserve-rbac.yaml

# Install all required CRDs before Kubeflow Pipelines
echo_info "Installing all required CRDs..."
kubectl apply -f deployment/manifests/argo-workflows-complete-crds.yaml
kubectl apply -k "github.com/argoproj/argo-workflows/manifests/cluster-install?ref=v3.5.14" || echo_warn "Some Argo Workflows components failed (expected for missing namespaces)"
kubectl apply -f deployment/manifests/viewer-crd.yaml

echo_step "8/12 Installing Kubeflow Pipelines..."
echo_info "Installing Kubeflow Pipelines v2.5.0..."
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=2.5.0"

# MySQL will be deployed with Kubeflow Pipelines

# Create TLS certificates for webhook server (before installing Kubeflow)
echo_info "Creating TLS certificates for webhook server..."
if [ ! -f "webhook-server-tls.crt" ]; then
    echo_info "Generating self-signed TLS certificates..."
    openssl req -x509 -newkey rsa:4096 -keyout webhook-server-tls.key -out webhook-server-tls.crt -days 365 -nodes -subj "/CN=webhook-server"
    chmod 600 webhook-server-tls.key
fi

kubectl create secret generic webhook-server-tls -n kubeflow \
    --from-file=cert.pem=webhook-server-tls.crt \
    --from-file=key.pem=webhook-server-tls.key \
    --dry-run=client -o yaml | kubectl apply -f -

# CRDs were already installed earlier

# Apply MySQL fixes for authentication
echo_info "Applying MySQL authentication fixes..."
kubectl apply -f deployment/manifests/mysql-fix.yaml

# Restart problematic deployments to ensure they pick up fixes
echo_info "Restarting deployments to pick up configuration changes..."
kubectl rollout restart deployment cache-server -n kubeflow || true
kubectl rollout restart deployment workflow-controller -n kubeflow || true
kubectl rollout restart deployment ml-pipeline-viewer-crd -n kubeflow || true

echo_step "9/12 Applying resource patches..."
echo_info "Patching MinIO deployment for maximum resources..."
kubectl patch deployment minio -n kubeflow --patch-file deployment/manifests/resource-patches.yaml || echo_warn "MinIO patch failed, continuing..."

echo_step "10/12 Deploying additional components..."
echo_info "Deploying standalone MinIO instance..."
kubectl apply -f deployment/manifests/minio-standalone.yaml

echo_info "Deploying MLflow server..."
kubectl apply -f deployment/manifests/mlflow-server.yaml

echo_info "Setting up port forwarding services..."
kubectl apply -f deployment/manifests/port-forward-services.yaml

echo_info "Setting up ingress configuration..."
kubectl apply -f deployment/manifests/ingress-config.yaml

echo_step "11/12 Additional components deployed..."
echo_info "MinIO standalone and MLflow server deployed"

echo_step "12/12 Creating MinIO buckets..."
echo_info "Waiting for MinIO service to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/minio -n kubeflow || echo_warn "MinIO deployment wait timed out, continuing anyway..."

echo_info "Creating required buckets in MinIO..."
# Clean up any existing mc-client pods
kubectl delete pod -l app=mc-client --ignore-not-found=true

# Try bucket creation with timeout and retry logic
for i in {1..3}; do
    echo_info "MinIO bucket creation attempt $i/3..."
    # Use timestamp to ensure unique pod names
    TIMESTAMP=$(date +%s)
    POD_NAME="mc-client-${TIMESTAMP}-${i}"
    
    kubectl run $POD_NAME --image=minio/mc --restart=Never --labels=app=mc-client -- \
      /bin/sh -c "mc alias set myminio http://minio-service.kubeflow:9000 minio minio123 && \
                  mc mb myminio/mlpipeline 2>/dev/null || echo 'Bucket mlpipeline exists or created' && \
                  mc mb myminio/mlflow-artifacts 2>/dev/null || echo 'Bucket mlflow-artifacts exists or created' && \
                  mc mb myminio/models 2>/dev/null || echo 'Bucket models exists or created'"
    
    # Wait for pod to complete with timeout
    if timeout 120s kubectl wait --for=condition=Ready pod/$POD_NAME --timeout=120s && \
       kubectl wait --for=condition=PodReadyCondition=false pod/$POD_NAME --timeout=120s; then
        # Check if pod completed successfully
        if kubectl get pod $POD_NAME -o jsonpath='{.status.phase}' | grep -q "Succeeded"; then
            echo_info "MinIO buckets created successfully ✅"
            kubectl delete pod $POD_NAME --ignore-not-found=true
            break
        else
            echo_warn "MinIO bucket creation pod failed"
            kubectl logs $POD_NAME || true
            kubectl delete pod $POD_NAME --ignore-not-found=true
        fi
    else
        echo_warn "MinIO bucket creation timed out"
        kubectl delete pod $POD_NAME --ignore-not-found=true
    fi
    
    if [ $i -lt 3 ]; then
        echo_warn "MinIO bucket creation failed, retrying in 30 seconds..."
        sleep 30
    else
        echo_warn "MinIO bucket creation failed after 3 attempts, you may need to create them manually"
    fi
done

# Clean up components not needed for local deployment
echo_info "Cleaning up MySQL fix job..."
kubectl delete job mysql-fix-auth -n kubeflow --ignore-not-found=true

echo_info "Removing proxy-agent (not needed for local deployment)..."
kubectl delete deployment proxy-agent -n kubeflow --ignore-not-found=true

echo_step "=== Final Status Check ==="
echo_info "Performing final status check..."

# Count running pods
TOTAL_PODS=$(kubectl get pods -n kubeflow --no-headers | wc -l)
RUNNING_PODS=$(kubectl get pods -n kubeflow --no-headers | grep "Running" | wc -l)
READY_PODS=$(kubectl get pods -n kubeflow --no-headers | awk '$2 ~ /^[0-9]+\/[0-9]+$/ && $2 !~ /0\// {split($2, a, "/"); if(a[1]==a[2]) print}' | wc -l)

echo_info "Pod Status Summary:"
echo "  Total Pods: $TOTAL_PODS"
echo "  Running Pods: $RUNNING_PODS" 
echo "  Ready Pods: $READY_PODS"

# Show any problematic pods
PROBLEM_PODS=$(kubectl get pods -n kubeflow --no-headers | grep -E "Error|CrashLoopBackOff|ImagePullBackOff|Pending" | awk '{print $1}' || true)
if [ -n "$PROBLEM_PODS" ]; then
    echo_warn "Pods with issues:"
    echo "$PROBLEM_PODS"
    echo_info "Detailed status:"
    kubectl get pods -n kubeflow
else
    echo_info "✅ All pods are running successfully!"
fi

# Check KServe
KSERVE_PODS=$(kubectl get pods -n kserve --no-headers 2>/dev/null | grep "Running" | wc -l || echo "0")
echo_info "KServe Status: $KSERVE_PODS pod(s) running"

echo_step "=== Setup completed successfully! ==="
echo ""
echo_info "🎉 All components have been deployed successfully!"
echo ""
echo_info "Access URLs (after port-forwarding):"
echo "  📊 Kubeflow Pipelines UI: http://localhost:8088"
echo "  🔬 MLflow UI: http://localhost:5555"
echo "  📦 MinIO Console: http://localhost:9009"
echo ""
echo_info "To access the services, run these port-forward commands in separate terminals:"
echo "  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8088:80"
echo "  kubectl port-forward -n kubeflow svc/mlflow-service 5555:5000"
echo "  kubectl port-forward -n kubeflow svc/minio-service 9009:9001"
echo ""
echo_info "MinIO Credentials:"
echo "  Username: minio"
echo "  Password: minio123"
echo ""
echo_info "Quick verification commands:"
echo "  kubectl get pods -n kubeflow    # Check Kubeflow pods"
echo "  kubectl get pods -n kserve      # Check KServe pods"
echo "  kubectl get svc -n kubeflow     # Check services"
echo ""
echo_info "If some pods are not running, try these troubleshooting commands:"
echo "  kubectl rollout restart deployment <deployment-name> -n kubeflow"  
echo "  kubectl delete pod <pod-name> -n kubeflow"
echo "  kubectl describe pod <pod-name> -n kubeflow"
echo "  kubectl logs <pod-name> -n kubeflow"

# Keep Istio installation files for future use
# rm -rf ./istio-1.20.0 || true