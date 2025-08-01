#!/bin/bash
# Kubeflow setup script with staged deployment
# Usage: ./kubeflow-setup.sh [local|staging|production]

set -e

# Default to local environment
ENVIRONMENT=${1:-local}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

log_step "=== Kubeflow Setup - Environment: $ENVIRONMENT ==="

# Validate environment
case $ENVIRONMENT in
    local|staging|production)
        log_info "Setting up for $ENVIRONMENT environment"
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT. Use: local, staging, or production"
        exit 1
        ;;
esac

# Check dependencies
for cmd in kubectl kustomize; do
    if ! command -v $cmd &> /dev/null; then
        log_error "Missing dependency: $cmd"
        exit 1
    fi
done

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    log_error "No Kubernetes cluster found. Run ./kind-cluster-setup.sh first"
    exit 1
fi

log_info "All dependencies available"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_DIR="$SCRIPT_DIR/../environments/$ENVIRONMENT"

# Function to apply stage with retries
apply_stage() {
    local stage=$1
    local max_retries=3
    local retry=0
    
    log_info "Applying $stage..."
    
    while [[ $retry -lt $max_retries ]]; do
        if kubectl apply -k "$MANIFEST_DIR/$stage" 2>&1; then
            log_info "$stage applied successfully"
            return 0
        else
            retry=$((retry + 1))
            if [[ $retry -lt $max_retries ]]; then
                log_warn "$stage failed, retrying in 10 seconds..."
                sleep 10
            else
                log_error "Failed to apply $stage after $max_retries attempts"
                return 1
            fi
        fi
    done
}

# Function to wait for deployments
wait_for_ready() {
    local namespace=$1
    local timeout=${2:-300}
    
    log_info "Waiting for deployments in $namespace to be ready..."
    kubectl wait --for=condition=available --timeout=${timeout}s \
        deployment --all -n $namespace 2>/dev/null || {
        log_warn "Some deployments in $namespace may not be ready yet"
        kubectl get pods -n $namespace
    }
}

# Staged deployment process
log_step "1/4 Stage 1: CRDs and Basic Infrastructure"
if ! apply_stage "stage1"; then
    log_error "Stage 1 failed. Cannot proceed."
    exit 1
fi

log_info "Waiting for cert-manager to be ready..."
wait_for_ready "cert-manager" 180

log_step "2/4 Stage 2: Infrastructure Components"
if ! apply_stage "stage2"; then
    log_error "Stage 2 failed. Cannot proceed."
    exit 1
fi

log_info "Waiting for Knative to be ready..."
wait_for_ready "knative-serving" 180

log_step "3/4 Stage 3: Kubeflow Applications"
if ! apply_stage "stage3"; then
    log_error "Stage 3 failed. Cannot proceed."
    exit 1
fi

log_step "4/4 Verification and Final Configuration"
log_info "Waiting for core services to be ready..."
wait_for_ready "kubeflow" 300

# Apply immediate fixes for local development
if [ "$ENVIRONMENT" = "local" ]; then
    log_info "Applying local development fixes..."
    
    # Disable Istio injection for namespaces to prevent webhook errors
    kubectl label namespace kubeflow istio-injection=disabled --overwrite 2>/dev/null || true
    kubectl label namespace dummypipeline istio-injection=disabled --overwrite 2>/dev/null || true
    
    # Restart deployments to pick up new configurations
    kubectl rollout restart deployment/centraldashboard -n kubeflow 2>/dev/null || true
    
    log_info "Local development fixes applied"
fi

# Final status check
log_info "Final deployment status:"
kubectl get pods -n kubeflow

FAILED_PODS=$(kubectl get pods -n kubeflow --field-selector=status.phase!=Running,status.phase!=Succeeded --no-headers 2>/dev/null | wc -l)
if [[ $FAILED_PODS -gt 0 ]]; then
    log_warn "Some pods are still not running. You may need to run this script again."
    kubectl get pods -n kubeflow --field-selector=status.phase!=Running,status.phase!=Succeeded
else
    log_info "All pods are running successfully"
fi

log_step "=== Setup Complete ==="
log_info "Kubeflow deployed successfully in $ENVIRONMENT environment"

if [ "$ENVIRONMENT" = "local" ]; then
    echo ""
    log_info "Access Kubeflow services:"
    echo "  Main Gateway:"
    echo "    kubectl port-forward -n kubeflow svc/kubeflow-gateway-proxy 8081:80"
    echo ""
    echo "  Service URLs (via localhost:8081):"
    echo "    Central Dashboard: http://localhost:8081/"
    echo "    Jupyter Notebooks: http://localhost:8081/jupyter/"
    echo "    Kubeflow Pipelines: http://localhost:8081/pipeline/"
    echo "    Volumes Manager: http://localhost:8081/volumes/"
    echo "    KServe Models: http://localhost:8081/kserve-endpoints/"
    echo "    MLflow Registry: http://localhost:8081/mlflow/"
    echo ""
    log_info "Credentials:"
    echo "  MinIO: minio/minio123"
fi

echo ""
log_info "Usage:"
echo "  Local:      ./kubeflow-setup.sh local"
echo "  Staging:    ./kubeflow-setup.sh staging"
echo "  Production: ./kubeflow-setup.sh production"