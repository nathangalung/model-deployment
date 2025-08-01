#!/bin/bash
# Cleanup script for Kubeflow and all related resources

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

log_info "Starting complete Kubeflow cleanup..."

# Function to safely delete namespace
delete_namespace() {
    local ns=$1
    if kubectl get namespace "$ns" &>/dev/null; then
        log_info "Deleting namespace: $ns"
        kubectl delete namespace "$ns" --timeout=60s || {
            log_warn "Force deleting namespace: $ns"
            kubectl delete namespace "$ns" --force --grace-period=0 || true
        }
    else
        log_info "Namespace $ns does not exist"
    fi
}

# List of namespaces to clean up (as specified by user)
NAMESPACES=(
    "kubeflow"
    "dummypipeline"
    "kserve"
    "knative-serving"
    "istio-system"
    "cert-manager"
    "ingress-nginx"
    "local-path-storage"
)

# Delete Kubeflow applications first to prevent hanging resources
log_info "Cleaning up Kubeflow applications..."
kubectl delete pods -l run=mc-client --ignore-not-found=true --all-namespaces || true

# Delete resources in kubeflow namespace first
if kubectl get namespace kubeflow &>/dev/null; then
    log_info "Cleaning up resources in kubeflow namespace..."
    for resource in deployment service configmap secret pvc statefulset job cronjob; do
        kubectl delete $resource --all -n kubeflow --ignore-not-found=true --timeout=30s || true
    done
fi

# Delete all specified namespaces
log_info "Deleting all specified namespaces..."
for ns in "${NAMESPACES[@]}"; do
    delete_namespace "$ns"
done

# Clean up CRDs
log_info "Cleaning up Custom Resource Definitions..."
kubectl delete crd -l app.kubernetes.io/name=istio --ignore-not-found=true --timeout=30s || true
kubectl delete crd -l app.kubernetes.io/name=knative-serving --ignore-not-found=true --timeout=30s || true
kubectl delete crd -l app.kubernetes.io/name=cert-manager --ignore-not-found=true --timeout=30s || true
kubectl delete crd poddefaults.kubeflow.org --ignore-not-found=true --timeout=30s || true

# Remove Kind clusters if they exist
log_info "Removing Kind clusters..."
if command -v kind &> /dev/null; then
    for cluster in $(kind get clusters 2>/dev/null || echo ""); do
        log_info "Deleting Kind cluster: $cluster"
        kind delete cluster --name "$cluster" || true
    done
else
    log_warn "Kind command not found, skipping cluster cleanup"
fi

# Clean up Docker containers from Kind
log_info "Cleaning up Docker containers..."
docker ps -a --filter "label=io.x-k8s.kind.cluster" --format "{{.ID}}" | xargs -r docker rm -f || true

# Clean up Kind networks
log_info "Cleaning up Docker networks..."
docker network ls --filter "name=kind" --format "{{.ID}}" | xargs -r docker network rm || true

# Clean up kubectl contexts
log_info "Cleaning up kubectl contexts..."
for context in $(kubectl config get-contexts --no-headers | awk '{print $2}' | grep "kind-" 2>/dev/null || echo ""); do
    kubectl config delete-context "$context" || true
done

# Clean up kubectl clusters
for cluster in $(kubectl config get-clusters --no-headers | grep "kind-" 2>/dev/null || echo ""); do
    kubectl config delete-cluster "$cluster" || true
done

# Clean up kubectl users
for user in $(kubectl config get-users --no-headers | grep "kind-" 2>/dev/null || echo ""); do
    kubectl config delete-user "$user" || true
done

log_info "Complete cleanup finished"
log_info "All specified namespaces, Kind clusters, and kubectl configs have been removed"
log_info "Cleaned namespaces: ${NAMESPACES[*]}"