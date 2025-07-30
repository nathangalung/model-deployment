#!/bin/bash
# Cleanup script for Kubeflow deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

echo_info "Starting cleanup of Kubeflow deployment..."

# Delete all custom resources and namespaces
echo_info "Deleting Kubeflow resources..."
kubectl delete namespace kubeflow --ignore-not-found=true
kubectl delete namespace kserve --ignore-not-found=true
kubectl delete namespace knative-serving --ignore-not-found=true
kubectl delete namespace istio-system --ignore-not-found=true
kubectl delete namespace cert-manager --ignore-not-found=true

echo_info "Deleting CRDs..."
kubectl delete crd applications.app.k8s.io --ignore-not-found=true

# Delete ALL Kind clusters
echo_info "Deleting all Kind clusters..."
CLUSTERS=$(kind get clusters 2>/dev/null || true)
if [ -n "$CLUSTERS" ]; then
    for cluster in $CLUSTERS; do
        echo_info "Deleting Kind cluster: $cluster"
        kind delete cluster --name "$cluster"
        echo_info "Kind cluster '$cluster' deleted successfully"
    done
else
    echo_warn "No Kind clusters found"
fi

# Kill any port-forward processes
echo_info "Stopping port-forward processes..."
pkill -f "kubectl port-forward" || true

# Clean up any remaining files
echo_info "Cleaning up temporary files..."
rm -rf ./istio-1.20.0 || true

# Check for processes using common ports and warn user
echo_info "Checking for processes using ports 8080, 5000, 9001..."
for port in 8080 5000 9001; do
    if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        echo_warn "Port $port is still in use. You may need to stop the process manually."
        echo_warn "Run: sudo lsof -i :$port to identify the process"
    fi
done

echo_info "Cleanup completed successfully!"