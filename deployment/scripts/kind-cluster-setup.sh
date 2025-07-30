#!/bin/bash
# Complete Kind cluster setup with Kubeflow, KServe, MinIO, and MLflow

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

# Check if Kind is installed
if ! command -v kind &> /dev/null; then
    echo_error "Kind is not installed. Please install it first:"
    echo "curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64"
    echo "chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo_error "kubectl is not installed. Please install it first."
    exit 1
fi

# Delete existing cluster if it exists
echo_info "Checking for existing Kind cluster..."
if kind get clusters | grep -q "kubeflow-cluster"; then
    echo_warn "Deleting existing kubeflow-cluster..."
    kind delete cluster --name kubeflow-cluster
fi

# Create Kind cluster configuration
echo_info "Creating Kind cluster configuration..."
cat > /tmp/kind-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: kubeflow-cluster
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 8088
    protocol: TCP
  - containerPort: 443
    hostPort: 8443
    protocol: TCP
  - containerPort: 30000
    hostPort: 5555
    protocol: TCP
  - containerPort: 30001
    hostPort: 9009
    protocol: TCP
- role: worker
- role: worker
EOF

# Create Kind cluster
echo_info "Creating Kind cluster with 3 nodes..."
kind create cluster --config /tmp/kind-config.yaml --wait 300s

# Verify cluster is ready
echo_info "Verifying cluster nodes are ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

# Install NGINX Ingress Controller
echo_info "Installing NGINX Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Wait for ingress controller to be ready
echo_info "Waiting for NGINX Ingress Controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s

echo_info "Kind cluster setup completed successfully!"
echo_info "Cluster name: kubeflow-cluster"
echo_info "Nodes: $(kubectl get nodes --no-headers | wc -l)"
echo_info "Next: Run ./deployment/scripts/setup.sh to deploy all components"

# Clean up temporary files
rm -f /tmp/kind-config.yaml