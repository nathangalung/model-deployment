#!/bin/bash
# Kind cluster setup for Kubeflow development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Kind is installed
if ! command -v kind &> /dev/null; then
    log_error "Kind is not installed. Please install it first:"
    echo "curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64"
    echo "chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed. Please install it first."
    exit 1
fi

# Delete existing cluster if it exists
log_info "Checking for existing Kind cluster..."
if kind get clusters | grep -q "kubeflow-cluster"; then
    log_warn "Deleting existing kubeflow-cluster..."
    kind delete cluster --name kubeflow-cluster
fi

# Create Kind cluster configuration
log_info "Creating Kind cluster configuration..."
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
log_info "Creating Kind cluster with 3 nodes..."
kind create cluster --config /tmp/kind-config.yaml --wait 300s

# Verify cluster is ready
log_info "Verifying cluster nodes are ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

# Install NGINX Ingress Controller
log_info "Installing NGINX Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Wait for ingress controller to be ready
log_info "Waiting for NGINX Ingress Controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s

log_info "Kind cluster setup completed successfully"
log_info "Cluster name: kubeflow-cluster"
log_info "Nodes: $(kubectl get nodes --no-headers | wc -l)"
log_info "Next: Run ./kubeflow-setup.sh [local|staging|production] to deploy components"

# Clean up temporary files
rm -f /tmp/kind-config.yaml