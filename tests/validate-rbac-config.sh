#!/bin/bash
# Validate RBAC configurations in Kubernetes cluster

set -e

echo "🔐 Validating Kubeflow RBAC Configurations"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED_CHECKS=0
TOTAL_CHECKS=0

# Helper function for validation checks
validate_resource() {
    local resource_type="$1"
    local resource_name="$2"
    local namespace="${3:-}"
    
    echo -n "Checking $resource_type '$resource_name'... "
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    local cmd="kubectl get $resource_type $resource_name"
    if [[ -n "$namespace" ]]; then
        cmd="$cmd -n $namespace"
    fi
    
    if $cmd > /dev/null 2>&1; then
        echo -e "${GREEN}✅ EXISTS${NC}"
        return 0
    else
        echo -e "${RED}❌ MISSING${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ ERROR: kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check cluster connectivity
echo "Checking Kubernetes cluster connectivity..."
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${RED}❌ ERROR: Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"
echo ""

# Validate ClusterRoles
echo "🔑 Validating ClusterRoles..."
validate_resource "clusterrole" "kubeflow-admin"
validate_resource "clusterrole" "kubeflow-pipelines-edit"
validate_resource "clusterrole" "notebook-controller-kubeflow-notebooks-admin"
validate_resource "clusterrole" "kubeflow-kserve-admin"

echo ""
# Validate ClusterRoleBindings
echo "🔗 Validating ClusterRoleBindings..."
validate_resource "clusterrolebinding" "kubeflow-admin-anonymous"
validate_resource "clusterrolebinding" "kubeflow-users-admin"
validate_resource "clusterrolebinding" "kubeflow-pipelines-admin-anonymous"
validate_resource "clusterrolebinding" "kubeflow-notebooks-admin-anonymous"
validate_resource "clusterrolebinding" "kubeflow-kserve-admin-anonymous"

echo ""
# Validate RoleBindings in specific namespaces
echo "🎯 Validating Namespace-specific RoleBindings..."
for namespace in "dummypipeline" "kubeflow"; do
    echo "Checking namespace: $namespace"
    validate_resource "rolebinding" "kubeflow-admin-anonymous" "$namespace"
    validate_resource "rolebinding" "kubeflow-pipelines-admin-anonymous" "$namespace"
    validate_resource "rolebinding" "kubeflow-notebooks-admin-anonymous" "$namespace"
    validate_resource "rolebinding" "kubeflow-kserve-admin-anonymous" "$namespace"
done

echo ""
# Check actual permissions using kubectl auth can-i
echo "✅ Validating User Permissions (kubectl auth can-i)..."

# Test permissions for anonymous@kubeflow.org
echo "Testing permissions for user 'anonymous@kubeflow.org':"

PERMISSION_TESTS=(
    "get pods --namespace=dummypipeline"
    "list pipelines.kubeflow.org --namespace=dummypipeline"
    "create notebooks.kubeflow.org --namespace=dummypipeline"
    "get inferenceservices.serving.kserve.io --namespace=dummypipeline"
    "list persistentvolumeclaims --namespace=dummypipeline"
    "get profiles.kubeflow.org"
)

for test in "${PERMISSION_TESTS[@]}"; do
    echo -n "  $test... "
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if kubectl auth can-i $test --as=anonymous@kubeflow.org > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ALLOWED${NC}"
    else
        echo -e "${RED}❌ DENIED${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
done

echo ""
# Check ServiceAccounts and their bindings
echo "👤 Validating ServiceAccounts..."
SERVICEACCOUNTS=(
    "pipeline-runner:kubeflow"
    "default-editor:dummypipeline"
    "default-viewer:dummypipeline"
)

for sa_info in "${SERVICEACCOUNTS[@]}"; do
    IFS=':' read -r sa_name sa_namespace <<< "$sa_info"
    validate_resource "serviceaccount" "$sa_name" "$sa_namespace"
done

echo ""
echo "=========================================="
echo "📋 RBAC Validation Summary"
echo "=========================================="
echo ""

PASSED_CHECKS=$((TOTAL_CHECKS - FAILED_CHECKS))
PASS_RATE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

echo -e "Total Checks: ${TOTAL_CHECKS}"
echo -e "Passed: ${GREEN}${PASSED_CHECKS}${NC}"
echo -e "Failed: ${RED}${FAILED_CHECKS}${NC}"
echo -e "Success Rate: ${PASS_RATE}%"
echo ""

if [[ $FAILED_CHECKS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All RBAC configurations are valid!${NC}"
    echo -e "${GREEN}✅ Kubeflow RBAC is properly configured${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run the HTTP endpoint tests: ./test-rbac-permissions.sh"
    echo "2. Deploy your ML pipelines: cd app && python upload.py"
else
    echo -e "${RED}⚠️  Some RBAC configurations are missing or invalid!${NC}"
    echo ""
    echo "To fix issues:"
    echo "1. Apply RBAC configurations:"
    echo "   kubectl apply -f infras/environments/local/kubeflow-rbac-fix.yaml"
    echo "   kubectl apply -f infras/environments/local/namespace-rbac-fix.yaml"
    echo "2. Wait for changes to propagate (30-60 seconds)"
    echo "3. Re-run this validation script"
    echo ""
    exit 1
fi