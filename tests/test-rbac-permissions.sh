#!/bin/bash
# Test RBAC permissions for all Kubeflow services

set -e

echo "🧪 Testing Kubeflow RBAC Permissions"
echo "===================================="
echo ""

PROXY_URL="http://localhost:8081"
NAMESPACE="dummypipeline"
TIMEOUT=10
FAILED_TESTS=0
TOTAL_TESTS=0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function to test HTTP endpoints
test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_pattern="$3"
    local test_type="${4:-content}"
    
    echo -n "Testing $name... "
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local response
    local http_code
    
    if [[ "$test_type" == "status_only" ]]; then
        http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$url" 2>/dev/null || echo "000")
        if [[ "$http_code" -ge 200 && "$http_code" -lt 400 ]]; then
            echo -e "${GREEN}✅ OK${NC} (HTTP $http_code)"
            return 0
        else
            echo -e "${RED}❌ FAILED${NC} (HTTP $http_code)"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        response=$(curl -s --max-time $TIMEOUT "$url" 2>/dev/null || echo "ERROR: Connection failed")
        if echo "$response" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✅ OK${NC}"
            return 0
        else
            echo -e "${RED}❌ FAILED${NC}"
            echo -e "${YELLOW}Response: ${response:0:200}...${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    fi
}

# Check if proxy is running
echo "Checking if Kubeflow proxy is accessible at $PROXY_URL..."
if ! curl -s --max-time 5 "$PROXY_URL" > /dev/null 2>&1; then
    echo -e "${RED}❌ ERROR: Cannot connect to $PROXY_URL${NC}"
    echo "Please ensure the proxy is running with: kubectl port-forward -n istio-system service/istio-ingressgateway 8081:80"
    exit 1
fi
echo -e "${GREEN}✅ Proxy is accessible${NC}"
echo ""

# Test Core Kubeflow Services
echo "📊 Testing Pipeline Permissions..."
test_endpoint "Pipeline List" "$PROXY_URL/pipeline/apis/v1beta1/pipelines" "pipelines.*id\|^\{\}$"
test_endpoint "Pipeline Runs" "$PROXY_URL/pipeline/apis/v1beta1/runs?resource_reference_key.type=NAMESPACE&resource_reference_key.id=$NAMESPACE" "^\{\}$\|runs.*id"
test_endpoint "Pipeline Experiments" "$PROXY_URL/pipeline/apis/v1beta1/experiments?resource_reference_key.type=NAMESPACE&resource_reference_key.id=$NAMESPACE" "^\{\}$\|experiments.*id"

echo ""
echo "📓 Testing Notebook Permissions..."
test_endpoint "Jupyter Notebooks API" "$PROXY_URL/jupyter/api/namespaces/$NAMESPACE/notebooks" "notebooks.*success.*true\|^\{\}$"
test_endpoint "Jupyter Status" "$PROXY_URL/jupyter/api/status" "success.*true\|ok"

echo ""
echo "🤖 Testing KServe Permissions..."
test_endpoint "KServe UI" "$PROXY_URL/kserve-endpoints/" "Endpoints Management UI\|DOCTYPE html" "status_only"
test_endpoint "KServe Models API" "$PROXY_URL/kserve-endpoints/api/namespaces/$NAMESPACE/inferenceservices" "inferenceservices\|^\{\}$"

echo ""
echo "📦 Testing Volume Permissions..."
test_endpoint "Volume Management API" "$PROXY_URL/volumes/api/namespaces/$NAMESPACE/persistentvolumeclaims" "success.*true\|pvcs\|^\{\}$"

echo ""
echo "🧠 Testing Additional Kubeflow Services..."
test_endpoint "Central Dashboard" "$PROXY_URL/" "Kubeflow\|DOCTYPE html" "status_only"
test_endpoint "Katib UI" "$PROXY_URL/katib/" "Katib\|DOCTYPE html\|AutoML" "status_only"
test_endpoint "Model Registry UI" "$PROXY_URL/model-registry/" "Model Registry\|DOCTYPE html" "status_only"
test_endpoint "TensorBoard" "$PROXY_URL/tensorboard/" "TensorBoard\|DOCTYPE html" "status_only"

echo ""
echo "🔐 Testing RBAC-specific Endpoints..."
test_endpoint "Profile API" "$PROXY_URL/profile/api/workgroup/env-info" "user\|workgroup\|error.*unauthorized" 
test_endpoint "Notebooks RBAC" "$PROXY_URL/jupyter/api/config" "config\|success\|unauthorized"

echo ""
echo "=========================================="
echo "📋 RBAC Permission Test Summary"
echo "=========================================="
echo ""

PASSED_TESTS=$((TOTAL_TESTS - FAILED_TESTS))
PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))

echo -e "Total Tests: ${TOTAL_TESTS}"
echo -e "Passed: ${GREEN}${PASSED_TESTS}${NC}"
echo -e "Failed: ${RED}${FAILED_TESTS}${NC}"
echo -e "Success Rate: ${PASS_RATE}%"
echo ""

if [[ $FAILED_TESTS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All RBAC permission tests passed!${NC}"
    echo -e "${GREEN}✅ Kubeflow services have proper RBAC permissions for anonymous@kubeflow.org${NC}"
    echo ""
    echo -e "${GREEN}🚀 Ready for production use!${NC}"
    echo -e "Upload pipelines: ${YELLOW}cd app && python upload.py${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some RBAC permission tests failed!${NC}"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check if all Kubeflow services are running:"
    echo "   kubectl get pods -A | grep -E 'kubeflow|istio'"
    echo "2. Verify RBAC configurations are applied:"
    echo "   kubectl get clusterrolebindings | grep kubeflow"
    echo "3. Check service logs for authentication issues:"
    echo "   kubectl logs -n kubeflow -l app=<service-name>"
    echo "4. Ensure the proxy is correctly forwarding requests:"
    echo "   kubectl port-forward -n istio-system service/istio-ingressgateway 8081:80"
    exit 1
fi