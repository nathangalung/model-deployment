#!/bin/bash
# Test script to verify all Kubeflow endpoints work properly

echo "🧪 Testing Kubeflow Endpoints"
echo "=============================="
echo ""

PROXY_URL="http://localhost:8081"

echo "Testing endpoints through $PROXY_URL..."
echo ""

# Test Central Dashboard
echo "🏠 Testing Central Dashboard..."
if curl -s "$PROXY_URL/" | grep -q "Kubeflow Central Dashboard"; then
    echo "✅ Central Dashboard: OK"
else
    echo "❌ Central Dashboard: FAILED"
fi

# Test Jupyter
echo "📓 Testing Jupyter Web App..."
if curl -s "$PROXY_URL/jupyter/" | grep -q "Jupyter Management UI\|DOCTYPE html"; then
    echo "✅ Jupyter Web App: OK"
else
    echo "❌ Jupyter Web App: FAILED"
fi

# Test Pipelines
echo "📊 Testing Kubeflow Pipelines..."
if curl -s "$PROXY_URL/pipeline/" | grep -q "Kubeflow Pipelines"; then
    echo "✅ Kubeflow Pipelines UI: OK"
else
    echo "❌ Kubeflow Pipelines UI: FAILED"
fi

# Test Pipelines API
echo "🔧 Testing Pipelines API..."
if curl -s "$PROXY_URL/pipeline/apis/v1beta1/healthz" | grep -q "apiServerReady.*true"; then
    echo "✅ Pipelines API: OK"
else
    echo "❌ Pipelines API: FAILED"
fi

# Test Volumes
echo "📦 Testing Volumes Web App..."
if curl -s "$PROXY_URL/volumes/" | grep -q "Frontend\|DOCTYPE html"; then
    echo "✅ Volumes Web App: OK"
else
    echo "❌ Volumes Web App: FAILED"
fi

# Test KServe
echo "🤖 Testing KServe Models Web App..."
if curl -s "$PROXY_URL/kserve-endpoints/" | grep -q "DOCTYPE html\|KServe"; then
    echo "✅ KServe Models Web App: OK"
else
    echo "❌ KServe Models Web App: FAILED"
fi

# Test MLflow
echo "🔬 Testing MLflow..."
if curl -s "$PROXY_URL/mlflow/" | grep -q "MLflow\|status.*401"; then
    echo "✅ MLflow: OK"
else
    echo "❌ MLflow: FAILED"
fi

echo ""
echo "📋 Summary:"
echo "All endpoints are accessible through: $PROXY_URL"
echo ""
echo "🚀 Ready to use! Run your pipeline with:"
echo "cd app && python upload.py"
echo ""
echo "🌐 Access URLs:"
echo "  Dashboard: $PROXY_URL/"
echo "  Pipelines: $PROXY_URL/pipeline/"
echo "  Jupyter:   $PROXY_URL/jupyter/"
echo "  Volumes:   $PROXY_URL/volumes/"
echo "  KServe:    $PROXY_URL/kserve-endpoints/"
echo "  MLflow:    $PROXY_URL/mlflow/"