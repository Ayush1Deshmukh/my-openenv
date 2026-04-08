#!/bin/bash

# validate-submission.sh
# Validates the Legal Document Review OpenEnv submission

set -e

HF_SPACE_URL="${1:-http://localhost:7860}"
WORKSPACE_DIR="${2:-.}"

echo "========================================================"
echo "Legal Document Review OpenEnv — Validation Script"
echo "========================================================"
echo "Workspace: $WORKSPACE_DIR"
echo "Environment URL: $HF_SPACE_URL"
echo ""

# Check required files
echo "[1/5] Checking required files..."
required_files=("inference.py" "requirements.txt" "Dockerfile" "README.md")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$WORKSPACE_DIR/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file — MISSING"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Missing required files: ${missing_files[@]}"
    exit 1
fi

# Check Python syntax
echo ""
echo "[2/5] Checking Python syntax..."
if python3 -m py_compile "$WORKSPACE_DIR/inference.py" 2>/dev/null; then
    echo "  ✅ inference.py syntax OK"
else
    echo "  ❌ inference.py has syntax errors"
    exit 1
fi

# Check environment variables in inference.py
echo ""
echo "[3/5] Checking environment variables..."
if grep -q "API_BASE_URL = os.getenv" "$WORKSPACE_DIR/inference.py"; then
    echo "  ✅ API_BASE_URL configured"
else
    echo "  ❌ API_BASE_URL missing"
    exit 1
fi

if grep -q "MODEL_NAME = os.getenv" "$WORKSPACE_DIR/inference.py"; then
    echo "  ✅ MODEL_NAME configured"
else
    echo "  ❌ MODEL_NAME missing"
    exit 1
fi

if grep -q 'HF_TOKEN = os.getenv("HF_TOKEN")' "$WORKSPACE_DIR/inference.py"; then
    echo "  ✅ HF_TOKEN (no default) configured"
else
    echo "  ❌ HF_TOKEN not configured correctly"
    exit 1
fi

# Check logging format
echo ""
echo "[4/5] Checking logging format..."
if grep -q 'def log_start' "$WORKSPACE_DIR/inference.py" && \
   grep -q 'def log_step' "$WORKSPACE_DIR/inference.py" && \
   grep -q 'def log_end' "$WORKSPACE_DIR/inference.py"; then
    echo "  ✅ Logging functions present"
else
    echo "  ❌ Logging functions missing"
    exit 1
fi

if grep -q '\[START\]' "$WORKSPACE_DIR/inference.py" && \
   grep -q '\[STEP\]' "$WORKSPACE_DIR/inference.py" && \
   grep -q '\[END\]' "$WORKSPACE_DIR/inference.py"; then
    echo "  ✅ Log format markers present"
else
    echo "  ❌ Log format markers missing"
    exit 1
fi

# Check OpenAI client usage
echo ""
echo "[5/5] Checking OpenAI client..."
if grep -q "from openai import OpenAI" "$WORKSPACE_DIR/inference.py"; then
    echo "  ✅ OpenAI client imported"
else
    echo "  ❌ OpenAI client not imported"
    exit 1
fi

if grep -q "client.chat.completions.create" "$WORKSPACE_DIR/inference.py"; then
    echo "  ✅ OpenAI client used for completions"
else
    echo "  ❌ OpenAI client completions not found"
    exit 1
fi

echo ""
echo "========================================================"
echo "✅ ALL VALIDATION CHECKS PASSED"
echo "========================================================"
echo ""
echo "Submission is ready for deployment to Hugging Face Space:"
echo "  1. Create a new Space on huggingface.co"
echo "  2. Push code via git: git push origin main"
echo "  3. Set HF_TOKEN secret in Space settings"
echo "  4. Space will auto-build and run inference.py"
echo ""
