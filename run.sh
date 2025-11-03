#!/bin/bash
# Launch script for Activation Steering Lab

echo "🧠 Activation Steering Learning Lab"
echo "=================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if cache is set up
CACHE_DIR="activation_steering_lab/models_cache"
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR 2>/dev/null | grep -v README)" ]; then
    echo "💡 Tip: Setup local cache to avoid re-downloading models"
    echo "   Run: ./setup_local_cache.sh"
    echo ""
fi

# Activate venv and run
echo "🚀 Launching Gradio interface..."
echo ""
source venv/bin/activate
python -m activation_steering_lab.main
