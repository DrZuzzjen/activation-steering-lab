#!/bin/bash
# Setup local model cache by symlinking to global HuggingFace cache
# This avoids re-downloading models and keeps them available locally

echo "🔗 Setting up local model cache..."
echo ""

# Change to script's directory (project root)
cd "$(dirname "$0")"

# Define paths
GLOBAL_CACHE="$HOME/.cache/huggingface/hub"
LOCAL_CACHE="activation_steering_lab/models_cache"

# Create local cache directory
mkdir -p "$LOCAL_CACHE"

# Check if global cache exists
if [ ! -d "$GLOBAL_CACHE" ]; then
    echo "❌ Global HuggingFace cache not found at: $GLOBAL_CACHE"
    echo "Models will be downloaded on first use."
    exit 0
fi

echo "📦 Found global cache: $GLOBAL_CACHE"
echo "📁 Local cache: $LOCAL_CACHE"
echo ""

# Function to link a model
link_model() {
    local model_dir=$1
    local model_name=$(basename "$model_dir")

    if [ -d "$GLOBAL_CACHE/$model_dir" ]; then
        if [ -L "$LOCAL_CACHE/$model_name" ] || [ -e "$LOCAL_CACHE/$model_name" ]; then
            echo "  ✓ $model_name (already linked)"
        else
            ln -s "$GLOBAL_CACHE/$model_dir" "$LOCAL_CACHE/$model_name"
            echo "  ✓ Linked $model_name"
        fi
    fi
}

# Link Mistral model if it exists
echo "Checking for cached models..."
link_model "models--mistralai--Mistral-7B-Instruct-v0.2"
link_model "models--microsoft--Phi-3-mini-4k-instruct"

echo ""
echo "✅ Local cache setup complete!"
echo ""
echo "Models will now load from: $LOCAL_CACHE"
echo "This avoids re-downloading on each initialization."
echo ""

# Show cache size
if [ -d "$LOCAL_CACHE" ]; then
    echo "Cache contents:"
    ls -lh "$LOCAL_CACHE" 2>/dev/null || echo "  (empty - will populate on first use)"
fi
