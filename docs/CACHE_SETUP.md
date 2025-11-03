# Model Cache Setup Guide

## Problem: Re-downloading 14GB Models

Without proper caching, the model would download every time you initialize the app. This wastes:
- ⏱️ **Time**: 5-10 minutes per download
- 💾 **Bandwidth**: 14GB for Mistral-7B
- 🔄 **Redundancy**: Multiple copies of the same model

## Solution: Local Project Cache

We've configured the app to use a **local cache directory** within the project:
```
activation_steering_lab/models_cache/
```

## How It Works

### 1. Cache Directory Setup
```python
# In model_wrapper.py
self.cache_dir = "activation_steering_lab/models_cache"
```

When loading models:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=self.cache_dir,  # Use local cache
    local_files_only=False      # Allow download if needed
)
```

### 2. Symlink to Global Cache
Instead of duplicating models, we create **symlinks**:

```bash
scripts/setup_local_cache.sh
```

This script:
1. Checks your global HuggingFace cache (`~/.cache/huggingface/hub/`)
2. Creates symlinks for any existing models
3. Saves disk space by reusing downloads

Result:
```
activation_steering_lab/models_cache/
└── models--mistralai--Mistral-7B-Instruct-v0.2 → ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2
```

## Setup Instructions

### Option 1: Using Existing Models (Recommended)
If you already have Mistral-7B downloaded:

```bash
# One-time setup
scripts/setup_local_cache.sh
```

✅ **Result**: Instant model loading, no re-download!

### Option 2: Fresh Download
If you don't have the model yet:

```bash
# Just run the app
scripts/run.sh

# Click "Initialize Model & Library"
# Model will download to local cache (~5-10 min first time)
# Subsequent runs will be instant
```

✅ **Result**: Model cached locally, fast on next run!

## Verification

Test that cache is working:
```bash
python test_cache.py
```

Expected output:
```
✓ Model wrapper created
✓ Cache directory exists
✓ Found 1 cached model(s):
  - models--mistralai--Mistral-7B-Instruct-v0.2 → /Users/.../hub/models--mistralai--Mistral-7B-Instruct-v0.2
✓ Cache configuration is correct
```

## Cache Locations

### Local Project Cache (Used by App)
```
activation_steering_lab/models_cache/
├── README.md
└── models--mistralai--Mistral-7B-Instruct-v0.2/ (symlink)
```

### Global HuggingFace Cache (Source)
```
~/.cache/huggingface/hub/
└── models--mistralai--Mistral-7B-Instruct-v0.2/
    ├── snapshots/
    ├── blobs/
    └── refs/
```

## Benefits

✅ **No Re-downloads**: Model persists between sessions
✅ **Fast Loading**: Loads in 2-3 seconds, not 5-10 minutes
✅ **Disk Efficient**: Symlinks save 14GB
✅ **Project Portable**: Everything in one directory
✅ **Git Friendly**: Cache ignored by version control

## Troubleshooting

### Cache Still Empty After Setup
```bash
# Check if global cache has the model
ls ~/.cache/huggingface/hub/ | grep Mistral

# If found, re-run setup
./setup_local_cache.sh

# If not found, model will download on first use
```

### Model Still Downloading Every Time
```bash
# Check cache directory
ls -lh activation_steering_lab/models_cache/

# Should show symlink to global cache
# If empty, run setup script
./setup_local_cache.sh
```

### Symlink Broken
```bash
# Remove broken symlink
rm activation_steering_lab/models_cache/models--mistralai--*

# Re-run setup
./setup_local_cache.sh
```

### Want to Use Different Cache Location
Edit `model_wrapper.py`:
```python
def __init__(self, cache_dir: str = "/path/to/your/cache"):
    self.cache_dir = cache_dir
```

### Clear Cache to Free Space
```bash
# Remove local cache (will re-download on next use)
rm -rf activation_steering_lab/models_cache/models--*

# Or remove global cache
rm -rf ~/.cache/huggingface/hub/models--mistralai--*
```

## Technical Details

### Why Symlinks?
- **Space**: Don't duplicate 14GB
- **Updates**: Global cache updates benefit local project
- **Flexibility**: Can switch between projects easily

### Why Local Cache at All?
- **Portability**: Project is self-contained
- **Reproducibility**: Exact model version locked
- **Testing**: Easy to test with different models
- **Offline**: Works without re-downloading

### Cache Priority
1. **Local project cache** (fastest)
2. **Global HuggingFace cache** (if local not found)
3. **Download from internet** (if neither found)

## Best Practices

✅ **Run setup_local_cache.sh** after installation
✅ **Keep global cache** (~/.cache/huggingface/) for reuse
✅ **Don't commit cache** to git (already in .gitignore)
✅ **Test with test_cache.py** before first run

❌ **Don't delete global cache** if using symlinks
❌ **Don't manually edit cache** (use HuggingFace CLI)
❌ **Don't commit models** to version control

## Migration Guide

### From Global Cache Only
```bash
# Before: Model in ~/.cache/huggingface/hub/ only
# After: Symlink from project cache to global cache

./setup_local_cache.sh
```

### From Manual Download
```bash
# If you downloaded model manually:
# 1. Move to global cache
mv /path/to/model ~/.cache/huggingface/hub/

# 2. Create symlink
./setup_local_cache.sh
```

## Disk Space Requirements

### With Symlinks (Recommended)
- **Project cache**: ~1KB (symlink)
- **Global cache**: ~14GB (Mistral) or ~7GB (Phi-3)
- **Total**: ~14GB

### Without Symlinks
- **Project cache**: ~14GB (full copy)
- **Global cache**: ~14GB (original)
- **Total**: ~28GB ❌ Wasteful!

## Summary

The cache system ensures:
1. ⚡ **Fast initialization** (seconds, not minutes)
2. 💾 **Efficient storage** (symlinks, no duplication)
3. 🔒 **Project isolation** (local cache, reproducible)
4. 🌐 **Works offline** (once cached)

**TL;DR**: Run `./setup_local_cache.sh` once, enjoy fast model loading forever! 🚀
