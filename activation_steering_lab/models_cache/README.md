# Models Cache Directory

This directory stores cached HuggingFace models to avoid re-downloading on each use.

## Setup

Run from the project root:
```bash
./setup_local_cache.sh
```

This script will:
1. Check your global HuggingFace cache (`~/.cache/huggingface/hub/`)
2. Create symlinks to any already-downloaded models
3. Save ~14GB by reusing existing downloads

## How It Works

When you initialize the app:
- First time: Downloads Mistral-7B (~14GB) or Phi-3 (~7GB) to this directory
- Subsequent times: Loads instantly from local cache

## Contents

You should see:
- `models--mistralai--Mistral-7B-Instruct-v0.2/` - Main model (14GB)
- `models--microsoft--Phi-3-mini-4k-instruct/` - Fallback model (7GB)

## Why Local Cache?

✅ **No re-downloading**: Model persists between sessions
✅ **Project-local**: Everything self-contained
✅ **Fast initialization**: Loads in seconds, not minutes
✅ **Disk space efficient**: Uses symlinks when possible

## Troubleshooting

### Cache is empty
- Run `./setup_local_cache.sh` from project root
- Or just launch the app - it will download on first use

### Model still downloading
- Check symlinks: `ls -lh activation_steering_lab/models_cache/`
- If broken, delete and re-run setup script

### Out of disk space
- Mistral-7B: ~14GB
- Phi-3: ~7GB
- Free up space and re-download

## Git Ignore

This directory is in `.gitignore` - models won't be committed to version control.
