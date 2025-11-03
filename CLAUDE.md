# Activation Steering Learning Lab - Project Documentation

## Project Overview

An interactive educational tool for learning and experimenting with activation steering in Large Language Models (LLMs). Built with Python, Gradio, and PyTorch, designed to run on macOS M4 with 24GB RAM.

**Current Status**: Migrating from Mistral-7B to Phi-3-mini due to MPS memory issues on Apple Silicon.

## Core Concept

**Activation Steering**: A technique to manipulate language model behavior by injecting concept vectors into specific layers during inference, without retraining the model.

- **Concept Vector**: Computed as `activation(concept_prompt) - activation(baseline_prompt)`
- **Injection**: Adding the scaled concept vector to layer activations during generation
- **Result**: Model behavior changes (e.g., becomes more enthusiastic, technical, or concise)

## Architecture

### File Structure

```
activation_layers/
├── activation_steering_lab/
│   ├── main.py                    # Gradio UI (4 tabs: Education, Create, Playground, Advanced)
│   ├── model_wrapper.py           # Model loading with PyTorch hooks
│   ├── vector_library.py          # Concept vector storage/management
│   ├── injection_engine.py        # Steering logic (single/multi-concept)
│   ├── educational_content.py     # Learning materials and explanations
│   ├── models_cache/              # Local HuggingFace model cache (symlinked)
│   └── saved_vectors/             # Pre-computed concept vectors (51 vectors: 17 concepts × 3 layers)
├── tests/
│   ├── test_setup.py              # Verification script
│   ├── test_steering.py           # CLI test script (no Gradio)
│   └── test_cache.py              # Verify cache setup
├── docs/                          # Documentation
│   ├── QUICKSTART.md              # 5-minute setup guide
│   ├── CACHE_SETUP.md             # Model caching details
│   ├── PROJECT_SUMMARY.md         # Technical architecture overview
│   └── claude.md                  # This file - development notes
├── public/                        # Assets
│   └── img/demo.webp              # Screenshot for README
├── setup_local_cache.sh           # Create symlinks to avoid re-downloading models (root)
├── run.sh                         # Launch script (root)
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Modern Python packaging
└── README.md                      # User documentation
```

### Key Components

#### 1. model_wrapper.py
- Loads LLM from HuggingFace (currently Mistral-7B, migrating to Phi-3-mini)
- Registers PyTorch forward hooks on transformer layers
- Captures activations during forward pass
- Injects concept vectors at specified layers
- **Critical Issue**: MPS (Metal Performance Shaders) on M4 Mac hangs when loading 14GB models
- **Solution**: Use CPU fallback for large models, or switch to Phi-3-mini (3.8B params, ~7GB)

#### 2. vector_library.py
- Computes concept vectors via activation subtraction
- Saves/loads vectors to disk (`.pt` tensors + `.json` metadata)
- Currently has 17 pre-computed concepts:
  - Emotions: happy, sad, angry, calm
  - Styles: technical, casual, formal, enthusiastic
  - Tone: brief, verbose, confident, humble
  - Personalities: optimistic, pessimistic, creative, analytical, empathetic

#### 3. injection_engine.py
- `generate_comparison()`: Generates normal vs steered output side-by-side
- `generate_with_steering()`: Single steered generation
- `generate_with_multiple_concepts()`: Multi-concept steering (e.g., happy + technical)
- `analyze_layer_effects()`: Test steering across different layers
- `create_mixed_emotion()`: Weighted combination of concepts

#### 4. main.py (Gradio UI)
- **Education Tab**: Layer-by-layer explanations, interactive layer explorer
- **Create Concepts Tab**: Custom concept creation
- **Steering Playground Tab**: Real-time steering with side-by-side comparison
- **Advanced Experiments Tab**: Layer analysis, strength testing, emotion mixing

## Environment Setup

### Hardware
- **Device**: macOS M4 with 24GB unified memory
- **Target**: MPS (Metal Performance Shaders) for GPU acceleration
- **Reality**: CPU fallback required for Mistral-7B due to MPS memory issues

### Software
- Python 3.9+ (venv)
- PyTorch with MPS support
- Transformers (HuggingFace)
- Gradio 4.0+
- Model cache: `~/.cache/huggingface/hub/` (symlinked to project)

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./setup_local_cache.sh  # Symlink existing models
```

## Known Issues & Solutions

### Issue 1: MPS Memory Hang
**Problem**: Mistral-7B (14GB model) hangs when loading on MPS, even with 24GB RAM.

**Root Cause**:
- Model weights: ~14GB (float16)
- During `.to("mps")`, PyTorch temporarily needs:
  - Original model in CPU RAM
  - Copy being moved to MPS
  - Activation memory
  - **Total spike: 25-30GB+**
- MPS doesn't fail gracefully—just hangs

**Symptoms**:
```
Loading checkpoint shards: 100%|██████████| 3/3 [00:22<00:00, 7.45s/it]
Some parameters are on the meta device because they were offloaded to the disk.
[HANGS HERE - NO ERROR]
```

**Solutions Tried**:
1. ❌ `device_map="auto"` → Causes "meta device" errors
2. ❌ Load to CPU then `.to("mps")` → Hangs during transfer
3. ✅ **CPU fallback** → Works but slow
4. ✅ **Phi-3-mini (3.8B)** → Fits in MPS, fully compatible

**Current Fix**:
```python
# In model_wrapper.py
if actual_device == "mps":
    print(f"  ⚠️  MPS detected but using CPU to avoid memory issues")
    actual_device = "cpu"
```

### Issue 2: Empty Dropdowns in Gradio
**Problem**: Concept dropdowns appeared as text input fields instead of dropdowns.

**Root Cause**: Variable name mismatch in `update_all_dropdowns()`:
- Used `steer_concept_ref` (doesn't exist)
- Should be `steer_concept` (line 466)

**Fixed**: [main.py:595](activation_steering_lab/main.py#L595)
```python
init_btn.click(fn=update_all_dropdowns, outputs=[steer_concept, layer_concept, strength_concept])
```

### Issue 3: Vectors Not Loading on Restart
**Problem**: After extracting 51 concept vectors, they weren't loaded on restart.

**Root Cause**: `initialize()` re-extracted instead of calling `load_all()`.

**Fixed**: [main.py:84-93](activation_steering_lab/main.py#L84-L93)
```python
try:
    self.library.load_all()
    num_loaded = len(self.library.list_concepts())
    if num_loaded > 0:
        yield f"✓ Loaded {num_loaded} concepts from cache!"
except Exception:
    # Fall back to extracting
```

## Model Migration: Mistral-7B → Phi-3-mini

### Why Phi-3-mini?
1. ✅ **Size**: 3.8B params (~7GB) vs Mistral 7B (~14GB)
2. ✅ **MPS Compatible**: Runs on M4 Mac without hanging
3. ✅ **Activation Steering Support**: Microsoft published research (Oct 2024) confirming compatibility
4. ✅ **Same Architecture**: Transformer-based, our code works unchanged
5. ✅ **Apple Optimized**: Works with MLX framework for acceleration

### Model Details
- **HuggingFace ID**: `microsoft/Phi-3-mini-4k-instruct`
- **Context Length**: 4096 tokens
- **Layers**: 32 (vs Mistral's 32)
- **Memory**: ~7GB float16 (vs Mistral's ~14GB)
- **Performance**: Similar quality for instruction-following tasks

### Migration Steps
1. Update `model_wrapper.py` default model
2. Update layer count expectations (if different)
3. Re-extract concept vectors for Phi-3 (17 concepts × 3 layers = 51 vectors)
4. Update documentation
5. Test all features (steering, comparison, multi-concept)

## Cache System

### Problem
Without caching, models re-download every time (14GB × multiple runs = waste).

### Solution
Local project cache with symlinks to global HuggingFace cache.

**Structure**:
```
activation_steering_lab/models_cache/
└── models--mistralai--Mistral-7B-Instruct-v0.2/  [symlink]
    → ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/
```

**Setup**:
```bash
./setup_local_cache.sh
```

**Verification**:
```bash
python tests/test_cache.py
```

**Benefits**:
- ✅ No re-downloads (model persists between sessions)
- ✅ Fast loading (2-3 seconds vs 5-10 minutes)
- ✅ Disk efficient (symlinks save 14GB)
- ✅ Git-friendly (cache ignored by .gitignore)

## Concept Vectors

### Pre-computed Concepts
Located in `activation_steering_lab/saved_vectors/`:

| Category | Concepts |
|----------|----------|
| **Emotions** | happy, sad, angry, calm |
| **Styles** | technical, casual, formal, enthusiastic |
| **Tone** | brief, verbose, confident, humble |
| **Personalities** | optimistic, pessimistic, creative, analytical, empathetic |

**Storage**: 51 files (17 concepts × 3 layers)
- `{concept}_layer{N}_tensor.pt` (9.6KB each)
- `{concept}_layer{N}_meta.json` (~220 bytes each)

### Extraction Process
1. Select recommended layers (typically layers 8, 12, 16)
2. For each concept:
   - Generate activation for concept prompt
   - Generate activation for baseline prompt
   - Compute difference: `concept_vector = concept_act - baseline_act`
3. Save to disk (instant loading on next run)

**Time**: ~3 minutes first time, instant thereafter

## Usage Examples

### CLI Testing (No Gradio)
```bash
python tests/test_steering.py
```

This tests:
1. Model loading from cache
2. Vector library loading (17 concepts)
3. Injection engine setup
4. Generation comparison (normal vs steered with "happy")

### Gradio Interface
```bash
./run.sh
```

**Tabs**:
1. **Education**: Learn about layers and activation steering
2. **Create Concepts**: Make custom concept vectors
3. **Playground**: Try steering with any concept
4. **Advanced**: Layer analysis, strength testing, emotion mixing

## Development Notes

### Adding New Concepts
1. Go to "Create Concepts" tab
2. Enter concept name (e.g., "sarcastic")
3. Provide concept prompt (e.g., "Respond with heavy sarcasm")
4. Provide baseline prompt (e.g., "Respond normally")
5. Select layer (default: 16)
6. Click "Extract Concept"
7. Vector saved automatically to `saved_vectors/`

### Debugging
- Check cache: `python tests/test_cache.py`
- Test steering: `python tests/test_steering.py`
- View logs: Model loading prints detailed progress
- Check vectors: `ls -lh activation_steering_lab/saved_vectors/`

### Performance Tuning
- **CPU**: Slow but stable (Mistral-7B ~30s/generation)
- **MPS**: Fast but may hang with large models (Phi-3 ~5s/generation)
- **Layers**: Middle layers (8-16) work best for most concepts
- **Strength**: 1.5-3.0 range typically effective

## Next Steps

1. **Immediate**: Complete migration to Phi-3-mini
2. **Testing**: Verify all features work with new model
3. **Optimization**: Enable MPS for Phi-3 (should work without hanging)
4. **Enhancement**: Add on-demand extraction (Option 3 from earlier discussion)
5. **Documentation**: Update README with Phi-3 specifics

## References

### Research
- **Activation Steering**: [Microsoft AI Research (Oct 2024)](https://www.marktechpost.com/2024/10/22/microsoft-ai-introduces-activation-steering-a-novel-ai-approach-to-improving-instruction-following-in-large-language-models/)
- **Phi-3 Technical Report**: [arXiv:2404.14219](https://arxiv.org/abs/2404.14219)

### Models
- **Mistral-7B**: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Phi-3-mini**: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

### Apple Silicon
- **MPS Backend**: [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- **MLX Framework**: [Apple MLX for Phi-3](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/accelerate-phi-3-use-on-macos-a-beginners-guide-to-using-apple-mlx-framework/4174656)

## Troubleshooting

### "Model hangs during loading"
- **Cause**: MPS memory allocation issue
- **Fix**: CPU fallback enabled automatically
- **Future**: Use Phi-3-mini (fits in MPS)

### "Concept dropdown is empty"
- **Cause**: Vectors not loaded or dropdown update failed
- **Fix**: Click "Initialize Model & Library" button
- **Verify**: Check `saved_vectors/` directory has files

### "Generation is very slow"
- **Cause**: Running on CPU instead of MPS
- **Expected**: 20-30 seconds per generation with Mistral-7B on CPU
- **Improvement**: Use Phi-3-mini on MPS (~5 seconds)

### "ModuleNotFoundError: No module named 'activation_steering_lab'"
- **Cause**: Not running from project root
- **Fix**: `cd /path/to/activation_layers && python -m activation_steering_lab.main`

## Git Status

**Tracked Files**: All core code, documentation, scripts
**Ignored**:
- `venv/` (virtual environment)
- `*.pyc, __pycache__/` (Python cache)
- `activation_steering_lab/saved_vectors/*.pt, *.json` (concept vectors)
- `activation_steering_lab/models_cache/*` (model cache)
- `.DS_Store` (macOS)

**Commit Strategy**:
- Commit code changes, not model weights or vectors
- Document major changes in commit messages
- Use branches for experimental features

## Contact & Support

**User**: franzuzz (24GB M4 Mac)
**AI Assistant**: Claude (Sonnet 4.5)
**Session**: Building activation steering educational tool
**Last Updated**: 2025-11-03

---

**Note**: This is a living document. Update as the project evolves, especially during the Phi-3 migration.
