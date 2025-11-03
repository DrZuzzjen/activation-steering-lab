# 🧠 Activation Steering Learning Lab

An interactive educational tool to understand and experiment with **activation steering** in large language models.

## What is Activation Steering?

Activation steering is a technique to control language model behavior by injecting "concept vectors" into the model's internal layers during generation. Think of it as adjusting the model's "mood" or "thinking style" in real-time!

**Key Idea:**
```python
concept_vector = activation("happy text") - activation("neutral text")
steered_output = generate(prompt, inject=concept_vector)
```

## Features

### 📚 Educational Components
- **Layer Explorer**: Understand what happens at each transformer layer
- **Concept Creator**: Extract custom concept vectors from text pairs
- **Interactive Tutorials**: Learn through hands-on experimentation
- **Visual Comparisons**: See normal vs steered outputs side-by-side

### 🎮 Experimentation Tools
- **Steering Playground**: Try different concepts, layers, and strengths
- **Layer Analysis**: Compare effects across all layers
- **Strength Explorer**: Understand the strength parameter
- **Emotion Mixer**: Combine multiple concepts

### 🎯 Pre-loaded Concepts
- **Emotions**: happy, sad, angry, fearful, excited, calm
- **Styles**: formal, casual, poetic, technical
- **Personalities**: pirate, shakespeare, enthusiastic, pessimistic
- **Brevity**: brief, verbose

## Installation

### Requirements
- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.9+**
- **24GB RAM** (for Mistral-7B) or 16GB (for Phi-3)

### Setup

```bash
# Clone or navigate to the project
cd activation_layers

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup local model cache (avoids re-downloading 14GB!)
./setup_local_cache.sh
```

**Important**: The `setup_local_cache.sh` script creates symlinks to any already-downloaded HuggingFace models in your global cache. This avoids re-downloading Mistral-7B (~14GB) if you already have it!

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python -m activation_steering_lab.main
```

The Gradio interface will open in your browser at `http://127.0.0.1:7860`

### First Steps

1. **Click "Initialize Model & Library"** - This loads Mistral-7B and creates concept vectors (~5 min)
2. **Explore the Layer Education tab** - Learn how layers work
3. **Try the Steering Playground** - Generate text with steering!

## How to Use

### Basic Steering

1. Go to **Steering Playground**
2. Enter a prompt: `"Tell me about the weather"`
3. Select a concept: `happy`
4. Choose a layer: `16` (middle layers work best!)
5. Set strength: `2.0`
6. Click **Generate!**

Compare the outputs - the steered version will be happier!

### Creating Custom Concepts

1. Go to **Create Concepts**
2. Enter:
   - **Name**: `enthusiastic`
   - **Concept Prompt**: `I am SO excited and thrilled about this!`
   - **Baseline Prompt**: `I am neutral about this.`
   - **Layer**: `16`
3. Click **Extract Concept Vector**

### Advanced Experiments

#### Layer Analysis
See how the same concept affects different layers:
```
Prompt: "In my opinion,"
Concept: enthusiastic
Strength: 2.0
```

#### Mixing Concepts
Combine multiple concepts:
```
happy:0.7,excited:0.3 → cheerful
formal:0.5,friendly:0.5 → professional-warm
```

## Project Structure

```
activation_layers/
├── activation_steering_lab/
│   ├── __init__.py
│   ├── main.py                 # Gradio interface
│   ├── model_wrapper.py        # Model loading & hooks
│   ├── vector_library.py       # Concept storage
│   ├── injection_engine.py     # Steering logic
│   ├── educational_content.py  # Tutorials & explanations
│   └── saved_vectors/          # Pre-computed concepts
├── requirements.txt
└── README.md
```

## Technical Details

### Model Architecture

The app supports:
- **Mistral-7B-Instruct-v0.2** (default, best quality)
- **Phi-3-mini-4k-instruct** (fallback, lower memory)

Loaded with:
- `torch.float16` for memory efficiency
- `device_map="auto"` for optimal M4 chip usage
- Forward hooks for activation capture/injection

### How It Works

1. **Extraction Phase**:
   ```python
   concept_activation = model(concept_prompt)[layer]
   baseline_activation = model(baseline_prompt)[layer]
   concept_vector = concept_activation - baseline_activation
   ```

2. **Injection Phase**:
   ```python
   def hook(module, input, output):
       output[:, -1, :] += strength * concept_vector
       return output
   ```

3. **Generation**:
   - Hook modifies activations in real-time
   - Model generates with "steered" thinking
   - Hook is removed after generation

### Memory Management

- Uses MPS (Metal Performance Shaders) for Apple Silicon
- Automatic garbage collection
- Efficient hook cleanup
- float16 precision reduces memory by 50%

## Educational Goals

By using this tool, you'll learn:

✅ **What activations are** (vs embeddings)
✅ **How transformer layers process information**
✅ **Where concepts live in the network**
✅ **Why middle layers work best for steering**
✅ **How to balance injection strength**
✅ **The importance of good baselines**
✅ **How to debug and experiment with LLMs**

## Example Experiments

### 1. Emotion Injection
```
Prompt: "The meeting went"
Concept: happy (strength 2.0, layer 16)
→ "The meeting went wonderfully! Everyone was engaged..."

Concept: sad (strength 2.0, layer 16)
→ "The meeting went poorly. I felt discouraged..."
```

### 2. Style Transfer
```
Prompt: "Quantum mechanics is"
Concept: pirate (strength 3.0, layer 18)
→ "Quantum mechanics be a strange beast, matey! Arrr..."
```

### 3. Brevity Control
```
Prompt: "The main idea is"
Concept: brief (strength 4.0, layer 20)
→ "The main idea is simple."

Concept: verbose (strength 2.0, layer 20)
→ "The main idea is, fundamentally speaking, and in great detail..."
```

### 4. Layer Comparison
Test `happy` at layers 5, 15, 25:
- Layer 5: Minor word choice changes
- Layer 15: Clear emotional shift ⭐
- Layer 25: Minimal effect (too late)

## Troubleshooting

### Out of Memory Error
```bash
# The app will automatically try Phi-3 if Mistral fails
# Or manually edit model_wrapper.py to start with Phi-3
```

### Model Loading Slow
- First run downloads ~14GB (Mistral) or ~7GB (Phi-3)
- Cached for future runs
- Concept extraction takes ~5 minutes

### Steering Not Working
- ✅ Use middle layers (50-75% depth)
- ✅ Try strength 2.0-3.0 first
- ✅ Check concept was created successfully
- ✅ Use clear, distinct concepts

### Generation Incoherent
- ⬇️ Lower strength (try 1.0-2.0)
- ⬇️ Try earlier layer
- ⬇️ Use simpler concepts

## Performance Tips

- **Best layers**: 12-24 (for 32-layer models)
- **Best strength**: 1.5-3.0 for most concepts
- **Best concepts**: Strong emotions, distinct styles
- **Temperature**: 0.7 (default) balances creativity and coherence
- **Max tokens**: 50 is good for comparisons

## Research Background

This tool is inspired by:
- [Anthropic's Representation Engineering](https://www.anthropic.com/research)
- [Activation Engineering Papers](https://arxiv.org/abs/2308.10248)
- Interpretability research on transformer internals

## Limitations

- Works best with clear, distinct concepts
- Very abstract concepts may not steer well
- Extremely high strength can break coherence
- Some concepts are layer-dependent
- Results vary by model architecture

## Contributing

This is an educational tool! Feel free to:
- Add new pre-defined concepts
- Create better baselines
- Improve the UI
- Add more experiments
- Write better explanations

## License

MIT License - Feel free to use for learning and education!

## Citation

If you use this tool in research or teaching:
```
Activation Steering Learning Lab (2024)
An educational tool for understanding activation steering in LLMs
```

## Learn More

### Key Concepts
- **Activations**: Internal neural network values at each layer
- **Embeddings**: Static word→vector lookup (layer 0 only)
- **Hooks**: PyTorch functions that intercept forward passes
- **Steering**: Modifying activations to change behavior

### Recommended Reading Order
1. Layer Education tab → Understand transformer layers
2. Activations vs Embeddings → Core concept
3. Why Addition? → Why we don't replace
4. Create a simple concept → Hands-on practice
5. Steering Playground → See it work!
6. Layer Analysis → Deep understanding
7. Advanced Experiments → Master level

## Acknowledgments

Built with:
- 🤗 Transformers (HuggingFace)
- 🔥 PyTorch
- 🎨 Gradio
- 🍎 Apple Silicon MPS

---

**Happy Steering!** 🎯

If you discover interesting concepts or experiments, share them with others learning about LLM internals!
