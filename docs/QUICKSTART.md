# Quick Start Guide

## Installation (One-Time Setup)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies (takes ~2 minutes)
pip install -r requirements.txt

# 4. Setup local model cache (avoids re-downloading)
scripts/setup_local_cache.sh

# 5. Test the setup
python -m pytest tests/test_setup.py -v
```

**Note**: Step 4 creates a local cache for models. If you've already downloaded Mistral-7B or Phi-3, it will symlink them instead of re-downloading (~14GB saved!).

## Running the App

```bash
# Option 1: Use the launch script
scripts/run.sh

# Option 2: Manual launch
source venv/bin/activate
python -m activation_steering_lab.main
```

The app will open at: http://127.0.0.1:7860

## First Time Use

### Step 1: Initialize (5 minutes)
1. Click the **"🚀 Initialize Model & Library"** button
2. Wait for the model to download (~14GB for Mistral, ~7GB for Phi-3)
3. Wait for concept vectors to be extracted

You'll see:
```
✓ Model loaded: mistralai/Mistral-7B-Instruct-v0.2
✓ Layers: 32
✓ Concepts: 10
✓ Recommended layers: [8, 12, 16, 20, 24]
```

### Step 2: Your First Steering (30 seconds)
1. Go to **"🎮 Steering Playground"** tab
2. Fill in:
   - **Prompt**: `Tell me about the weather`
   - **Concept**: `happy` (from dropdown)
   - **Layer**: `16` (recommended)
   - **Strength**: `2.0`
3. Click **"Generate!"**
4. Compare the two outputs!

### Step 3: Explore
- Try different concepts: `sad`, `pirate`, `formal`
- Try different layers: see how 5, 15, 25 differ
- Try different strengths: 0.5, 2.0, 5.0

## Example Experiments

### Make a pirate explain quantum physics
```
Prompt: "Quantum mechanics works by"
Concept: pirate
Layer: 18
Strength: 3.0
```

### Make sad weather reports
```
Prompt: "The weather today is"
Concept: sad
Layer: 16
Strength: 2.5
```

### Force brevity
```
Prompt: "Let me explain in detail how"
Concept: brief
Layer: 20
Strength: 4.0
```

## Creating Custom Concepts

1. Go to **"🎨 Create Concepts"** tab
2. Example - Creating "nervous":
   ```
   Name: nervous
   Concept Prompt: "I'm so nervous and anxious about this!"
   Baseline Prompt: "I feel neutral about this."
   Layer: 16
   ```
3. Click **"Extract Concept Vector"**
4. Use it in the Steering Playground!

## Tips for Best Results

✅ **DO:**
- Use middle layers (12-20 for 32-layer models)
- Start with strength 2.0
- Use clear, distinct concepts (happy, sad, formal)
- Match baseline structure to concept prompt

❌ **DON'T:**
- Use very early layers (< 5) - won't work well
- Use very late layers (> 28) - concepts already decided
- Use strength > 5.0 - usually breaks coherence
- Use vague concepts - won't steer effectively

## Advanced Features

### Layer Analysis
Test concept across all layers:
1. Go to **Advanced Experiments → Layer Analysis**
2. Enter prompt and concept
3. Click **"Analyze Layers"**
4. See which layers work best!

### Emotion Mixer
Combine multiple concepts:
1. Go to **Advanced Experiments → Emotion Mixer**
2. Enter: `happy:0.7,excited:0.3`
3. Layer: 16
4. Click **"Mix Concepts"**
5. Use in Steering Playground!

## Troubleshooting

### "Out of memory" error
- App will auto-fallback to Phi-3 (smaller model)
- Or close other apps and retry

### Model download is slow
- First run downloads ~14GB
- Subsequent runs use cached model
- Takes 5-10 min on fast connection

### Steering doesn't seem to work
- Check you're using middle layers (12-20)
- Try increasing strength to 3.0
- Make sure concept was created successfully
- Some prompts are more steerable than others

### App won't start
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Test setup
python -m pytest tests/test_setup.py -v
```

## Learning Path

### Beginner (30 minutes)
1. ✅ Run first steering experiment
2. ✅ Try 3 different concepts
3. ✅ Read "Layer Education" tab
4. ✅ Understand activations vs embeddings

### Intermediate (1 hour)
1. ✅ Create custom concept
2. ✅ Run layer analysis
3. ✅ Test strength range
4. ✅ Understand why middle layers work

### Advanced (2+ hours)
1. ✅ Mix multiple concepts
2. ✅ Test opposing concepts at different layers
3. ✅ Extract 5+ custom concepts
4. ✅ Experiment with edge cases

## Getting Help

### In the App
- Read explanations in "📚 Layer Education"
- Check "💡 Tips & Tricks" accordion
- Look at recommended experiments

### Common Questions

**Q: Why isn't my concept working?**
A: Check layer (use 12-20), strength (try 2.0-3.0), and make sure concept prompt is distinct from baseline.

**Q: What's the best layer?**
A: Usually around 50-70% of total depth. For 32 layers: 16-24.

**Q: Can I steer behavior permanently?**
A: No - steering only affects current generation. This is a feature!

**Q: Why do results vary?**
A: Temperature adds randomness. Try temperature=0.5 for more consistent results.

## Next Steps

1. 🎯 Master basic steering
2. 🎨 Create your own concepts
3. 🔬 Run systematic experiments
4. 📚 Read the full README.md
5. 🧪 Try advanced multi-concept injection

---

**Have fun steering!** 🚀

For detailed documentation, see [README.md](README.md)
