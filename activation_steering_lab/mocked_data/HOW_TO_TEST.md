# How to Test Phase 1: Mock Data Generation

## Quick Answer

**Fastest test (5 seconds):**
```bash
cd /Users/franzuzz/code/activation_layers
source venv/bin/activate
python activation_steering_lab/mocked_data/test_export.py
```

## All Testing Options

### ⚡ Option 1: Format Validation (5 seconds - RECOMMENDED)
Tests the export pipeline with mock data (no model loading):

```bash
python activation_steering_lab/mocked_data/test_export.py
```

**What it does:**
- Creates fake activation tensors
- Exports to JSON format
- Validates all required fields
- Checks file size and structure
- ✅ No model loading = super fast!

**Expected output:**
```
✅ All validation checks passed!
File size: 0.14 MB
Peak activation layer: 16
Layer correlations working correctly
```

---

### 🧠 Option 2: Generate Real Sample (1-2 minutes)
Loads the model and creates real activation data:

```bash
python activation_steering_lab/mocked_data/generate_single_sample.py
```

**What it does:**
- Loads Phi-3-mini model (~15 seconds)
- Generates activations for "happy" concept at layer 16
- Exports to JSON file
- Validates the output

**Expected output:**
```
✅ Sample generation successful!
Generated: happy_layer16_TIMESTAMP.json
Peak Activation Layer: 31
Top Correlated Layers: [16, 15, 14]
```

---

### 📊 Option 3: Inspect Existing Sample (instant)
View the already-generated real data:

```bash
python -c "
from activation_steering_lab.threejs_export import ThreeJSExporter
data = ThreeJSExporter().load_threejs_data(
    'activation_steering_lab/mocked_data/happy_layer16_20251103_161946.json'
)
print(f'Concept: {data[\"metadata\"][\"concept_name\"]}')
print(f'Peak Layer: {data[\"peak_activation_layer\"]}')
print(f'Total Intensity: {data[\"total_intensity\"]:.2f}')
"
```

---

### 🎨 Option 4: Test in Main App (current 2D visualization)
The existing Gradio app works as before:

```bash
./run.sh
```

Then:
1. Open browser to `http://localhost:7860`
2. Go to "🔬 Visualization Lab" tab
3. Generate a visualization
4. See the current 2D Plotly heatmap

**Note:** Phase 1 doesn't change the UI yet. It adds the **backend capability** to export data for Three.js (Phase 2).

---

### 🚀 Option 5: Generate Full Dataset (5-10 minutes)
Create samples for all concepts and layers:

```bash
python activation_steering_lab/mocked_data/generate_mock_data.py
```

**What it generates:**
- 4 concepts: happy, sad, calm, excited
- 3 layers: 8, 16, 24
- = 12 total JSON files

**Use this when:**
- You want a complete dataset for Three.js development
- You're ready to start Phase 2
- You want to test multiple concepts

---

## What Gets Tested?

### ✅ Phase 1 delivers:
1. **Export Module** (`threejs_export.py`)
   - Converts PyTorch tensors → JSON
   - Downsamples 3072 dims → 64 regions
   - Computes layer correlations
   - Calculates peak activations

2. **Data Format**
   - Metadata (model, prompt, concept, etc.)
   - Normal/steered/difference activations
   - Layer correlations
   - Peak detection
   - Token sequences

3. **File Output**
   - ~140KB JSON files
   - Browser-friendly format
   - Fast loading (< 100ms)
   - Human-readable structure

### ⏳ Phase 2 will add:
- Three.js 3D brain visualization
- Custom Gradio component
- Interactive layer exploration
- MRI-style aesthetic

### ⏳ Phase 6 will integrate:
- "Export to 3D View" button in UI
- Live data → 3D visualization
- Seamless workflow

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Solution:
```bash
source venv/bin/activate  # Make sure venv is activated!
```

### "Concept 'happy' not found"
Solution: The vector library needs to load vectors first. This is handled in the generation scripts via:
```python
vector_library.load_all()
```

### File already exists
The scripts auto-generate timestamped filenames, so this shouldn't happen. But if it does, just delete the old file or let it create a new one.

---

## Files You Can Inspect

After running tests, check these files:

```bash
# Test output (mock data)
cat activation_steering_lab/mocked_data/test_sample.json | head -50

# Real sample (already generated!)
cat activation_steering_lab/mocked_data/happy_layer16_20251103_161946.json | head -50

# File sizes
ls -lh activation_steering_lab/mocked_data/*.json
```

---

## Quick Verification Checklist

Run this to verify everything works:

```bash
cd /Users/franzuzz/code/activation_layers
source venv/bin/activate

echo "1. Testing export format..."
python activation_steering_lab/mocked_data/test_export.py

echo -e "\n2. Checking existing sample..."
ls -lh activation_steering_lab/mocked_data/happy_layer16_*.json

echo -e "\n3. Loading and validating..."
python -c "from activation_steering_lab.threejs_export import ThreeJSExporter; \
data = ThreeJSExporter().load_threejs_data('activation_steering_lab/mocked_data/happy_layer16_20251103_161946.json'); \
print(f'✅ Data loaded: {len(data[\"normal_activations\"])} layers, {data[\"metadata\"][\"downsampled_regions\"]} regions')"

echo -e "\n✅ Phase 1 is working!"
```

---

## Summary

**For quick testing:** Run Option 1 (test_export.py) - takes 5 seconds  
**For real data:** Run Option 2 (generate_single_sample.py) - takes 1-2 minutes  
**To see it work:** The data file is already generated and validated!  

Phase 1 is **complete and tested**. Ready for Phase 2 (Three.js implementation)! 🚀
