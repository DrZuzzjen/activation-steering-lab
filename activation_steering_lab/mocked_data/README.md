# Mocked Data for Three.js Development

This directory contains pre-captured activation data for rapid Three.js visualization development.

## Purpose

Instead of reloading the 14GB Phi-3 model and running inference every time we iterate on the 3D visualization, we:
1. Run the model ONCE to capture real activations
2. Save to compact JSON format (~2-5MB per file)
3. Use these JSON files for all Three.js development
4. Enable fast iteration without model overhead

## Data Format

Each JSON file contains:

```json
{
  "metadata": {
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "Tell me about happiness",
    "concept_name": "happy",
    "injection_layer": 16,
    "injection_strength": 2.0,
    "sequence_length": 15,
    "num_layers": 32,
    "hidden_size": 3072,
    "downsampled_regions": 64,
    "timestamp": "2025-11-03T15:00:00Z"
  },
  "normal_activations": {
    "0": [64 region values],
    "1": [64 region values],
    ...
    "31": [64 region values]
  },
  "steered_activations": { ... },
  "difference_activations": { ... },
  "tokens": ["Tell", "me", "about", ...],
  "layer_correlations": {
    "0": 0.12,
    "16": 1.0,  // Injection layer
    "17": 0.87,
    ...
  },
  "peak_activation_layer": 16,
  "total_intensity": 458.3,
  "top_correlated_layers": [16, 17, 18]
}
```

## Downsampling

- Original: 3072 dimensions per layer
- Downsampled: 64 regions per layer (~48 dims averaged per region)
- Why: Keeps file size manageable and enables real-time 3D rendering

## Usage

### Generate Mock Data

```bash
# From project root
source venv/bin/activate
python activation_steering_lab/mocked_data/generate_mock_data.py
```

This will generate samples for:
- 4 concepts: happy, sad, calm, excited
- 3 layers: 8 (early), 16 (middle), 24 (late)
- Total: 12 files

### Test Export Format

```bash
# Validate JSON structure without running model
python activation_steering_lab/mocked_data/test_export.py
```

### Load in Three.js

```javascript
// Frontend code
fetch('path/to/happy_layer16_timestamp.json')
  .then(response => response.json())
  .then(data => {
    // data.metadata
    // data.difference_activations
    // data.layer_correlations
    // ... use for 3D visualization
  });
```

### Load in Python

```python
from activation_steering_lab.threejs_export import ThreeJSExporter

exporter = ThreeJSExporter()
data = exporter.load_threejs_data('activation_steering_lab/mocked_data/happy_layer16.json')

# Access data
print(data['metadata']['concept_name'])  # 'happy'
print(data['peak_activation_layer'])     # 16
print(data['difference_activations']['16'])  # [64 region values]
```

## Files

- `generate_mock_data.py` - Main script to generate dataset
- `test_export.py` - Validation script for JSON format
- `test_sample.json` - Test file with mock activations
- `happy_layer*.json` - Real activation data for "happy" concept
- `sad_layer*.json` - Real activation data for "sad" concept
- `calm_layer*.json` - Real activation data for "calm" concept
- `excited_layer*.json` - Real activation data for "excited" concept

## Performance

- File size: ~0.1-5 MB per file (depending on compression)
- Load time: < 100ms in browser
- No model loading overhead during visualization development
- Enables 60 FPS real-time 3D rendering

## Next Steps

1. ✅ Phase 1: Generate mock data (this directory)
2. Phase 2: Create Three.js scene with these files
3. Phase 3: Map activation heatmaps to 3D geometry
4. Phase 4: Add visual enhancements (lighting, markers)
5. Phase 5: Add interactivity (rotation, selection)
6. Phase 6: Integrate with live model pipeline

See `/docs/THREEJS_MRI_VISUALIZATION.md` for full implementation plan.
