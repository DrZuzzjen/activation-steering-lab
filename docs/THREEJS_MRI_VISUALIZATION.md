# Three.js MRI-Style Brain Visualization for Activation Steering

## Executive Summary

This document outlines the technical approach for creating a 3D fMRI-style "brain scan" visualization of transformer layer activations using Three.js. The goal is to achieve the aesthetic and depth of medical brain imaging (similar to the reference image provided by the user) while showing how concept injection propagates through transformer layers.

**Key Constraint**: Use saved/mocked activation data instead of running model inference repeatedly during visualization development. This enables rapid iteration without the overhead of loading Phi-3 (14GB) and running inference each time.

---

## 1. Mocked Activation Data Format

### 1.1 Current Data Structure

From [model_wrapper.py:51-52](activation_steering_lab/model_wrapper.py#L51-L52), activations are stored as:

```python
self.captured_activations: Dict[int, torch.Tensor] = {}
# Structure: {layer_idx: tensor[batch, seq_len, hidden_size]}
```

**For Phi-3-mini-4k-instruct:**
- 32 layers (L0-L31)
- Hidden size: 3072 dimensions per layer
- Batch size: 1 (single generation)
- Sequence length: varies (prompt_tokens + max_new_tokens)

### 1.2 Saved Data Format for Three.js Development

Create a JSON file that can be loaded instantly without model inference:

**File: `activation_steering_lab/mocked_data/sample_activations.json`**

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
    "timestamp": "2025-11-03T15:00:00Z"
  },
  "normal_activations": {
    "0": [[/* 15 tokens × 3072 dims, downsampled to 64 regions */]],
    "1": [[/* ... */]],
    "...": "...",
    "31": [[/* ... */]]
  },
  "steered_activations": {
    "0": [[/* ... */]],
    "...": "..."
  },
  "tokens": ["Tell", "me", "about", "happiness", "▁I", "▁would", "..."],
  "downsampled_regions": 64,
  "layer_correlations": {
    "0": 0.12,
    "1": 0.15,
    "...": "...",
    "16": 1.0,
    "17": 0.87,
    "18": 0.76,
    "...": "..."
  }
}
```

### 1.3 Downsampling Strategy

To keep file size manageable and enable real-time 3D rendering:

1. **Spatial downsampling**: 3072 dimensions → 64 "neural regions" (average chunks of ~48 dims)
2. **Focus on last token**: Use activations from token position -1 (where steering has max impact)
3. **Difference vector**: Store `steered - normal` to show steering effect

**Python helper function (for Steve to implement):**

```python
def save_activations_for_threejs(
    normal_acts: Dict[int, torch.Tensor],
    steered_acts: Dict[int, torch.Tensor],
    tokens: List[str],
    metadata: dict,
    output_path: str,
    num_regions: int = 64
) -> None:
    """
    Save activation data in Three.js-friendly format.

    Args:
        normal_acts: Normal generation activations {layer_idx: tensor[1, seq_len, 3072]}
        steered_acts: Steered generation activations
        tokens: Token strings
        metadata: Dict with prompt, concept_name, injection_layer, etc.
        output_path: Where to save JSON file
        num_regions: Number of spatial regions to downsample to (default 64)
    """
    # Implementation details:
    # 1. Extract last token activations for each layer
    # 2. Downsample 3072 dims → 64 regions using averaging
    # 3. Compute difference: steered - normal
    # 4. Calculate layer correlations with injection layer
    # 5. Save as JSON with proper structure
    pass
```

---

## 2. Three.js Visualization Architecture

### 2.1 Visual Design Goals (from fMRI reference image)

**Key characteristics of medical brain imaging to replicate:**

1. **3D anatomical structure**: Brain-like 3D mesh with depth, shadows, lighting
2. **Colored activation overlays**: Heat map colors (dark red → orange → yellow → white) projected onto 3D surface
3. **Spatial organization**: Layers organized in 3D space (not flat 2D grid)
4. **Depth perception**: Camera perspective, lighting, shadows create depth
5. **Interactive exploration**: Rotate, zoom, select individual layers

### 2.2 Three.js Scene Structure

**Core Components:**

```
Scene
├── Camera (PerspectiveCamera with OrbitControls)
├── Lights
│   ├── AmbientLight (soft base lighting)
│   ├── DirectionalLight (creates shadows, depth)
│   └── SpotLight (highlights active regions)
├── Brain Structure (32 layers as 3D objects)
│   ├── Layer 0 (BoxGeometry or custom mesh)
│   ├── Layer 1
│   ├── ...
│   └── Layer 31
└── Annotations
    ├── Injection marker (cyan lightning bolt at L16)
    ├── Correlation arrows (between highly correlated layers)
    └── Metrics panel (2D overlay)
```

### 2.3 Layer Representation Options

**Option A: Stacked Slices (Recommended for MVP)**
- Each layer = flat plane (BoxGeometry with thin height)
- Stacked vertically with spacing
- Texture mapped with activation heatmap
- Pros: Simple to implement, clear layer separation
- Cons: Less organic "brain-like" feel

**Option B: 3D Volumetric Mesh**
- Single 3D mesh representing entire "brain"
- Vertex colors/shaders show activations
- Pros: More organic, looks like real fMRI
- Cons: Complex shader programming required

**Recommendation**: Start with Option A for rapid prototyping, iterate to Option B if needed.

### 2.4 Activation Heatmap Mapping

**Color Scale (matching current Plotly implementation):**

```javascript
const colorScale = [
  { value: 0.0, color: new THREE.Color(0.235, 0.235, 0.235) },  // rgb(60,60,60) - dark gray
  { value: 0.3, color: new THREE.Color(0.392, 0.0, 0.0) },      // rgb(100,0,0) - dark red
  { value: 0.5, color: new THREE.Color(0.784, 0.0, 0.0) },      // rgb(200,0,0) - red
  { value: 0.7, color: new THREE.Color(1.0, 0.392, 0.0) },      // rgb(255,100,0) - orange
  { value: 0.85, color: new THREE.Color(1.0, 0.784, 0.0) },     // rgb(255,200,0) - yellow
  { value: 1.0, color: new THREE.Color(1.0, 1.0, 1.0) }         // rgb(255,255,255) - white
];
```

**Mapping strategy:**
1. Load downsampled regions (64 values per layer)
2. Normalize to [0, 1] using percentile scaling (5th to 95th percentile)
3. Create texture/vertex colors using color scale
4. Apply to layer geometry

---

## 3. Gradio Integration Strategy

### 3.1 Custom Gradio Component Approach

**Why custom component?**
- `gr.HTML` cannot execute JavaScript by default
- Need full control over Three.js scene lifecycle
- Want interactive controls (rotate, zoom, layer selection)

**Implementation path:**

1. Create custom Gradio component using Svelte
2. Embed Three.js canvas in component
3. Pass activation JSON data as component input
4. Render 3D scene client-side (no Python dependency after data load)

### 3.2 File Structure

```
activation_steering_lab/
├── gradio_components/
│   └── threejs_brain_viewer/
│       ├── __init__.py
│       ├── frontend/
│       │   ├── Index.svelte          # Main component
│       │   ├── BrainScene.js         # Three.js scene setup
│       │   ├── ActivationRenderer.js # Heatmap → 3D mapping
│       │   ├── Controls.js           # Camera, interaction
│       │   └── package.json
│       └── backend/
│           └── component.py          # Python wrapper
└── mocked_data/
    ├── sample_activations.json       # For development
    └── generate_mock_data.py         # Script to create samples
```

### 3.3 Gradio Component Creation Steps

**Step 1: Initialize custom component**

```bash
cd activation_steering_lab
gradio cc create threejs_brain_viewer
```

**Step 2: Add Three.js dependencies**

In `frontend/package.json`:
```json
{
  "dependencies": {
    "three": "^0.160.0",
    "three/examples/jsm/controls/OrbitControls": "latest"
  }
}
```

**Step 3: Implement Svelte component**

In `frontend/Index.svelte`:
```svelte
<script>
  import { onMount } from 'svelte';
  import { createBrainScene } from './BrainScene.js';

  export let value; // Activation JSON data from Python
  export let label = "3D Brain Activation Viewer";

  let canvas;
  let scene;

  onMount(() => {
    scene = createBrainScene(canvas, value);
    return () => scene.dispose(); // Cleanup
  });

  $: if (scene && value) {
    scene.updateActivations(value);
  }
</script>

<div class="threejs-container">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .threejs-container {
    width: 100%;
    height: 800px;
    background: rgb(10, 10, 10); /* Dark background like MRI */
  }
</style>
```

**Step 4: Implement Three.js scene**

In `frontend/BrainScene.js`:
```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export function createBrainScene(canvas, activationData) {
  // Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a); // Very dark like MRI

  // Camera with perspective
  const camera = new THREE.PerspectiveCamera(
    60, // FOV
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    1000
  );
  camera.position.set(50, 30, 50);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.shadowMap.enabled = true;

  // Orbit controls for rotation/zoom
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;

  // Lighting (critical for depth perception)
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
  directionalLight.position.set(10, 20, 10);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  // Create layer meshes
  const layers = createLayerMeshes(activationData);
  layers.forEach(layer => scene.add(layer));

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return {
    updateActivations: (newData) => updateLayerColors(layers, newData),
    dispose: () => {
      renderer.dispose();
      controls.dispose();
    }
  };
}

function createLayerMeshes(activationData) {
  const layers = [];
  const numLayers = activationData.metadata.num_layers;
  const numRegions = activationData.downsampled_regions;

  for (let i = 0; i < numLayers; i++) {
    // Create thin box for each layer
    const geometry = new THREE.BoxGeometry(numRegions * 0.8, 1, numRegions * 0.8);
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide
    });

    // Color vertices based on activation values
    const colors = computeVertexColors(
      activationData.steered_activations[i],
      activationData.normal_activations[i]
    );
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.y = i * 2; // Stack layers vertically
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    layers.push(mesh);
  }

  return layers;
}

function computeVertexColors(steeredActs, normalActs) {
  // Compute difference and map to color scale
  // Returns Float32Array of RGB values for each vertex
  // ... implementation details ...
}

function updateLayerColors(layers, newData) {
  // Update vertex colors when new activation data loads
  // ... implementation details ...
}
```

### 3.4 Python Backend Integration

In `backend/component.py`:

```python
from gradio.components import Component
import json
from pathlib import Path

class ThreeJSBrainViewer(Component):
    """
    Custom Gradio component for 3D brain activation visualization.
    """

    def __init__(
        self,
        value: str | dict | None = None,
        label: str = "3D Brain Activation Viewer",
        **kwargs
    ):
        """
        Args:
            value: Either path to activation JSON file or dict of activation data
        """
        super().__init__(value=value, label=label, **kwargs)

    def preprocess(self, x):
        """Convert input to format expected by frontend."""
        if isinstance(x, str):
            # Load from file path
            with open(x, 'r') as f:
                return json.load(f)
        elif isinstance(x, dict):
            return x
        return None

    def postprocess(self, x):
        """Format data for frontend."""
        return x
```

### 3.5 Usage in Main Gradio App

In [main.py](activation_steering_lab/main.py):

```python
from activation_steering_lab.gradio_components.threejs_brain_viewer import ThreeJSBrainViewer

# In Gradio interface:
with gr.Tab("🧠 3D Brain Scan"):
    gr.Markdown("""
    ## fMRI-Style 3D Visualization
    Explore how activation steering propagates through transformer layers in 3D space.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            load_sample_btn = gr.Button("Load Sample Data")
            concept_dropdown = gr.Dropdown(
                choices=["happy", "sad", "calm", "excited"],
                value="happy",
                label="Concept"
            )
            layer_slider = gr.Slider(0, 31, value=16, step=1, label="Injection Layer")

        with gr.Column(scale=3):
            brain_viewer = ThreeJSBrainViewer(
                value="activation_steering_lab/mocked_data/sample_activations.json"
            )

    load_sample_btn.click(
        fn=load_sample_activation_data,
        inputs=[concept_dropdown, layer_slider],
        outputs=brain_viewer
    )
```

---

## 4. Implementation Roadmap for Steve

### Phase 1: Mocked Data Generation (Estimated: 2-3 hours)

**Tasks:**
1. Create `activation_steering_lab/mocked_data/` folder
2. Implement `save_activations_for_threejs()` function in new module `activation_steering_lab/threejs_export.py`
3. Run real model once to capture activations for "happy" concept at L16
4. Save as `sample_activations.json` with proper downsampling
5. Create additional samples for "sad", "calm", "excited" concepts
6. Validate JSON structure loads correctly

**Success criteria:**
- JSON file < 5MB (enables fast loading)
- Contains all required fields (metadata, activations, correlations)
- Can be loaded and parsed in Python and JavaScript

### Phase 2: Basic Three.js Scene (Estimated: 4-6 hours)

**Tasks:**
1. Initialize custom Gradio component: `gradio cc create threejs_brain_viewer`
2. Set up Three.js dependencies in `package.json`
3. Implement basic `BrainScene.js`:
   - Scene, camera, renderer setup
   - Orbit controls for rotation/zoom
   - Basic lighting (ambient + directional)
4. Create 32 layer meshes as stacked boxes
5. Test rendering with dummy colors (no activation data yet)
6. Integrate into main Gradio app as new tab

**Success criteria:**
- 3D scene renders in browser
- Can rotate and zoom with mouse
- 32 layers visible and stacked correctly
- No console errors

### Phase 3: Activation Heatmap Mapping (Estimated: 4-5 hours)

**Tasks:**
1. Implement `computeVertexColors()` in `BrainScene.js`:
   - Load downsampled activation regions from JSON
   - Compute difference: steered - normal
   - Normalize using percentile scaling
   - Map to color scale (dark gray → red → orange → yellow → white)
2. Apply vertex colors to layer geometries
3. Add texture mapping as alternative to vertex colors (if needed)
4. Test with real sample_activations.json
5. Validate colors match Plotly heatmap intensity

**Success criteria:**
- Layers show activation heatmaps correctly
- Color intensity reflects activation magnitude
- Injection layer (L16) shows strongest activation
- Surrounding layers show propagation effect

### Phase 4: Visual Enhancements (Estimated: 3-4 hours)

**Tasks:**
1. Add injection layer marker (cyan lightning bolt or vertical line)
2. Implement correlation arrows between highly correlated layers
3. Add 2D overlay panel with metrics:
   - Peak activation layer
   - Total intensity
   - Top 3 correlated layers
4. Improve lighting for better depth perception:
   - Add spotlight on injection layer
   - Enhance shadows
5. Add background effects (subtle grid, dark atmosphere)

**Success criteria:**
- Injection layer clearly marked
- Correlation arrows show propagation visually
- Metrics panel readable and informative
- Overall aesthetic similar to fMRI reference image

### Phase 5: Interactivity & Polish (Estimated: 3-4 hours)

**Tasks:**
1. Add layer selection (click on layer to highlight and show details)
2. Implement layer hover tooltips showing activation values
3. Add camera presets (front view, side view, top view buttons)
4. Optimize rendering performance:
   - Use InstancedMesh for repeated geometries
   - Implement frustum culling
   - Add loading spinner during data load
5. Add animation: smooth transition when switching between concepts
6. Test on different browsers (Chrome, Firefox, Safari)

**Success criteria:**
- Interactive layer selection works
- Hover tooltips provide useful info
- Camera controls intuitive
- Runs at 60 FPS on typical hardware
- No memory leaks during repeated use

### Phase 6: Integration & Testing (Estimated: 2-3 hours)

**Tasks:**
1. Connect to real visualization pipeline:
   - Capture activations during real steering
   - Export to JSON format
   - Load into 3D viewer
2. Add button to "Export Current Activations to 3D View"
3. Test full workflow: prompt → steer → visualize in 3D
4. Write tests for Three.js component (if feasible)
5. Update documentation with screenshots/GIFs

**Success criteria:**
- Can visualize activations from real model runs
- Export process takes < 5 seconds
- 3D view updates correctly with new data
- Documentation clear and comprehensive

---

## 5. Technical Considerations & Challenges

### 5.1 Performance

**Potential bottlenecks:**
- 32 layers × 64 regions = 2048 mesh faces (manageable)
- Vertex color computation on every update
- JSON parsing for large activation files

**Optimizations:**
- Use InstancedMesh if all layers share same base geometry
- Precompute vertex colors in Python, store in JSON
- Use WebWorkers for heavy computation (if needed)
- Implement LOD (Level of Detail) for distant layers

### 5.2 Browser Compatibility

**Considerations:**
- WebGL 2.0 required (supported by all modern browsers 2019+)
- Gradio runs in iframe, ensure Three.js works in iframe context
- Test on Safari (WebGL support sometimes quirky)
- Mobile support (lower poly count, simplified shaders)

### 5.3 Data Transfer Size

**Current approach:**
- JSON file with 32 layers × 64 regions × 2 (normal + steered)
- Estimated size: ~2-5 MB depending on precision

**If too large:**
- Use binary format (ArrayBuffer) instead of JSON
- Compress with gzip (Gradio supports)
- Stream data progressively (load layers incrementally)

### 5.4 Debugging Three.js in Gradio

**Tips for Steve:**
- Use `console.log()` extensively in JavaScript
- Install Three.js editor as standalone for isolated testing
- Use browser DevTools → Network tab to verify JSON loads
- Add debug mode flag to show wireframes, axes helpers

---

## 6. Alternative Approaches (If Three.js Proves Difficult)

### Option B: VTK.js (Medical Imaging Library)

**Pros:**
- Designed specifically for medical/scientific visualization
- Built-in volume rendering support
- Better suited for fMRI-like visualizations

**Cons:**
- Steeper learning curve
- Less documentation than Three.js
- Heavier library size

**When to consider:** If Three.js cannot achieve adequate fMRI aesthetic after Phase 4

### Option C: Babylon.js

**Pros:**
- Similar to Three.js but with better built-in effects
- Excellent lighting/shadow system
- Good performance

**Cons:**
- Larger library size
- Similar complexity to Three.js

**When to consider:** If Three.js performance issues arise

### Option D: WebGL Shaders (Custom)

**Pros:**
- Maximum control over visuals
- Best performance
- Can achieve exact fMRI look

**Cons:**
- Requires GLSL shader programming expertise
- Time-intensive development
- Hard to debug

**When to consider:** Only if all library-based approaches fail

---

## 7. Success Metrics

**How to know if this visualization is successful:**

1. **Visual Similarity to fMRI Reference**: Side-by-side comparison shows similar aesthetic (depth, colors, 3D structure)
2. **Scientific Clarity**: Injection layer and propagation clearly visible in 3D space
3. **Performance**: Runs at 60 FPS, loads in < 3 seconds
4. **User Engagement**: More intuitive than current 2D heatmap (get user feedback)
5. **Educational Value**: Helps users understand how steering propagates through layers

**User acceptance criteria (from original request):**
> "represent the MRI effect I'm looking for" - 3D depth, anatomical structure, colored overlays
> "faster and easier to see" - Uses mocked data, no repeated model loading
> "mind blowing way" - Visually impressive, unlike flat Plotly charts

---

## 8. Resources & References

### Three.js Documentation
- Official docs: https://threejs.org/docs/
- Examples: https://threejs.org/examples/
- Brain visualization example: https://discourse.threejs.org/t/3d-brain-reconstruction-with-high-resolution-mri/17966

### Medical Visualization Projects
- BrainBrowser: https://brainbrowser.cbrain.mcgill.ca/
- three-brain-js: https://github.com/dipterix/three-brain-js
- WebGL Volume Rendering: https://observablehq.com/@mroehlig/3d-volume-rendering-with-webgl-three-js

### Gradio Custom Components
- Official guide: https://www.gradio.app/guides/custom-components-in-five-minutes
- Custom HTML/JS: https://www.gradio.app/guides/custom-CSS-and-JS

### Color Science
- Perceptually uniform colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- Medical imaging colors: Typically use "hot" colormap (black → red → yellow → white)

---

## 9. Questions for Clarification (Before Steve Starts)

1. **Mocked data generation**: Should Steve run the model once to generate real samples, or should I provide pre-captured activation data?

2. **Scope of Phase 1**: Should we aim for basic 3D scene first (stacked boxes) or go directly for volumetric rendering?

3. **Browser support priority**: Primary target is desktop Chrome? Do we need mobile support?

4. **Performance vs. visual quality**: If performance is an issue, prefer simplify visuals or accept lower FPS?

5. **Integration timeline**: Is this for next release or experimental branch?

---

## 10. Next Steps (Immediate Action Items)

**For Steve:**

1. **Read this entire document** to understand the architecture
2. **Review Three.js basics** if unfamiliar (1-2 hour tutorial)
3. **Start with Phase 1**: Create mocked data generation script
4. **Ask questions early** if anything is unclear before Phase 2

**For me (documentation maintainer):**

1. **Stand by for questions** as Steve implements
2. **Review Steve's Phase 1 output** to ensure data format is correct
3. **Test Three.js component** once basic scene is working
4. **Provide visual feedback** comparing to fMRI reference throughout

**Estimated total implementation time: 18-25 hours across 6 phases**

---

## Appendix A: Sample Mocked Data (Abbreviated)

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
    "0": [0.12, 0.15, 0.10, /* ... 64 values */],
    "1": [0.14, 0.18, 0.12, /* ... */],
    "...": "...",
    "31": [0.22, 0.28, 0.25, /* ... */]
  },
  "steered_activations": {
    "0": [0.13, 0.16, 0.11, /* ... */],
    "...": "..."
  },
  "difference_activations": {
    "0": [0.01, 0.01, 0.01, /* steered - normal */],
    "16": [2.50, 2.80, 2.45, /* injection layer shows large diffs */],
    "17": [1.20, 1.35, 1.10, /* propagation to next layer */],
    "...": "..."
  },
  "tokens": ["Tell", "me", "about", "happiness", "▁I", "▁would", "▁say", "..."],
  "layer_correlations": {
    "0": 0.12,
    "16": 1.0,
    "17": 0.87,
    "18": 0.76,
    "19": 0.68,
    "...": "..."
  },
  "peak_activation_layer": 16,
  "total_intensity": 458.3,
  "top_correlated_layers": [16, 17, 18]
}
```

---

**End of Document**

This document provides Steve with a complete technical specification for implementing the Three.js MRI-style visualization. All implementation decisions are clearly documented, alternatives are noted, and the phased approach ensures incremental progress with testable milestones.
