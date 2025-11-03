# Feature Specification: Activation Visualization

**Feature Name**: Live Activation Visualization
**Branch**: `feature/activation-visualization`
**Priority**: HIGH
**Estimated Time**: 4-6 hours
**Assigned To**: Steve (AI Coding Assistant)

---

## 📋 Executive Summary

Add real-time visualization of transformer layer activations to the Gradio interface. This will allow students to **see** activation steering happening inside the model, like an "MRI scan for AI thoughts."

**Key Deliverables**:
1. Token×Layer Activation Heatmap (shows what changed)
2. 2D Concept Space Explorer (shows where concepts live)
3. New "Activation Visualizer" tab in Gradio UI

**Educational Impact**: This will be the **first** activation steering tool with live visualization, making the abstract concept concrete and visual.

---

## 🎯 Goals

### Primary Goals
1. Visualize how activations change across layers during steering
2. Show concept relationships in 2D geometric space
3. Provide interactive, educational visualizations

### Success Criteria
- ✅ Heatmap shows clear difference before/after steering
- ✅ Concept space shows movement toward target concept
- ✅ Renders in <3 seconds on M4 Mac
- ✅ Memory overhead <20MB per visualization
- ✅ Educational explanations included

---

## 🏗️ Technical Architecture

### New File Structure

```
activation_steering_lab/
├── visualization.py          # NEW - All visualization logic
├── main.py                   # MODIFIED - Add new tab
├── model_wrapper.py          # MODIFIED - Add capture_all_layers()
├── injection_engine.py       # No changes needed
└── educational_content.py    # MODIFIED - Add viz explanations
```

---

## 📝 Implementation Specifications

### Phase 1: Create visualization.py Module

**File**: `activation_steering_lab/visualization.py`

**Class Structure**:
```python
class ActivationVisualizer:
    """Handles all activation visualization for the steering lab"""

    def __init__(self, model_wrapper, vector_library):
        """
        Args:
            model_wrapper: Instance of ModelWrapper
            vector_library: Instance of VectorLibrary
        """
        self.model = model_wrapper
        self.library = vector_library

    def capture_activations_for_comparison(self, prompt, concept_name,
                                          layer_idx, strength,
                                          max_new_tokens=50, temperature=0.7):
        """
        Generate text twice and capture all activations for comparison.

        Implementation:
        1. Clear any existing hooks
        2. Register capture hooks for ALL layers (0 to num_layers-1)
        3. Generate normal output (no steering)
        4. Store captured activations in normal_acts dict
        5. Clear hooks
        6. Register capture hooks again for ALL layers
        7. Generate steered output (with concept injection)
        8. Store captured activations in steered_acts dict
        9. Clear hooks
        10. Tokenize the prompt to get token strings

        Returns:
            tuple: (normal_text, steered_text, normal_acts, steered_acts, tokens)
            - normal_text: str
            - steered_text: str
            - normal_acts: dict[layer_idx] -> torch.Tensor [1, seq_len, hidden_dim]
            - steered_acts: dict[layer_idx] -> torch.Tensor [1, seq_len, hidden_dim]
            - tokens: list[str]
        """
        # TODO: Implement this

    def create_token_layer_heatmap(self, normal_acts, steered_acts,
                                   injection_layer, tokens, concept_name):
        """
        Create side-by-side heatmaps comparing normal vs steered activations.

        Implementation:
        1. Convert activation dicts to matrices:
           - For each layer, compute L2 norm across hidden_dim
           - Result: matrix[num_layers, seq_len] for normal and steered
        2. Create figure with 3 subplots (1 row, 3 columns)
        3. Subplot 1: Heatmap of normal activations
           - Use seaborn.heatmap with YlOrRd colormap
           - X-axis: token strings
           - Y-axis: layer indices (L0, L1, L2, ...)
           - Add horizontal line at injection_layer (cyan color)
        4. Subplot 2: Heatmap of steered activations
           - Same as subplot 1
           - Title includes concept_name
           - Add text annotation "← Injection" at injection_layer
        5. Subplot 3: Difference heatmap (steered - normal)
           - Use RdBu_r colormap (red=increase, blue=decrease)
           - Center colormap at 0
           - Symmetric vmin/vmax
           - Add horizontal line at injection_layer (yellow color)
        6. Set figure size to (18, 8)
        7. Use plt.tight_layout()

        Args:
            normal_acts: dict[layer_idx] -> torch.Tensor [1, seq_len, hidden_dim]
            steered_acts: dict[layer_idx] -> torch.Tensor [1, seq_len, hidden_dim]
            injection_layer: int
            tokens: list[str]
            concept_name: str

        Returns:
            matplotlib.figure.Figure
        """
        # TODO: Implement this

    def create_concept_space_2d(self, normal_acts, steered_acts,
                               layer_idx, concept_name):
        """
        Create interactive 2D visualization of concept space using PCA.

        Implementation:
        1. Gather all concept vectors at layer_idx:
           - Iterate through library.list_concepts()
           - For each concept, get vector at layer_idx
           - Store in list: concept_vectors (each is numpy array)
           - Store names in list: concept_names
        2. If fewer than 3 concepts, return placeholder plot with message
        3. Extract last token activation from normal_acts[layer_idx]
           - Shape: [1, seq_len, hidden_dim] -> take [0, -1, :] -> [hidden_dim]
           - Convert to numpy
        4. Extract last token activation from steered_acts[layer_idx]
           - Same as above
        5. Stack all vectors: concept_vectors + [normal_vec] + [steered_vec]
        6. Apply PCA with n_components=2
        7. Split PCA results:
           - concept_coords = first N points (N = num concepts)
           - normal_coord = N+1 point
           - steered_coord = N+2 point
        8. Create Plotly figure with 4 traces:
           a. Concept vectors (red diamonds with text labels)
           b. Target concept highlighted (gold star) if concept_name in list
           c. Normal output (blue circle)
           d. Steered output (green star)
        9. Add arrow annotation from normal to steered (purple)
        10. Set layout with proper labels, title, hover mode
        11. Figure size: 800x600

        Args:
            normal_acts: dict[layer_idx] -> torch.Tensor [1, seq_len, hidden_dim]
            steered_acts: dict[layer_idx] -> torch.Tensor [1, seq_len, hidden_dim]
            layer_idx: int
            concept_name: str

        Returns:
            plotly.graph_objects.Figure
        """
        # TODO: Implement this
```

**Required Imports**:
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch
```

**Dependencies to Add** (if not present):
- `plotly` - Add to requirements.txt
- `seaborn` - Add to requirements.txt
- `scikit-learn` - Add to requirements.txt

---

### Phase 2: Modify model_wrapper.py

**File**: `activation_steering_lab/model_wrapper.py`

**Add Method**:
```python
def capture_all_layers(self):
    """
    Register capture hooks for ALL transformer layers.

    Implementation:
    1. Loop from 0 to self.num_layers - 1
    2. Call self.register_capture_hook(layer_idx) for each

    This allows capturing activations at every layer during generation.
    """
    # TODO: Implement this
```

**Usage in visualization.py**:
```python
# In capture_activations_for_comparison():
self.model.capture_all_layers()  # Instead of registering one layer
```

---

### Phase 3: Modify main.py - Add Integration

**File**: `activation_steering_lab/main.py`

#### Step 1: Import and Initialize Visualizer

**Add to imports** (top of file):
```python
from activation_steering_lab.visualization import ActivationVisualizer
```

**In `ActivationSteeringApp.__init__`**:
```python
def __init__(self):
    # ... existing code ...
    self.visualizer = None  # Will be initialized after model loads
```

**In `ActivationSteeringApp.initialize`** (after model and library are ready):
```python
# After engine initialization
self.visualizer = ActivationVisualizer(self.model, self.library)
```

#### Step 2: Add New Method

**Add to `ActivationSteeringApp` class**:
```python
def generate_steered_with_viz(self, prompt, concept, layer_idx, strength,
                               max_tokens, temperature, progress=gr.Progress()):
    """
    Generate with steering AND create visualizations.

    Implementation:
    1. Check if initialized, return empty figures if not
    2. Call visualizer.capture_activations_for_comparison()
       - This returns: normal_text, steered_text, normal_acts, steered_acts, tokens
    3. Create heatmap: visualizer.create_token_layer_heatmap()
    4. Create concept space: visualizer.create_concept_space_2d()
    5. Generate explanation markdown text (see template below)
    6. Return: normal_text, steered_text, heatmap_fig, concept_space_fig, explanation

    Args:
        prompt: str
        concept: str (concept name)
        layer_idx: int or str (convert to int)
        strength: float or str (convert to float)
        max_tokens: int or str (convert to int)
        temperature: float or str (convert to float)
        progress: gr.Progress

    Returns:
        tuple: (normal_output, steered_output, heatmap_fig, concept_space_fig, explanation_md)

    Error Handling:
    - If not initialized: return error message for text, empty matplotlib figure, empty plotly figure
    - If exception during generation: return error message and empty figures
    """
    # TODO: Implement this
```

**Explanation Template**:
```markdown
**Visualization Guide:**

**🧠 Activation Heatmap**
- Shows activation magnitudes across all layers and tokens
- Brighter colors = higher activation
- Cyan/Yellow line = injection layer (Layer {layer_idx})
- Right panel = difference (red=increase, blue=decrease)

**📐 Concept Space**
- 2D projection showing where concepts "live" in activation space
- Red diamonds = available concepts in the library
- Gold star = target concept "{concept}"
- Blue circle = normal output position
- Green star = steered output position
- Purple arrow = steering direction and magnitude

**Injection Details:**
- Concept: {concept}
- Layer: {layer_idx} (middle layers typically most effective)
- Strength: {strength}
- Effect: Activations moved toward "{concept}" in conceptual space

**What You're Seeing:**
The heatmap proves that injecting the concept vector at Layer {layer_idx} changes
the activation patterns in all downstream layers. The concept space shows this as
geometric movement toward the "{concept}" cluster.
```

#### Step 3: Add New Gradio Tab

**Add AFTER existing tabs** (before `demo.launch()`):

```python
with gr.Tab("🔬 Activation Visualizer"):
    gr.Markdown("""
    ## See Inside the Model's "Mind"

    Visualize how steering changes the model's internal activations in real-time.
    Like an MRI scan for AI thoughts!

    **What you'll see:**
    - 🧠 **Activation Heatmap**: Layer-by-layer activation patterns (before/after/difference)
    - 📐 **Concept Space**: 2D map showing where concepts live and how steering moves activations
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            viz_prompt = gr.Textbox(
                value="The meeting went",
                label="Prompt",
                lines=2,
                placeholder="Enter your prompt..."
            )
            viz_concept = gr.Dropdown(
                choices=[],  # Will be populated after initialization
                label="Concept",
                value=None
            )
            viz_layer = gr.Slider(
                minimum=0,
                maximum=31,  # Will be updated based on model
                value=16,
                step=1,
                label="Injection Layer"
            )
            viz_strength = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=2.0,
                step=0.5,
                label="Steering Strength"
            )
            viz_max_tokens = gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
                step=10,
                label="Max Tokens"
            )
            viz_temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            visualize_btn = gr.Button("🔍 Visualize Activations", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Text Outputs")
            viz_normal_out = gr.Textbox(label="Normal Output", lines=4)
            viz_steered_out = gr.Textbox(label="Steered Output", lines=4)

    gr.Markdown("### 🧠 Activation Heatmap")
    gr.Markdown("*Shows activation magnitudes across all layers and tokens. Bright colors = high activation.*")
    with gr.Row():
        activation_heatmap = gr.Plot(label="Token × Layer Heatmap")

    gr.Markdown("### 📐 2D Concept Space")
    gr.Markdown("*Interactive visualization showing where concepts live in the model's internal space.*")
    with gr.Row():
        concept_space = gr.Plot(label="Concept Space Explorer")

    with gr.Accordion("📖 Understanding the Visualizations", open=False):
        viz_explanation = gr.Markdown("")

    # Event handlers
    visualize_btn.click(
        fn=app.generate_steered_with_viz,
        inputs=[viz_prompt, viz_concept, viz_layer, viz_strength,
                viz_max_tokens, viz_temperature],
        outputs=[viz_normal_out, viz_steered_out, activation_heatmap,
                concept_space, viz_explanation]
    )

    # Update concept dropdown after initialization
    init_btn.click(
        fn=lambda: gr.update(choices=app.get_concept_list()),
        inputs=None,
        outputs=viz_concept
    )
```

---

## 🧪 Testing Requirements

### Unit Tests

**Create**: `tests/test_visualization.py`

```python
"""Test activation visualization functionality"""

def test_visualizer_initialization():
    """Test ActivationVisualizer can be initialized"""
    # Load model and library
    # Create visualizer
    # Assert visualizer.model and visualizer.library are set

def test_capture_activations():
    """Test activation capture for normal and steered generation"""
    # Generate with simple prompt
    # Assert normal_acts is dict with all layer keys
    # Assert steered_acts is dict with all layer keys
    # Assert activations have correct shape [1, seq_len, hidden_dim]

def test_heatmap_creation():
    """Test heatmap generation"""
    # Create mock activations
    # Call create_token_layer_heatmap()
    # Assert returns matplotlib.figure.Figure
    # Assert figure has 3 subplots

def test_concept_space_creation():
    """Test 2D concept space visualization"""
    # Create mock activations
    # Call create_concept_space_2d()
    # Assert returns plotly.graph_objects.Figure
    # Assert figure has expected traces

def test_with_insufficient_concepts():
    """Test concept space with <3 concepts"""
    # Mock library with only 2 concepts
    # Call create_concept_space_2d()
    # Assert returns placeholder figure with error message
```

### Integration Tests

**Manual Testing Checklist**:
1. ✅ Launch app: `./run.sh`
2. ✅ Initialize model
3. ✅ Navigate to "🔬 Activation Visualizer" tab
4. ✅ Select prompt: "The meeting went"
5. ✅ Select concept: "happy"
6. ✅ Set layer: 16, strength: 2.0
7. ✅ Click "🔍 Visualize Activations"
8. ✅ Verify:
   - Normal text appears
   - Steered text appears (different from normal)
   - Heatmap shows 3 panels (normal, steered, difference)
   - Injection layer marked with line
   - Concept space shows scatter plot with arrow
   - Explanation text appears
9. ✅ Test with different concepts (sad, formal, technical)
10. ✅ Test with different layers (8, 16, 24)
11. ✅ Test with different strengths (0.5, 2.0, 5.0)

### Performance Tests

**Requirements**:
- Visualization generation: <3 seconds
- Memory overhead: <20MB
- No memory leaks (test 10 consecutive generations)

**Test Script** (`tests/test_viz_performance.py`):
```python
import time
import tracemalloc

def test_visualization_performance():
    # Start memory tracking
    tracemalloc.start()

    start = time.time()
    # Run visualization
    elapsed = time.time() - start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time: {elapsed:.2f}s")
    print(f"Memory: {peak / 1024 / 1024:.2f} MB")

    assert elapsed < 3.0, "Too slow!"
    assert peak / 1024 / 1024 < 20, "Too much memory!"
```

---

## 📊 Success Metrics

### Functional Requirements
- ✅ Heatmap shows clear visual difference between normal/steered
- ✅ Concept space shows movement toward target concept
- ✅ All visualizations render without errors
- ✅ Interactive features work (zoom, pan, hover in concept space)

### Performance Requirements
- ✅ Total generation + visualization: <5 seconds
- ✅ Memory overhead: <20MB per visualization
- ✅ No crashes with repeated use

### Educational Requirements
- ✅ Explanations are clear and accurate
- ✅ Visualizations are self-explanatory with legends
- ✅ Students can understand what they're seeing

---

## 🚨 Edge Cases to Handle

1. **Model not initialized**
   - Return error message: "Initialize model first"
   - Return empty matplotlib and plotly figures

2. **Concept not found**
   - Graceful error in explanation
   - Show available concepts

3. **Fewer than 3 concepts in library**
   - Concept space shows placeholder with message
   - Heatmap still works

4. **Very long prompts (>100 tokens)**
   - Heatmap x-axis may be crowded
   - Consider truncating token labels or rotating

5. **Invalid layer index**
   - Validate layer is in range [0, num_layers-1]
   - Show error message

6. **Memory constraints**
   - If capturing all layers fails, fall back to key layers only
   - Show warning to user

---

## 📚 Dependencies

### New Python Packages

Add to `requirements.txt`:
```
plotly>=5.18.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

**Verify existing packages** (should already be installed):
- matplotlib
- numpy
- torch

### Import Checklist

**visualization.py**:
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch
```

**main.py additions**:
```python
from activation_steering_lab.visualization import ActivationVisualizer
```

---

## 🎨 Visual Design Specifications

### Heatmap Design
- **Colormap**: YlOrRd (Yellow-Orange-Red, shows intensity)
- **Difference colormap**: RdBu_r (Red-Blue reversed, centered at 0)
- **Injection marker**: Cyan horizontal line, 2-3px wide
- **Figure size**: 18 inches wide × 8 inches tall
- **Font size**: 12pt for titles, 10pt for labels
- **Token labels**: Rotated 45° if >10 tokens

### Concept Space Design
- **Concept markers**: Red diamonds, size 12
- **Target concept**: Gold star, size 20
- **Normal output**: Blue circle, size 15
- **Steered output**: Green star, size 15
- **Arrow**: Purple, width 3, arrowhead size 3
- **Background**: Light gray (240, 240, 240, 0.5)
- **Figure size**: 800px × 600px
- **Interactive**: Enable zoom, pan, hover tooltips

---

## 🔍 Code Review Checklist

Before marking complete, verify:

**Code Quality**:
- [ ] All functions have docstrings
- [ ] Type hints for function arguments
- [ ] Error handling for edge cases
- [ ] No hardcoded magic numbers
- [ ] Consistent naming conventions

**Functionality**:
- [ ] Heatmap shows 3 panels correctly
- [ ] Concept space shows all concepts + outputs
- [ ] Arrow points from normal to steered
- [ ] Injection layer is highlighted
- [ ] Explanations are accurate

**Performance**:
- [ ] No memory leaks (tested with 10 runs)
- [ ] Generation + viz completes in <5 seconds
- [ ] Cleanup of activations after each run

**Integration**:
- [ ] New tab appears in UI
- [ ] Button click triggers visualization
- [ ] Concept dropdown populates after init
- [ ] Error messages display properly

**Documentation**:
- [ ] CLAUDE.md updated with new feature
- [ ] README.md mentions visualization feature
- [ ] Inline comments for complex logic

---

## 📖 Reference Materials

### Research Findings
See the comprehensive research report that identified:
- Gradio supports matplotlib, plotly, seaborn via `gr.Plot()`
- PCA is fast enough for real-time (10ms for 20 vectors)
- Token×Layer heatmaps are standard in transformer interpretability
- 2D concept space uses proven dimensionality reduction techniques

### Existing Code to Reference
- `model_wrapper.py:register_capture_hook()` - How to capture activations
- `injection_engine.py:generate_with_steering()` - How to generate with steering
- `vector_library.py:get_vector()` - How to retrieve concept vectors
- `main.py` existing tabs - UI patterns to follow

### External Examples
- Anthropic Transformer Circuits: transformer-circuits.pub
- Distill.pub Activation Atlases: distill.pub/2019/activation-atlas
- Plotly scatter plot docs: plotly.com/python/line-and-scatter

---

## 🎯 Acceptance Criteria

Feature is **COMPLETE** when:

1. ✅ `visualization.py` exists with all 3 methods implemented
2. ✅ `model_wrapper.py` has `capture_all_layers()` method
3. ✅ `main.py` has new "Activation Visualizer" tab
4. ✅ `requirements.txt` includes plotly, seaborn, scikit-learn
5. ✅ Manual testing checklist 100% passed
6. ✅ Performance tests pass (<3s, <20MB)
7. ✅ Unit tests pass (all 4 test functions)
8. ✅ No console errors or warnings
9. ✅ README.md updated to mention visualization feature
10. ✅ Code committed to `feature/activation-visualization` branch

---

## 🚀 Deployment Plan

1. Steve completes implementation on feature branch
2. I review code and test manually
3. Run all tests (unit + integration + performance)
4. Create PR from `feature/activation-visualization` → `main`
5. Review PR, address any issues
6. Merge to main
7. Update README.md with new screenshots
8. Push to GitHub
9. Announce feature in README as "NEW! 🔬 Live Activation Visualization"

---

**End of Specification**

Steve: You have everything you need. Follow this spec step-by-step, test thoroughly, and create something amazing! 🚀
