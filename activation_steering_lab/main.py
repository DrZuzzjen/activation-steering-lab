"""
Main Gradio Interface for Activation Steering Lab
"""

import gradio as gr
import plotly.graph_objects as go
import torch
from pathlib import Path
import sys
from typing import Generator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_steering_lab.model_wrapper import ModelWrapper
from activation_steering_lab.vector_library import VectorLibrary
from activation_steering_lab.injection_engine import InjectionEngine
from activation_steering_lab.educational_content import (
    get_explanation, format_layer_info, get_recommended_experiments,
    get_tips_and_tricks, EXAMPLE_PROMPTS, get_concept_pairs
)
from activation_steering_lab.visualization import ActivationVisualizer
from activation_steering_lab.threejs_simple import create_threejs_html
from activation_steering_lab.threejs_export import ThreeJSExporter


class ActivationSteeringApp:
    """Main application class for the Gradio interface."""

    def __init__(self):
        """Initialize the ActivationSteeringApp with default values."""
        self.model = None
        self.library = None
        self.engine = None
        self.visualizer = None
        self.initialized = False

    def initialize(self, progress=gr.Progress()) -> Generator[str, None, str]:
        """Initialize the model and library."""
        if self.initialized:
            return "✓ Already initialized!"
        try:
            progress(0.0, desc="🔧 Initializing model wrapper...")
            yield "🔧 Initializing model wrapper..."

            # Try Phi-3 first (smaller, faster on M4)
            try:
                progress(0.05, desc="📦 Loading Phi-3-mini...")
                yield """📦 **Loading Phi-3-mini-4k-instruct...**

⏳ This step takes 2-3 seconds (loading from cache)
⚠️ Progress bar will appear stuck - this is normal!

The model is being loaded into memory...
"""

                self.model = ModelWrapper(
                    model_name="microsoft/Phi-3-mini-4k-instruct",
                    device="auto"
                )
                self.model.load_model()

            except Exception as e:
                progress(0.1, desc="⚠️ Falling back to Mistral...")
                yield f"⚠️ Phi-3 failed: {e}\n\nFalling back to Mistral-7B (larger but may be slower)..."

                self.model = ModelWrapper(
                    model_name="mistralai/Mistral-7B-Instruct-v0.2",
                    device="auto"
                )
                self.model.load_model()

            progress(0.35, desc="✓ Model loaded! Setting up vector library...")
            yield f"✓ Model loaded: {self.model.model_name}\n✓ Layers: {self.model.num_layers}\n\n📚 Setting up vector library..."

            self.library = VectorLibrary()

            progress(0.45, desc="🔧 Creating injection engine...")
            yield f"✓ Model loaded: {self.model.model_name}\n✓ Library initialized\n\n🔧 Creating injection engine..."

            self.engine = InjectionEngine(self.model, self.library)
            self.visualizer = ActivationVisualizer(self.model, self.library)

            # Try to load saved vectors first
            recommended = self.model.get_recommended_layers()

            progress(0.55, desc="📦 Loading saved concept vectors...")
            yield f"✓ Model loaded: {self.model.model_name}\n✓ Engine ready\n\n📦 Loading saved vectors from disk..."

            # Load any existing vectors first
            try:
                self.library.load_all()
            except:
                pass  # No saved vectors yet

            # Check if we have complete extraction (17 concepts × 3 layers = 51 vectors)
            from activation_steering_lab.educational_content import get_concept_pairs
            all_concept_pairs = []
            for category, pairs in get_concept_pairs().items():
                all_concept_pairs.extend(pairs)

            expected_concepts = len(all_concept_pairs)  # 17
            expected_layers = len(recommended[:3])      # 3
            expected_total = expected_concepts * expected_layers  # 51

            # Count existing vectors
            num_concepts = len(self.library.list_concepts())
            num_vectors = sum(len(layers) for layers in self.library.vectors.values())

            if num_vectors >= expected_total:
                # Complete extraction exists
                progress(0.85, desc=f"✓ Loaded {num_concepts} concepts from cache")
                yield f"✓ Model loaded\n✓ Engine ready\n✓ Loaded {num_concepts} concepts ({num_vectors} vectors) from cache!\n\n⚡ All vectors loaded instantly from disk"
            else:
                # Incomplete or no extraction - need to extract missing ones
                if num_vectors > 0:
                    yield f"✓ Model loaded\n✓ Engine ready\n\n⚠️  Found {num_vectors}/{expected_total} vectors\n🎨 Extracting missing concepts...\nThis will take ~3 minutes"
                else:
                    yield f"✓ Model loaded\n✓ Engine ready\n\n🎨 No saved vectors found, extracting...\nThis will take ~3 minutes (only once)"

                # Extract concepts with progress updates
                progress(0.55, desc="🎨 Extracting concepts...")
                all_concepts = all_concept_pairs  # Already loaded above

                total_extractions = len(all_concepts) * len(recommended[:3])
                current = 0

                for layer_idx in recommended[:3]:
                    for name, concept_prompt, baseline_prompt in all_concepts:
                        try:
                            current += 1
                            progress_pct = 0.55 + (current / total_extractions) * 0.30
                            progress(progress_pct, desc=f"🎨 Extracting '{name}' at layer {layer_idx}...")

                            vector = self.library.compute_concept_vector(
                                model_wrapper=self.model,
                                concept_prompt=concept_prompt,
                                baseline_prompt=baseline_prompt,
                                layer_idx=layer_idx,
                                concept_name=name
                            )
                            self.library.add_vector(vector)

                            yield f"✓ Model ready\n✓ Concepts: {current}/{total_extractions}\n\n🎨 Extracting '{name}' at layer {layer_idx}..."
                        except Exception as e:
                            print(f"    Error extracting {name}: {e}")

            progress(0.90, desc="💾 Saving vectors to disk...")
            yield f"✓ Model loaded: {self.model.model_name}\n✓ Concepts extracted: {len(self.library.list_concepts())}\n\n💾 Saving vectors..."

            self.library.save_all()

            self.initialized = True
            progress(1.0, desc="✅ Done!")

            final_message = f"""
✅ **Initialization Complete!**

✓ Model loaded: {self.model.model_name}
✓ Layers: {self.model.num_layers}
✓ Concepts: {len(self.library.list_concepts())}
✓ Recommended layers: {recommended}

**Ready to steer!** 🚀

Go to the "Steering Playground" tab to try it out!
            """

            yield final_message
            return final_message

        except Exception as e:
            error_msg = f"❌ Initialization failed: {e}\n\nPlease check the console for details."
            yield error_msg
            return error_msg

    def update_layer_info(self, layer_idx):
        """Update layer information display."""
        if not self.initialized or self.model is None:
            return "Initialize model first"

        return format_layer_info(layer_idx, self.model.num_layers)

    def get_concept_list(self):
        """Get list of available concept names.
        
        Returns:
            List[str]: List of all concept names available in the vector library
        """
        if not self.initialized:
            return []
        return self.library.list_concepts()

    def get_available_layers(self, concept_name):
        """Get list of layer indices where this concept is available.
        
        Args:
            concept_name (str): Name of the concept to check
            
        Returns:
            List[int]: Sorted list of layer indices where the concept has vectors available
        """
        if not self.initialized or not concept_name:
            return []
        if concept_name not in self.library.vectors:
            return []
        return sorted(list(self.library.vectors[concept_name].keys()))

    def create_custom_concept(self, name, concept_prompt, baseline_prompt, layer_idx, progress=gr.Progress()) -> Generator[str, None, str]:
        """Create a custom concept vector."""
        if not self.initialized:
            return "Initialize model first"

        try:
            progress(0, desc="Extracting concept...")

            vector = self.library.compute_concept_vector(
                model_wrapper=self.model,
                concept_prompt=concept_prompt,
                baseline_prompt=baseline_prompt,
                layer_idx=int(layer_idx),
                concept_name=name
            )

            self.library.add_vector(vector)
            self.library.save_vector(name, int(layer_idx))

            progress(1.0, desc="Done!")

            stats = self.library.get_vector_stats(name, int(layer_idx))

            return f"""
✓ Created concept: {name}

**Statistics:**
- Norm: {stats['norm']:.3f}
- Mean: {stats['mean']:.3f}
- Std: {stats['std']:.3f}
- Sparsity: {stats['sparsity']:.1%}

Concept saved and ready to use!
            """
        except Exception as e:
            return f"Error: {e}"

    def generate_steered(self, prompt, concept, layer_idx, strength, max_tokens, temperature, progress=gr.Progress()) -> Generator[str, None, str]:
        """Generate with steering and comparison."""
        if not self.initialized:
            return "Initialize model first", "Initialize model first", ""

        try:
            # Validate concept is available at this layer
            available_layers = self.get_available_layers(concept)
            if not available_layers:
                return f"❌ Concept '{concept}' not found", "", f"❌ Concept '{concept}' not loaded. Click 'Initialize Model & Library' first."

            if int(layer_idx) not in available_layers:
                return (
                    f"❌ Invalid layer selection",
                    "",
                    f"❌ **Concept '{concept}' not available at layer {layer_idx}**\n\n"
                    f"✓ Available layers: {available_layers}\n\n"
                    f"💡 **Tip**: Either:\n"
                    f"1. Change layer slider to one of: {available_layers}\n"
                    f"2. Or extract '{concept}' for layer {layer_idx} in the 'Create Concepts' tab"
                )

            progress(0, desc="Generating normal output...")

            # Generate comparison
            results = self.engine.generate_comparison(
                prompt=prompt,
                concept_name=concept,
                layer_idx=int(layer_idx),
                strength=float(strength),
                max_new_tokens=int(max_tokens),
                temperature=float(temperature)
            )

            progress(1.0, desc="Done!")

            # Get explanation
            summary = self.engine.get_injection_summary(concept, int(layer_idx), float(strength))

            explanation = f"""
**Injection Summary:**
- Concept: {concept}
- Layer: {layer_idx} ({summary.get('description', '')})
- Strength: {strength}
- Effective norm: {summary.get('effective_norm', 0):.3f}

**What to look for:**
Compare how the '{concept}' concept changed the generation!
            """

            return results['normal'], results['steered'], explanation

        except Exception as e:
            return f"Error: {e}", f"Error: {e}", f"Error: {e}"

    def generate_steered_with_viz(
        self,
        prompt,
        concept,
        layer_idx,
        strength,
        max_tokens,
        temperature,
        progress=gr.Progress(),
    ):
        """Generate text and build activation visualizations."""
        if not self.initialized or self.visualizer is None:
            message = "Initialize model first"
            return (
                message,
                message,
                self._blank_heatmap(message),
                message,
            )

        try:
            concept = concept or ""
            layer_idx_int = int(layer_idx)
            strength_val = float(strength)
            max_tokens_int = int(max_tokens)
            temperature_val = float(temperature)

            available_layers = self.get_available_layers(concept)
            if not available_layers:
                message = f"❌ Concept '{concept}' not found"
                return (
                    message,
                    "",
                    self._blank_heatmap(message),
                    message,
                )

            if layer_idx_int not in available_layers:
                message = (
                    f"❌ **Concept '{concept}' not available at layer {layer_idx_int}**\n\n"
                    f"✓ Available layers: {available_layers}"
                )
                return (
                    "",
                    "",
                    self._blank_heatmap(message),
                    message,
                )

            progress(0.1, desc="Capturing normal activations...")
            (
                normal_text,
                steered_text,
                normal_acts,
                steered_acts,
                tokens,
            ) = self.visualizer.capture_activations_for_comparison(
                prompt=prompt,
                concept_name=concept,
                layer_idx=layer_idx_int,
                strength=strength_val,
                max_new_tokens=max_tokens_int,
                temperature=temperature_val,
            )

            progress(0.6, desc="Building visualizations...")
            heatmap_fig = self.visualizer.create_token_layer_heatmap(
                normal_acts=normal_acts,
                steered_acts=steered_acts,
                injection_layer=layer_idx_int,
                tokens=tokens,
                concept_name=concept,
            )

            explanation = self._format_visualization_explanation(
                concept=concept,
                layer_idx=layer_idx_int,
                strength=strength_val,
            )

            progress(1.0, desc="Done!")
            return normal_text, steered_text, heatmap_fig, explanation

        except Exception as exc:  # pylint: disable=broad-except
            message = f"Error: {exc}"
            return (
                message,
                message,
                self._blank_heatmap(message),
                message,
            )

    def analyze_layers(self, prompt, concept, strength, progress=gr.Progress()):
        """Analyze effect across multiple layers."""
        if not self.initialized:
            return "Initialize model first"

        try:
            progress(0, desc="Testing across layers...")

            results = self.engine.analyze_layer_effects(
                prompt=prompt,
                concept_name=concept,
                strength=float(strength),
                max_new_tokens=30,
                layer_step=max(1, self.model.num_layers // 8)
            )

            progress(1.0, desc="Done!")

            # Format results
            output = f"# Layer Analysis for '{concept}'\n\n"
            for layer, text in sorted(results.items()):
                position = layer / self.model.num_layers
                if position < 0.25:
                    stage = "Early"
                elif position < 0.5:
                    stage = "Mid-Early"
                elif position < 0.75:
                    stage = "Mid-Late ⭐"
                else:
                    stage = "Late"

                output += f"**Layer {layer}** ({stage}):\n{text}\n\n---\n\n"

            return output

        except Exception as e:
            return f"Error: {e}"

    def test_strengths(self, prompt, concept, layer_idx, progress=gr.Progress()):
        """Test different strength values."""
        if not self.initialized:
            return "Initialize model first"

        try:
            progress(0, desc="Testing strengths...")

            results = self.engine.test_strength_range(
                prompt=prompt,
                concept_name=concept,
                layer_idx=int(layer_idx),
                strengths=[0.5, 1.0, 2.0, 3.0, 5.0],
                max_new_tokens=30
            )

            progress(1.0, desc="Done!")

            # Format results
            output = f"# Strength Analysis for '{concept}' at Layer {layer_idx}\n\n"
            for strength, text in sorted(results.items()):
                output += f"**Strength {strength}:**\n{text}\n\n---\n\n"

            return output

        except Exception as e:
            return f"Error: {e}"

    def mix_emotions(self, emotions_weights, layer_idx, progress=gr.Progress()):
        """Mix multiple emotions."""
        if not self.initialized:
            return "Initialize model first"

        try:
            # Parse weights (format: "happy:0.7,excited:0.3")
            weights_dict = {}
            for item in emotions_weights.split(','):
                parts = item.strip().split(':')
                if len(parts) == 2:
                    emotion, weight = parts
                    weights_dict[emotion.strip()] = float(weight.strip())

            if not weights_dict:
                return "Invalid format. Use: happy:0.7,excited:0.3"

            progress(0, desc="Mixing concepts...")

            mixed = self.engine.create_mixed_emotion(
                emotion_weights=weights_dict,
                layer_idx=int(layer_idx),
                name="mixed_" + "_".join(weights_dict.keys())
            )

            progress(1.0, desc="Done!")

            return f"""
✓ Created mixed concept: {mixed.name}

**Components:**
{chr(10).join(f"- {k}: {v}" for k, v in weights_dict.items())}

**Resulting norm:** {mixed.norm:.3f}

You can now use this in the Steering Playground!
            """

        except Exception as e:
            return f"Error: {e}"

    def _blank_heatmap(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(width=600, height=400, plot_bgcolor="rgba(240,240,240,0.5)")
        return fig

    @staticmethod
    def _format_visualization_explanation(concept: str, layer_idx: int, strength: float) -> str:
        return f"""**Visualization Guide:**

**🧠 Layer Activation Cascade**
- Brain scan-style visualization showing aggregate activation per layer
- Each bar = one transformer layer (L0 → L31)
- Bar height = total activation change magnitude (summed across all hidden dimensions)
- Color intensity = magnitude (red/orange/yellow hotspots like fMRI scans)
- Cyan line (⚡) marks injection layer {layer_idx}
- White circles mark top 3 most affected layers
- Arrows show correlation values (r=X.XX) indicating how strongly each layer responds

**Injection Details:**
- Concept: {concept}
- Layer: {layer_idx} (middle layers typically most effective)
- Strength: {strength}
- Effect: Creates activation cascade that propagates through adjacent layers

**What You're Seeing:**
The cascade visualization proves that injecting the concept vector at Layer {layer_idx} 
creates a wave of activation changes that spread through nearby layers. The correlation 
arrows show which layers are most strongly affected by the injection, just like fMRI 
scans show which brain regions activate together.
"""


def create_interface():
    """Create the Gradio interface."""

    app = ActivationSteeringApp()

    with gr.Blocks(title="Activation Steering Lab", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 Activation Steering Learning Lab

        **Learn how to steer language model behavior by injecting concept vectors into internal layers!**

        This is an educational tool to understand how transformers process information internally.
        """)

        # Initialization section
        with gr.Row():
            with gr.Column():
                init_btn = gr.Button("🚀 Initialize Model & Library", variant="primary", size="lg")
                init_status = gr.Textbox(label="Status", lines=8)

        # Main tabs
        with gr.Tabs():
            # Tab 1: Educational Layer Viewer
            with gr.Tab("📚 Layer Education"):
                gr.Markdown("""
                ## Understanding Transformer Layers

                Learn what happens at each layer of the model and where steering works best.
                """)

                with gr.Row():
                    with gr.Column():
                        layer_slider = gr.Slider(
                            minimum=0,
                            maximum=31,
                            value=16,
                            step=1,
                            label="Select Layer",
                            info="Drag to explore different layers"
                        )
                        layer_info = gr.Markdown("Initialize model first")

                    with gr.Column():
                        gr.Markdown(get_explanation('layer_roles'))

                layer_slider.change(fn=app.update_layer_info, inputs=layer_slider, outputs=layer_info)

                with gr.Accordion("📖 Learn More", open=False):
                    with gr.Tabs():
                        with gr.Tab("Activations vs Embeddings"):
                            gr.Markdown(get_explanation('activations_vs_embeddings'))
                        with gr.Tab("Why Addition?"):
                            gr.Markdown(get_explanation('why_addition'))
                        with gr.Tab("Strength Parameter"):
                            gr.Markdown(get_explanation('strength_parameter'))
                        with gr.Tab("Baseline Importance"):
                            gr.Markdown(get_explanation('baseline_importance'))

            # Tab 2: Concept Vector Creator
            with gr.Tab("🎨 Create Concepts"):
                gr.Markdown("""
                ## Create Custom Concept Vectors

                Extract your own concepts by providing paired prompts.
                """)

                with gr.Row():
                    with gr.Column():
                        concept_name = gr.Textbox(label="Concept Name", placeholder="e.g., 'cheerful'")
                        concept_prompt = gr.Textbox(
                            label="Concept Prompt",
                            placeholder="Text that evokes your concept",
                            lines=3
                        )
                        baseline_prompt = gr.Textbox(
                            label="Baseline Prompt",
                            placeholder="Neutral version of the same text",
                            lines=3
                        )
                        concept_layer = gr.Slider(0, 31, value=16, step=1, label="Layer to Extract From")
                        create_btn = gr.Button("Extract Concept Vector", variant="primary")

                    with gr.Column():
                        concept_result = gr.Markdown("Results will appear here")

                        with gr.Accordion("💡 Tips for Good Concepts", open=False):
                            gr.Markdown(get_explanation('concept_quality'))

                create_btn.click(
                    fn=app.create_custom_concept,
                    inputs=[concept_name, concept_prompt, baseline_prompt, concept_layer],
                    outputs=concept_result
                )

            # Tab 3: Steering Playground
            with gr.Tab("🎮 Steering Playground"):
                gr.Markdown("""
                ## Try Activation Steering!

                Generate text with and without steering to see the difference.
                """)

                with gr.Row():
                    with gr.Column():
                        steer_prompt = gr.Textbox(
                            label="Your Prompt",
                            placeholder="e.g., 'Tell me about the weather'",
                            lines=3
                        )

                        # ONE-CLICK DEMO BUTTON
                        with gr.Row():
                            demo_btn = gr.Button("🎯 Load Demo Example", size="sm", variant="secondary")

                        steer_concept = gr.Dropdown(
                            label="Concept",
                            choices=["Click 'Initialize Model & Library' first"],
                            value="Click 'Initialize Model & Library' first",
                            info="Will populate after initialization",
                            interactive=True
                        )

                        # Info box showing available layers for selected concept
                        concept_layer_info = gr.Markdown("*Select a concept to see available layers*")

                        steer_layer = gr.Slider(0, 31, value=16, step=1, label="Injection Layer")
                        steer_strength = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="Injection Strength",
                                                   info="⚠️ Phi-3 works best at 0.5-1.5. Values >2.0 may cause gibberish!")

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            max_tokens = gr.Slider(10, 100, value=50, step=5, label="Max New Tokens")
                            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")

                        generate_btn = gr.Button("Generate!", variant="primary", size="lg")

                    with gr.Column():
                        gr.Markdown("### Normal Output")
                        normal_output = gr.Textbox(label="Without Steering", lines=6)

                        gr.Markdown("### Steered Output")
                        steered_output = gr.Textbox(label="With Steering", lines=6)

                        explanation = gr.Markdown("Results will appear here")

                # Store reference to update later
                steer_concept_ref = steer_concept

                generate_btn.click(
                    fn=app.generate_steered,
                    inputs=[steer_prompt, steer_concept, steer_layer, steer_strength, max_tokens, temperature],
                    outputs=[normal_output, steered_output, explanation]
                )

            # Tab 4: Activation Visualizer
            with gr.Tab("🔬 Activation Visualizer"):
                gr.Markdown(
                    """
                    ## See Inside the Model's "Mind"

                    Visualize how steering changes the model's internal activations in real-time.
                    Like an fMRI scan for AI thoughts!

                    **What you'll see:**
                    - 🧠 **Layer Activation Cascade**: Brain scan-style bars showing which layers activate most strongly
                    """
                )

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
                            choices=[],
                            label="Concept",
                            value=None
                        )
                        viz_layer = gr.Slider(
                            minimum=0,
                            maximum=31,
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

                gr.Markdown("### 🧠 Layer Activation Cascade")
                gr.Markdown("*Brain scan showing aggregate activation changes per layer. Bar height = total activation magnitude.*")
                with gr.Row():
                    activation_heatmap = gr.Plot(label="Layer Activation Cascade")

                with gr.Accordion("📖 Understanding the Visualization", open=False):
                    viz_explanation = gr.Markdown("")

                visualize_btn.click(
                    fn=app.generate_steered_with_viz,
                    inputs=[
                        viz_prompt,
                        viz_concept,
                        viz_layer,
                        viz_strength,
                        viz_max_tokens,
                        viz_temperature,
                    ],
                    outputs=[
                        viz_normal_out,
                        viz_steered_out,
                        activation_heatmap,
                        viz_explanation,
                    ],
                )

            # Tab 5: 3D Brain Scan  
            with gr.Tab("🧠 3D Brain Scan"):
                gr.Markdown(
                    """
                    ## fMRI-Style 3D Visualization
                    
                    Explore activation steering in 3D space - see how concepts propagate through transformer layers
                    like a medical brain scan!
                    
                    **Controls:**
                    - 🖱️ **Drag** to rotate
                    - 🔄 **Scroll** to zoom
                    - 👁️ **Cyan layer** = Injection point
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")
                        brain_prompt = gr.Textbox(
                            value="Tell me about happiness",
                            label="Prompt",
                            lines=2
                        )
                        brain_concept = gr.Dropdown(
                            choices=[],
                            label="Concept",
                            value=None
                        )
                        brain_layer = gr.Slider(
                            minimum=0,
                            maximum=31,
                            value=16,
                            step=1,
                            label="Injection Layer"
                        )
                        brain_strength = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            value=2.0,
                            step=0.5,
                            label="Strength"
                        )
                        visualize_3d_btn = gr.Button("🧠 Generate 3D Brain Scan", variant="primary", size="lg")
                        
                        gr.Markdown("### Load Sample Data")
                        load_sample_btn = gr.Button("📁 Load Pre-generated Sample", size="sm")
                    
                    with gr.Column(scale=2):
                        brain_viewer = gr.HTML(label="3D Brain Visualization")
                
                def generate_3d_visualization(prompt, concept, layer_idx, strength):
                    """Generate 3D brain visualization."""
                    if not app.initialized:
                        return "<p style='color:red;'>Please initialize model first</p>"
                    
                    try:
                        # Capture activations
                        normal_text, steered_text, normal_acts, steered_acts, tokens = (
                            app.visualizer.capture_activations_for_comparison(
                                prompt=prompt,
                                concept_name=concept,
                                layer_idx=int(layer_idx),
                                strength=float(strength),
                                max_new_tokens=30,
                                temperature=0.7,
                            )
                        )
                        
                        # Export to Three.js format
                        exporter = ThreeJSExporter()
                        metadata = {
                            "model_name": app.model.model_name,
                            "prompt": prompt,
                            "concept_name": concept,
                            "injection_layer": int(layer_idx),
                            "injection_strength": float(strength),
                        }
                        
                        # Create in-memory data structure (no file saving)
                        import json
                        from pathlib import Path
                        import tempfile
                        
                        # Save temporarily
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            temp_path = f.name
                        
                        exporter.save_activations_for_threejs(
                            normal_acts=normal_acts,
                            steered_acts=steered_acts,
                            tokens=tokens,
                            metadata=metadata,
                            filename=Path(temp_path).name,
                        )
                        
                        # Load and create HTML
                        with open(temp_path, 'r') as f:
                            activation_data = json.load(f)
                        
                        html = create_threejs_html(activation_data)
                        
                        # Cleanup
                        Path(temp_path).unlink()
                        
                        return html
                        
                    except Exception as e:
                        return f"<p style='color:red;'>Error: {str(e)}</p>"
                
                def load_sample_visualization():
                    """Load pre-generated sample data."""
                    try:
                        import json
                        sample_path = "activation_steering_lab/mocked_data/happy_layer16_20251103_161946.json"
                        with open(sample_path, 'r') as f:
                            activation_data = json.load(f)
                        return create_threejs_html(activation_data)
                    except Exception as e:
                        return f"<p style='color:red;'>Error loading sample: {str(e)}</p>"
                
                visualize_3d_btn.click(
                    fn=generate_3d_visualization,
                    inputs=[brain_prompt, brain_concept, brain_layer, brain_strength],
                    outputs=brain_viewer
                )
                
                load_sample_btn.click(
                    fn=load_sample_visualization,
                    outputs=brain_viewer
                )

            # Tab 6: Advanced Experiments
            with gr.Tab("🔬 Advanced Experiments"):
                gr.Markdown("## Advanced Steering Experiments")

                with gr.Tabs():
                    # Sub-tab: Layer Analysis
                    with gr.Tab("Layer Analysis"):
                        gr.Markdown("""
                        Test the same concept across different layers to see where it works best.
                        """)

                        with gr.Row():
                            with gr.Column():
                                layer_prompt = gr.Textbox(label="Prompt", lines=2)
                                layer_concept = gr.Dropdown(
                                    label="Concept",
                                    choices=["Initialize first"],
                                    value="Initialize first",
                                    interactive=True
                                )
                                layer_strength = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Strength")
                                layer_analyze_btn = gr.Button("Analyze Layers", variant="primary")

                            with gr.Column():
                                layer_results = gr.Markdown("Results will appear here")

                        layer_analyze_btn.click(
                            fn=app.analyze_layers,
                            inputs=[layer_prompt, layer_concept, layer_strength],
                            outputs=layer_results
                        )

                    # Sub-tab: Strength Testing
                    with gr.Tab("Strength Explorer"):
                        gr.Markdown("""
                        Test different injection strengths to understand the strength parameter.
                        """)

                        with gr.Row():
                            with gr.Column():
                                strength_prompt = gr.Textbox(label="Prompt", lines=2)
                                strength_concept = gr.Dropdown(
                                    label="Concept",
                                    choices=["Initialize first"],
                                    value="Initialize first",
                                    interactive=True
                                )
                                strength_layer = gr.Slider(0, 31, value=16, step=1, label="Layer")
                                strength_test_btn = gr.Button("Test Strengths", variant="primary")

                            with gr.Column():
                                strength_results = gr.Markdown("Results will appear here")

                        strength_test_btn.click(
                            fn=app.test_strengths,
                            inputs=[strength_prompt, strength_concept, strength_layer],
                            outputs=strength_results
                        )

                    # Sub-tab: Emotion Mixer
                    with gr.Tab("Emotion Mixer"):
                        gr.Markdown("""
                        Mix multiple concepts with different weights.

                        **Format:** `concept1:weight1,concept2:weight2`
                        **Example:** `happy:0.7,excited:0.3`
                        """)

                        with gr.Row():
                            with gr.Column():
                                mix_input = gr.Textbox(
                                    label="Emotions & Weights",
                                    placeholder="happy:0.7,excited:0.3",
                                    lines=2
                                )
                                mix_layer = gr.Slider(0, 31, value=16, step=1, label="Layer")
                                mix_btn = gr.Button("Mix Concepts", variant="primary")

                            with gr.Column():
                                mix_result = gr.Markdown("Results will appear here")

                                with gr.Accordion("💡 Example Mixes", open=False):
                                    gr.Markdown("""
                                    - `happy:0.7,excited:0.3` → Cheerful
                                    - `formal:0.5,friendly:0.5` → Professional-warm
                                    - `sad:0.6,calm:0.4` → Melancholic peace
                                    """)

                        mix_btn.click(
                            fn=app.mix_emotions,
                            inputs=[mix_input, mix_layer],
                            outputs=mix_result
                        )

        # Footer with tips
        with gr.Accordion("💡 Tips & Tricks", open=False):
            tips = get_tips_and_tricks()
            gr.Markdown("\n".join(tips))

        # Demo button handler - pre-fills example values
        def load_demo_example():
            """Load a pre-filled example for quick testing (no typing needed!)"""
            return {
                steer_prompt: "Tell me about artificial intelligence",
                steer_concept: "enthusiastic",
                steer_layer: 16,
                steer_strength: 1.0  # Safe default for Phi-3 (tested range: 0.5-1.5)
            }

        demo_btn.click(
            fn=load_demo_example,
            outputs=[steer_prompt, steer_concept, steer_layer, steer_strength]
        )

        # Show available layers for selected concept
        def show_concept_layers(concept_name):
            if not concept_name or not app.initialized:
                return "*Select a concept to see available layers*"

            layers = app.get_available_layers(concept_name)
            if not layers:
                return f"❌ Concept '{concept_name}' not found"

            return f"✓ **'{concept_name}' available at layers**: {layers}\n\n💡 Adjust the layer slider to one of these values"

        # Update all concept dropdowns after initialization
        def update_all_dropdowns():
            if app.initialized:
                concepts = app.get_concept_list()
                # Return update dictionaries, not new components
                return [
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None),
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None),
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None),
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None),
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None),
                ]
            return [
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[]),
            ]

        # Update layer info when concept is selected
        steer_concept.change(
            fn=show_concept_layers,
            inputs=steer_concept,
            outputs=concept_layer_info
        )

        # Bind init button: chain initialization then update dropdowns
        init_btn.click(
            fn=app.initialize,
            outputs=init_status
        ).then(
            fn=update_all_dropdowns,
            outputs=[steer_concept, layer_concept, strength_concept, viz_concept, brain_concept]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1")
