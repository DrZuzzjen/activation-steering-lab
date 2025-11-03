"""
Main Gradio Interface for Activation Steering Lab
"""

import gradio as gr
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_steering_lab.model_wrapper import ModelWrapper
from activation_steering_lab.vector_library import VectorLibrary, create_default_concepts
from activation_steering_lab.injection_engine import InjectionEngine
from activation_steering_lab.educational_content import (
    get_explanation, format_layer_info, get_recommended_experiments,
    get_tips_and_tricks, EXAMPLE_PROMPTS, get_concept_pairs
)


class ActivationSteeringApp:
    """Main application class for the Gradio interface."""

    def __init__(self):
        self.model = None
        self.library = None
        self.engine = None
        self.initialized = False

    def initialize(self, progress=gr.Progress()):
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

            # Try to load saved vectors first
            recommended = self.model.get_recommended_layers()

            progress(0.55, desc="📦 Loading saved concept vectors...")
            yield f"✓ Model loaded: {self.model.model_name}\n✓ Engine ready\n\n📦 Loading saved vectors from disk..."

            try:
                self.library.load_all()
                num_loaded = len(self.library.list_concepts())

                if num_loaded > 0:
                    # Vectors loaded successfully from disk
                    progress(0.85, desc=f"✓ Loaded {num_loaded} concepts from cache")
                    yield f"✓ Model loaded\n✓ Engine ready\n✓ Loaded {num_loaded} concepts from cache!\n\n⚡ All vectors loaded instantly from disk"
                else:
                    # No saved vectors, extract them
                    raise FileNotFoundError("No saved vectors found")

            except Exception as e:
                # Fall back to extracting concepts
                progress(0.55, desc="🎨 No cache found, extracting concepts...")
                yield f"✓ Model loaded\n✓ Engine ready\n\n🎨 No saved vectors found, extracting...\nThis will take ~3 minutes (only once)"

                # Extract concepts with progress updates
                from activation_steering_lab.educational_content import get_concept_pairs
                all_concepts = []
                for category, pairs in get_concept_pairs().items():
                    all_concepts.extend(pairs)

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
        """Get list of available concepts."""
        if not self.initialized:
            return []
        return self.library.list_concepts()

    def create_custom_concept(self, name, concept_prompt, baseline_prompt, layer_idx, progress=gr.Progress()):
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

    def generate_steered(self, prompt, concept, layer_idx, strength, max_tokens, temperature, progress=gr.Progress()):
        """Generate with steering and comparison."""
        if not self.initialized:
            return "Initialize model first", "Initialize model first", ""

        try:
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

            # Tab 4: Advanced Experiments
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

        # Update all concept dropdowns after initialization
        def update_all_dropdowns():
            if app.initialized:
                concepts = app.get_concept_list()
                # Return update dictionaries, not new components
                return [
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None),
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None),
                    gr.Dropdown(choices=concepts, value=concepts[0] if concepts else None)
                ]
            return [
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[]),
                gr.Dropdown(choices=[])
            ]

        # Bind init button: chain initialization then update dropdowns
        init_btn.click(
            fn=app.initialize,
            outputs=init_status
        ).then(
            fn=update_all_dropdowns,
            outputs=[steer_concept, layer_concept, strength_concept]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1")
