"""
Quick test: Generate ONE real sample to validate the pipeline.
This runs faster than the full dataset generation.

Usage:
    source venv/bin/activate
    python activation_steering_lab/mocked_data/generate_single_sample.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from activation_steering_lab.model_wrapper import ModelWrapper
from activation_steering_lab.vector_library import VectorLibrary
from activation_steering_lab.visualization import ActivationVisualizer
from activation_steering_lab.threejs_export import ThreeJSExporter


def main():
    """Generate a single real sample for validation."""
    
    print("\n" + "=" * 70)
    print("🧪 Single Sample Generation Test")
    print("=" * 70)
    print("\nThis will generate ONE real sample:")
    print("  - Concept: 'happy' at layer 16")
    print("  - Prompt: 'Tell me about happiness'")
    print("  - Quick test before full dataset generation")
    print("\n" + "=" * 70)
    
    # Initialize model and libraries
    print("\n📦 Loading model and vector library...")
    model_wrapper = ModelWrapper(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        device="auto"
    )
    model_wrapper.load_model()
    
    vector_library = VectorLibrary(save_dir="activation_steering_lab/saved_vectors")
    # Load all saved vectors
    vector_library.load_all()
    print(f"✓ Loaded {len(vector_library.list_concepts())} concepts")
    
    visualizer = ActivationVisualizer(
        model_wrapper=model_wrapper,
        vector_library=vector_library
    )
    
    # Create exporter
    output_dir = str(Path(__file__).parent)
    exporter = ThreeJSExporter(output_dir=output_dir)
    
    # Generate single sample
    print("\n🧠 Generating sample...")
    output_path = exporter.create_sample_data(
        visualizer=visualizer,
        prompt="Tell me about happiness",
        concept_name="happy",
        layer_idx=16,
        strength=2.0,
        max_new_tokens=30,
    )
    
    print("\n✅ Sample generation successful!")
    print(f"\n📁 Generated file:")
    print(f"   {Path(output_path).name}")
    print(f"\n💡 File can now be used for Three.js development")
    print(f"   Location: {output_path}")
    
    # Cleanup
    model_wrapper.clear_hooks()
    print("\n🧹 Cleaned up model hooks")
    
    # Load and display summary
    print("\n📊 Sample Summary:")
    data = exporter.load_threejs_data(output_path)
    
    print(f"\n   Concept: {data['metadata']['concept_name']}")
    print(f"   Injection Layer: {data['metadata']['injection_layer']}")
    print(f"   Peak Activation Layer: {data['peak_activation_layer']}")
    print(f"   Total Intensity: {data['total_intensity']:.2f}")
    print(f"   Top Correlated Layers: {data['top_correlated_layers']}")
    
    print("\n   Layer Correlations (Top 5):")
    correlations = [(int(k), v) for k, v in data['layer_correlations'].items()]
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for layer, corr in correlations[:5]:
        marker = "🎯" if layer == data['metadata']['injection_layer'] else "  "
        print(f"   {marker} Layer {layer:2d}: {corr:+.3f}")
    
    print("\n" + "=" * 70)
    print("✅ Validation complete! Ready for full dataset generation.")
    print("=" * 70)
    print("\n📝 Next steps:")
    print("   1. If this looks good, run: python activation_steering_lab/mocked_data/generate_mock_data.py")
    print("   2. Start Phase 2: Three.js scene implementation")
    print("   3. Use this file for initial Three.js testing")


if __name__ == "__main__":
    main()
