"""
Generate Mock Data Script
Run this script once to capture real model activations and save them for Three.js development.

Usage:
    python -m activation_steering_lab.mocked_data.generate_mock_data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from activation_steering_lab.model_wrapper import ModelWrapper
from activation_steering_lab.vector_library import VectorLibrary
from activation_steering_lab.visualization import ActivationVisualizer
from activation_steering_lab.threejs_export import generate_sample_dataset


def main():
    """Generate sample activation data for Three.js development."""
    
    print("\n" + "=" * 70)
    print("🔬 Mock Data Generation for Three.js Visualization")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Load Phi-3-mini model (14GB - one time)")
    print("  2. Generate activations for 4 concepts × 3 layers = 12 samples")
    print("  3. Save JSON files to activation_steering_lab/mocked_data/")
    print("  4. Enable rapid Three.js development without reloading model")
    print("\n" + "=" * 70)
    
    input("\nPress Enter to start generation (or Ctrl+C to cancel)...")
    
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
    
    # Generate sample dataset
    output_dir = str(Path(__file__).parent)
    generated_files = generate_sample_dataset(
        visualizer=visualizer,
        output_dir=output_dir
    )
    
    print("\n✅ Mock data generation complete!")
    print(f"\n📁 Generated {len(generated_files)} files:")
    for filepath in generated_files:
        print(f"   - {Path(filepath).name}")
    
    print("\n💡 Next steps:")
    print("   1. Use these JSON files for Three.js development")
    print("   2. No need to reload the model during visualization work")
    print("   3. Each file is ~2-5MB for fast loading in browser")
    
    # Cleanup
    model_wrapper.clear_hooks()
    print("\n🧹 Model hooks cleared. You can now start Phase 2!")


if __name__ == "__main__":
    main()
