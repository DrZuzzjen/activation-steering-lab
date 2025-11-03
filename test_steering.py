#!/usr/bin/env python3
"""
Simple test script to verify activation steering works
WITHOUT Gradio - just pure Python
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def test_steering():
    """Test activation steering with a simple example."""
    print("=" * 70)
    print("Testing Activation Steering (No Gradio)")
    print("=" * 70)

    # Step 1: Load model
    print("\n[1/5] Loading model from cache...")
    from activation_steering_lab.model_wrapper import ModelWrapper
    model = ModelWrapper()
    model.load_model()
    print(f"✓ Model loaded: {model.model_name}")
    print(f"✓ Device: {model.device}")

    # Step 2: Create library
    print("\n[2/5] Creating vector library...")
    from activation_steering_lab.vector_library import VectorLibrary
    library = VectorLibrary()
    print("✓ Library created")

    # Step 3: Load or create concept vectors
    print("\n[3/5] Loading saved concept vectors...")
    try:
        library.load_all()
        concepts = library.list_concepts()
        print(f"✓ Loaded {len(concepts)} concepts from disk")
        print(f"  Available: {', '.join(concepts[:5])}...")
    except:
        concepts = []

    # If no concepts exist, extract 'happy' for testing
    if len(concepts) == 0:
        print("\n  No saved concepts found. Extracting 'happy' for testing...")
        print("  (This takes ~10 seconds)")
        vector = library.compute_concept_vector(
            model_wrapper=model,
            concept_prompt="Respond in a very happy and enthusiastic way!",
            baseline_prompt="Respond normally.",
            layer_idx=16,
            concept_name="happy"
        )
        library.add_vector(vector)
        library.save_vector("happy", 16)
        print("✓ 'happy' concept extracted and saved")

    # Step 4: Create injection engine
    print("\n[4/5] Creating injection engine...")
    from activation_steering_lab.injection_engine import InjectionEngine
    engine = InjectionEngine(model, library)
    print("✓ Engine ready")

    # Step 5: Test steering
    print("\n[5/5] Testing steering with 'happy' concept...")
    print("-" * 70)

    prompt = "Tell me about the weather"
    concept = "happy"
    layer = 16
    strength = 2.0

    print(f"\n📝 Prompt: '{prompt}'")
    print(f"🎨 Concept: '{concept}'")
    print(f"🔧 Layer: {layer}, Strength: {strength}")
    print("\n" + "=" * 70)

    # Generate comparison (both normal and steered)
    normal, steered = engine.generate_comparison(
        prompt=prompt,
        concept_name=concept,
        layer_idx=layer,
        strength=strength,
        max_new_tokens=50,
        temperature=0.7
    )

    print("\n🔵 NORMAL OUTPUT (no steering):")
    print("-" * 70)
    print(normal)

    print("\n" + "=" * 70)
    print(f"\n🟢 STEERED OUTPUT (with '{concept}'):")
    print("-" * 70)
    print(steered)

    print("\n" + "=" * 70)
    print("✅ STEERING TEST COMPLETE!")
    print("=" * 70)
    print("\n💡 If you see different outputs above, steering is working!")
    print("   The steered output should feel more positive/happy.")
    print("\n🎯 Now you can use the Gradio interface with confidence!")

    return True


if __name__ == "__main__":
    try:
        success = test_steering()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
