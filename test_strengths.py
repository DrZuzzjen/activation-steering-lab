#!/usr/bin/env python3
"""
Test different steering strengths to find optimal values for Phi-3
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def test_multiple_strengths():
    """Test steering with multiple strength values."""
    print("=" * 80)
    print("Testing Different Steering Strengths with Phi-3")
    print("=" * 80)

    # Step 1: Load model
    print("\n[1/3] Loading model from cache...")
    from activation_steering_lab.model_wrapper import ModelWrapper
    model = ModelWrapper()
    model.load_model()
    print(f"✓ Model loaded: {model.model_name}")

    # Step 2: Create library and load concepts
    print("\n[2/3] Loading concept vectors...")
    from activation_steering_lab.vector_library import VectorLibrary
    library = VectorLibrary()
    library.load_all()
    concepts = library.list_concepts()
    print(f"✓ Loaded {len(concepts)} concepts")

    # Step 3: Create injection engine
    print("\n[3/3] Creating injection engine...")
    from activation_steering_lab.injection_engine import InjectionEngine
    engine = InjectionEngine(model, library)
    print("✓ Engine ready\n")

    # Test parameters
    prompt = "Tell me about artificial intelligence"
    concept = "happy"
    layer = 16
    strengths = [0.5, 1.0, 1.5, 2.0, 3.0]

    print("=" * 80)
    print(f"📝 Prompt: '{prompt}'")
    print(f"🎨 Concept: '{concept}'")
    print(f"🔧 Layer: {layer}")
    print("=" * 80)

    # Generate normal output once (for comparison)
    print("\n🔵 NORMAL OUTPUT (no steering):")
    print("-" * 80)
    results = engine.generate_comparison(
        prompt=prompt,
        concept_name=concept,
        layer_idx=layer,
        strength=0.0,  # No steering
        max_new_tokens=50,
        temperature=0.7
    )
    print(results['normal'])
    print("-" * 80)

    # Test each strength
    for strength in strengths:
        print(f"\n\n{'=' * 80}")
        print(f"🟢 STEERED OUTPUT - Strength: {strength}")
        print("=" * 80)

        try:
            results = engine.generate_comparison(
                prompt=prompt,
                concept_name=concept,
                layer_idx=layer,
                strength=strength,
                max_new_tokens=50,
                temperature=0.7
            )
            steered = results['steered']
            print(steered)

            # Quick quality check
            if len(steered.split()) < 10:
                print("\n⚠️  WARNING: Output too short!")
            elif steered.count("!!") > 3 or steered.count("we we") > 2:
                print("\n⚠️  WARNING: Output seems incoherent/repetitive!")
            else:
                print("\n✓ Output looks reasonable")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n\n" + "=" * 80)
    print("✅ STRENGTH TEST COMPLETE!")
    print("=" * 80)
    print("\n📊 RECOMMENDATIONS:")
    print("  - Strengths 0.5-1.5: Subtle steering, coherent output")
    print("  - Strengths 2.0-3.0: Strong steering, may lose coherence")
    print("  - Strengths > 3.0: Very strong, likely incoherent")
    print("\n💡 For Phi-3, try staying in the 0.8-1.5 range for best results!")

    return True


if __name__ == "__main__":
    try:
        success = test_multiple_strengths()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
