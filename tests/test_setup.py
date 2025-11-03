"""
Quick test script to verify the setup works
This tests the core functionality without loading the full model.
"""

import sys
from pathlib import Path

# Add to path - updated for new tests/ directory structure
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from activation_steering_lab.model_wrapper import ModelWrapper
        print("✓ ModelWrapper")
    except Exception as e:
        print(f"✗ ModelWrapper: {e}")
        return False

    try:
        from activation_steering_lab.vector_library import VectorLibrary, ConceptVector
        print("✓ VectorLibrary")
    except Exception as e:
        print(f"✗ VectorLibrary: {e}")
        return False

    try:
        from activation_steering_lab.injection_engine import InjectionEngine
        print("✓ InjectionEngine")
    except Exception as e:
        print(f"✗ InjectionEngine: {e}")
        return False

    try:
        from activation_steering_lab.educational_content import get_explanation
        print("✓ EducationalContent")
    except Exception as e:
        print(f"✗ EducationalContent: {e}")
        return False

    try:
        import gradio as gr
        print("✓ Gradio")
    except Exception as e:
        print(f"✗ Gradio: {e}")
        return False

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")
    except Exception as e:
        print(f"✗ PyTorch: {e}")
        return False

    try:
        from transformers import AutoTokenizer
        print("✓ Transformers")
    except Exception as e:
        print(f"✗ Transformers: {e}")
        return False

    return True


def test_vector_library():
    """Test vector library functionality."""
    print("\nTesting VectorLibrary...")

    try:
        from activation_steering_lab.vector_library import VectorLibrary
        import torch

        library = VectorLibrary(save_dir="activation_steering_lab/test_vectors")
        print("✓ Library created")

        # Create a dummy concept vector
        from activation_steering_lab.vector_library import ConceptVector
        from datetime import datetime

        dummy_vector = ConceptVector(
            name="test_concept",
            vector=torch.randn(100),
            layer_idx=5,
            concept_prompt="test prompt",
            baseline_prompt="baseline prompt",
            norm=1.0,
            extraction_date=datetime.now().isoformat()
        )

        library.add_vector(dummy_vector)
        print("✓ Vector added")

        retrieved = library.get_vector("test_concept", 5)
        assert retrieved is not None
        print("✓ Vector retrieved")

        stats = library.get_vector_stats("test_concept", 5)
        assert 'norm' in stats
        print("✓ Stats computed")

        return True

    except Exception as e:
        print(f"✗ VectorLibrary test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_educational_content():
    """Test educational content."""
    print("\nTesting Educational Content...")

    try:
        from activation_steering_lab.educational_content import (
            get_explanation, get_recommended_experiments, get_tips_and_tricks
        )

        explanation = get_explanation('activations_vs_embeddings')
        assert len(explanation) > 0
        print("✓ Explanations working")

        experiments = get_recommended_experiments()
        assert len(experiments) > 0
        print("✓ Experiments loaded")

        tips = get_tips_and_tricks()
        assert len(tips) > 0
        print("✓ Tips loaded")

        return True

    except Exception as e:
        print(f"✗ Educational content test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Activation Steering Lab - Setup Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("VectorLibrary", test_vector_library),
        ("Educational Content", test_educational_content),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! Setup is correct.")
        print("\nNext steps:")
        print("1. Run: python -m activation_steering_lab.main")
        print("2. Click 'Initialize Model & Library' (takes ~5 minutes)")
        print("3. Start experimenting!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Missing dependencies: pip install -r requirements.txt")
        print("- Wrong Python version: Need Python 3.9+")
        print("- Import errors: Make sure you're in the right directory")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)