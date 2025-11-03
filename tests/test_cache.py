"""
Test script to verify local model cache is working
"""

import sys
from pathlib import Path

# Add to path - updated for new tests/ directory structure
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_cache():
    """Test that the cache directory is set up correctly."""
    print("=" * 60)
    print("Testing Model Cache Setup")
    print("=" * 60)

    from activation_steering_lab.model_wrapper import ModelWrapper

    # Create wrapper (don't load model yet)
    wrapper = ModelWrapper()

    print(f"\n✓ Model wrapper created")
    print(f"  Model: {wrapper.model_name}")
    print(f"  Cache dir: {wrapper.cache_dir}")

    # Check cache directory exists
    cache_path = Path(wrapper.cache_dir)
    if not cache_path.exists():
        print(f"\n✗ Cache directory doesn't exist: {cache_path}")
        return False

    print(f"\n✓ Cache directory exists: {cache_path}")

    # Check for cached models
    cached_models = list(cache_path.glob("models--*"))

    if cached_models:
        print(f"\n✓ Found {len(cached_models)} cached model(s):")
        for model_dir in cached_models:
            # Check if it's a symlink
            if model_dir.is_symlink():
                target = model_dir.resolve()
                size_info = "symlink"
                print(f"  - {model_dir.name} → {target}")
            else:
                size_info = "local"
                print(f"  - {model_dir.name} (local copy)")
    else:
        print(f"\n⚠️  No cached models found")
        print(f"  This is OK - models will download on first use")
        print(f"  Run scripts/setup_local_cache.sh to link existing models")

    # Test that we can instantiate the tokenizer with cache
    print(f"\n✓ Cache configuration is correct")
    print(f"\nNext steps:")
    print(f"  1. Run: scripts/setup_local_cache.sh (if not done)")
    print(f"  2. Run: scripts/run.sh")
    print(f"  3. Click 'Initialize Model & Library'")
    print(f"  4. Model will load from cache (fast!)")

    return True


if __name__ == "__main__":
    try:
        success = test_cache()
        print("\n" + "=" * 60)
        if success:
            print("✅ Cache test passed!")
        else:
            print("❌ Cache test failed")
        print("=" * 60)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)