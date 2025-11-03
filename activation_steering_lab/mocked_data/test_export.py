"""
Test script for Three.js export functionality.
Validates data format and structure without running full model.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from activation_steering_lab.threejs_export import ThreeJSExporter


def create_mock_activations(num_layers=32, seq_len=15, hidden_size=3072):
    """Create fake activation tensors for testing."""
    normal_acts = {}
    steered_acts = {}
    
    for layer_idx in range(num_layers):
        # Create random activations
        normal_acts[layer_idx] = torch.randn(1, seq_len, hidden_size) * 0.5
        
        # Steered activations are similar but with injection at layer 16
        steered_acts[layer_idx] = normal_acts[layer_idx].clone()
        if layer_idx == 16:
            # Add strong signal at injection layer
            steered_acts[layer_idx] += torch.randn(1, seq_len, hidden_size) * 2.0
        elif abs(layer_idx - 16) <= 3:
            # Weaker signal in nearby layers
            steered_acts[layer_idx] += torch.randn(1, seq_len, hidden_size) * 0.8
    
    return normal_acts, steered_acts


def test_export_format():
    """Test the Three.js export format."""
    print("\n" + "=" * 70)
    print("🧪 Testing Three.js Export Format")
    print("=" * 70)
    
    # Create exporter
    exporter = ThreeJSExporter(output_dir="activation_steering_lab/mocked_data")
    
    # Create mock data
    print("\n1. Creating mock activation data...")
    normal_acts, steered_acts = create_mock_activations()
    tokens = ["Tell", "me", "about", "happiness", "▁I", "▁would", "▁say", "..."]
    
    metadata = {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "prompt": "Tell me about happiness",
        "concept_name": "happy",
        "injection_layer": 16,
        "injection_strength": 2.0,
    }
    
    # Export data
    print("\n2. Exporting to JSON...")
    output_path = exporter.save_activations_for_threejs(
        normal_acts=normal_acts,
        steered_acts=steered_acts,
        tokens=tokens,
        metadata=metadata,
        filename="test_sample.json",
        num_regions=64,
    )
    
    # Load and validate
    print("\n3. Loading and validating JSON structure...")
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Validate structure
    required_keys = [
        "metadata",
        "normal_activations",
        "steered_activations",
        "difference_activations",
        "tokens",
        "layer_correlations",
        "peak_activation_layer",
        "total_intensity",
        "top_correlated_layers",
    ]
    
    print("\n4. Validation checks:")
    for key in required_keys:
        if key in data:
            print(f"   ✓ {key}")
        else:
            print(f"   ✗ {key} MISSING!")
            return False
    
    # Check metadata
    meta = data["metadata"]
    print("\n5. Metadata:")
    print(f"   Model: {meta['model']}")
    print(f"   Concept: {meta['concept_name']}")
    print(f"   Injection layer: {meta['injection_layer']}")
    print(f"   Layers: {meta['num_layers']}")
    print(f"   Hidden size: {meta['hidden_size']}")
    print(f"   Regions: {meta['downsampled_regions']}")
    
    # Check data shapes
    print("\n6. Data shapes:")
    print(f"   Normal activations: {len(data['normal_activations'])} layers")
    print(f"   Steered activations: {len(data['steered_activations'])} layers")
    print(f"   Difference activations: {len(data['difference_activations'])} layers")
    print(f"   Tokens: {len(data['tokens'])} tokens")
    print(f"   Layer correlations: {len(data['layer_correlations'])} entries")
    
    # Check region counts
    sample_layer = list(data['normal_activations'].keys())[0]
    regions_count = len(data['normal_activations'][sample_layer])
    print(f"   Regions per layer: {regions_count}")
    
    if regions_count != 64:
        print(f"   ✗ Expected 64 regions, got {regions_count}")
        return False
    
    # Check correlation values
    print("\n7. Layer correlations (sample):")
    correlations = [(int(k), v) for k, v in data['layer_correlations'].items()]
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for layer, corr in correlations[:5]:
        marker = "🎯" if layer == 16 else "  "
        print(f"   {marker} Layer {layer:2d}: {corr:+.3f}")
    
    # Check peak detection
    print("\n8. Peak activation:")
    print(f"   Peak layer: {data['peak_activation_layer']}")
    print(f"   Total intensity: {data['total_intensity']:.2f}")
    print(f"   Top correlated: {data['top_correlated_layers']}")
    
    # File size
    file_size = Path(output_path).stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    print(f"\n9. File size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 10:
        print("   ⚠️  Warning: File larger than 10MB - may need compression")
    else:
        print("   ✓ File size acceptable for browser loading")
    
    print("\n" + "=" * 70)
    print("✅ All validation checks passed!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   - Run generate_mock_data.py to create real samples")
    print("   - Use test_sample.json for Three.js development")
    print("   - Expected format validated and ready for frontend")
    
    return True


if __name__ == "__main__":
    success = test_export_format()
    sys.exit(0 if success else 1)
