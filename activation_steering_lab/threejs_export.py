"""
Three.js Export Module for Activation Data
Exports activation data in a format optimized for Three.js visualization.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class ThreeJSExporter:
    """Handles exporting activation data for Three.js visualization."""

    def __init__(self, output_dir: str = "activation_steering_lab/mocked_data"):
        """
        Initialize the exporter.
        
        Args:
            output_dir: Directory to save exported JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_activations_for_threejs(
        self,
        normal_acts: Dict[int, torch.Tensor],
        steered_acts: Dict[int, torch.Tensor],
        tokens: List[str],
        metadata: dict,
        filename: Optional[str] = None,
        num_regions: int = 64,
    ) -> str:
        """
        Save activation data in Three.js-friendly format.

        Args:
            normal_acts: Normal generation activations {layer_idx: tensor[1, seq_len, hidden_size]}
            steered_acts: Steered generation activations
            tokens: Token strings from generation
            metadata: Dict with prompt, concept_name, injection_layer, injection_strength, model_name
            filename: Output filename (auto-generated if None)
            num_regions: Number of spatial regions to downsample to (default 64)

        Returns:
            Path to saved JSON file
        """
        # Validate inputs
        if not normal_acts or not steered_acts:
            raise ValueError("Both normal_acts and steered_acts must be provided")

        layer_indices = sorted(set(normal_acts.keys()) & set(steered_acts.keys()))
        if not layer_indices:
            raise ValueError("No overlapping layers between normal and steered activations")

        # Extract metadata
        concept_name = metadata.get("concept_name", "unknown")
        injection_layer = metadata.get("injection_layer", 0)
        injection_strength = metadata.get("injection_strength", 1.0)
        model_name = metadata.get("model_name", "unknown")
        prompt = metadata.get("prompt", "")

        # Get dimensions - use the minimum sequence length across all layers
        sample_tensor = normal_acts[layer_indices[0]]
        batch_size, seq_len, hidden_size = sample_tensor.shape
        
        # Find minimum sequence length across all layers (they might differ)
        min_seq_len = seq_len
        for layer_idx in layer_indices:
            layer_seq_len = normal_acts[layer_idx].shape[1]
            min_seq_len = min(min_seq_len, layer_seq_len)
        
        seq_len = min_seq_len  # Use the minimum to avoid index errors
        num_layers = len(layer_indices)

        # Initialize data structures
        normal_activations_data = {}
        steered_activations_data = {}
        difference_activations_data = {}
        layer_correlations = {}

        # Downsample and extract activations for each layer
        injection_vector_full = None
        region_size = max(1, hidden_size // num_regions)

        print(f"Exporting activations: {num_layers} layers, {hidden_size} dims -> {num_regions} regions")

        for layer_idx in layer_indices:
            # Get actual sequence length for this layer (they may differ)
            layer_seq_len = normal_acts[layer_idx].shape[1]
            token_idx = layer_seq_len - 1  # Use last token of this layer
            
            # Safety check
            if token_idx < 0 or token_idx >= layer_seq_len:
                print(f"Warning: Invalid token_idx {token_idx} for layer {layer_idx} with seq_len {layer_seq_len}")
                token_idx = 0
            
            # Extract last token activations
            normal_vec = normal_acts[layer_idx][0, token_idx, :].detach().cpu().numpy()
            steered_vec = steered_acts[layer_idx][0, token_idx, :].detach().cpu().numpy()
            diff_vec = steered_vec - normal_vec

            # Clean up NaN/Inf values
            normal_vec = np.nan_to_num(normal_vec, nan=0.0, posinf=0.0, neginf=0.0)
            steered_vec = np.nan_to_num(steered_vec, nan=0.0, posinf=0.0, neginf=0.0)
            diff_vec = np.nan_to_num(diff_vec, nan=0.0, posinf=0.0, neginf=0.0)

            # Store full injection vector for correlation calculation
            if layer_idx == injection_layer:
                injection_vector_full = diff_vec.copy()

            # Downsample to regions by averaging chunks
            normal_regions = []
            steered_regions = []
            diff_regions = []

            for r in range(num_regions):
                start = r * region_size
                end = min(start + region_size, hidden_size)

                if start < hidden_size:
                    normal_regions.append(float(np.mean(normal_vec[start:end])))
                    steered_regions.append(float(np.mean(steered_vec[start:end])))
                    diff_regions.append(float(np.mean(np.abs(diff_vec[start:end]))))
                else:
                    normal_regions.append(0.0)
                    steered_regions.append(0.0)
                    diff_regions.append(0.0)

            # Store downsampled data
            normal_activations_data[str(layer_idx)] = normal_regions
            steered_activations_data[str(layer_idx)] = steered_regions
            difference_activations_data[str(layer_idx)] = diff_regions

        # Calculate layer correlations with injection layer
        if injection_vector_full is not None:
            for layer_idx in layer_indices:
                if layer_idx == injection_layer:
                    layer_correlations[str(layer_idx)] = 1.0
                else:
                    # Get actual sequence length for this layer
                    layer_seq_len = steered_acts[layer_idx].shape[1]
                    layer_token_idx = layer_seq_len - 1
                    
                    if layer_token_idx < 0 or layer_token_idx >= layer_seq_len:
                        layer_token_idx = 0
                    
                    layer_vec = steered_acts[layer_idx][0, layer_token_idx, :].detach().cpu().numpy()
                    layer_vec = layer_vec - normal_acts[layer_idx][0, layer_token_idx, :].detach().cpu().numpy()
                    layer_vec = np.nan_to_num(layer_vec, nan=0.0, posinf=0.0, neginf=0.0)

                    # Correlation coefficient
                    corr = np.corrcoef(injection_vector_full, layer_vec)[0, 1]
                    corr = np.nan_to_num(corr, nan=0.0)
                    layer_correlations[str(layer_idx)] = float(corr)
        else:
            # No injection layer found, set all correlations to 0
            for layer_idx in layer_indices:
                layer_correlations[str(layer_idx)] = 0.0

        # Find peak activation layer and top correlated layers
        diff_intensities = {
            layer_idx: sum(difference_activations_data[str(layer_idx)])
            for layer_idx in layer_indices
        }
        peak_layer = max(diff_intensities, key=diff_intensities.get)
        total_intensity = sum(diff_intensities.values())

        top_correlated = sorted(
            layer_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        top_correlated_layers = [int(layer) for layer, _ in top_correlated]

        # Build final JSON structure
        export_data = {
            "metadata": {
                "model": model_name,
                "prompt": prompt,
                "concept_name": concept_name,
                "injection_layer": injection_layer,
                "injection_strength": injection_strength,
                "sequence_length": seq_len,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "downsampled_regions": num_regions,
                "timestamp": datetime.now().isoformat(),
            },
            "normal_activations": normal_activations_data,
            "steered_activations": steered_activations_data,
            "difference_activations": difference_activations_data,
            "tokens": tokens[:seq_len],  # Include all tokens up to sequence length
            "layer_correlations": layer_correlations,
            "peak_activation_layer": peak_layer,
            "total_intensity": float(total_intensity),
            "top_correlated_layers": top_correlated_layers,
        }

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{concept_name}_layer{injection_layer}_{timestamp}.json"

        output_path = self.output_dir / filename
        
        # Save to JSON with compact formatting for array data
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved Three.js data to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Peak activation layer: {peak_layer}")
        print(f"  Total intensity: {total_intensity:.2f}")

        return str(output_path)

    def load_threejs_data(self, filepath: str) -> dict:
        """
        Load Three.js activation data from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Parsed activation data dictionary
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"✓ Loaded Three.js data from: {filepath}")
        print(f"  Concept: {data['metadata']['concept_name']}")
        print(f"  Injection layer: {data['metadata']['injection_layer']}")
        print(f"  Layers: {data['metadata']['num_layers']}")
        print(f"  Regions per layer: {data['metadata']['downsampled_regions']}")

        return data

    def create_sample_data(
        self,
        visualizer,
        prompt: str,
        concept_name: str,
        layer_idx: int,
        strength: float = 2.0,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Generate sample activation data by running the model once.

        Args:
            visualizer: ActivationVisualizer instance with loaded model
            prompt: Input prompt
            concept_name: Concept to inject (e.g., "happy", "sad")
            layer_idx: Layer to inject concept at
            strength: Injection strength
            max_new_tokens: Max tokens to generate

        Returns:
            Path to saved JSON file
        """
        print(f"\n🧠 Generating sample data for concept '{concept_name}' at layer {layer_idx}...")

        # Capture activations using existing visualizer
        normal_text, steered_text, normal_acts, steered_acts, tokens = (
            visualizer.capture_activations_for_comparison(
                prompt=prompt,
                concept_name=concept_name,
                layer_idx=layer_idx,
                strength=strength,
                max_new_tokens=max_new_tokens,
            )
        )

        print(f"\n📝 Generated text samples:")
        print(f"  Normal: {normal_text[:100]}...")
        print(f"  Steered: {steered_text[:100]}...")

        # Prepare metadata
        metadata = {
            "model_name": visualizer.model.model_name,
            "prompt": prompt,
            "concept_name": concept_name,
            "injection_layer": layer_idx,
            "injection_strength": strength,
            "normal_text": normal_text,
            "steered_text": steered_text,
        }

        # Export to Three.js format
        output_path = self.save_activations_for_threejs(
            normal_acts=normal_acts,
            steered_acts=steered_acts,
            tokens=tokens,
            metadata=metadata,
        )

        return output_path


def generate_sample_dataset(
    visualizer,
    output_dir: str = "activation_steering_lab/mocked_data"
) -> List[str]:
    """
    Generate a complete sample dataset for Three.js development.

    Creates activation data for multiple concepts at different layers.

    Args:
        visualizer: ActivationVisualizer instance with loaded model
        output_dir: Directory to save samples

    Returns:
        List of paths to generated JSON files
    """
    exporter = ThreeJSExporter(output_dir=output_dir)

    # Configuration for sample data generation
    concepts = [
        ("happy", "Tell me about happiness"),
        ("sad", "Tell me about sadness"),
        ("calm", "Tell me about peace and tranquility"),
        ("excited", "Tell me about adventure"),
    ]

    layers = [8, 16, 24]  # Early, middle, late layers
    strength = 2.0

    generated_files = []

    print("\n" + "=" * 70)
    print("🚀 Generating Sample Dataset for Three.js Development")
    print("=" * 70)

    for concept_name, prompt in concepts:
        for layer_idx in layers:
            try:
                output_path = exporter.create_sample_data(
                    visualizer=visualizer,
                    prompt=prompt,
                    concept_name=concept_name,
                    layer_idx=layer_idx,
                    strength=strength,
                    max_new_tokens=30,  # Keep it short for speed
                )
                generated_files.append(output_path)
                print(f"✓ Generated: {concept_name} @ L{layer_idx}")

            except Exception as e:
                print(f"✗ Failed to generate {concept_name} @ L{layer_idx}: {e}")

    print("\n" + "=" * 70)
    print(f"✓ Generated {len(generated_files)} sample files")
    print(f"  Location: {output_dir}")
    print("=" * 70)

    return generated_files
