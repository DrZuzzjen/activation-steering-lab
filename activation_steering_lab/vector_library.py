"""
Concept Vector Library
Stores, manages, and computes concept vectors for activation steering.
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ConceptVector:
    """
    Represents a single concept vector with metadata.

    Why store metadata?
    - Helps understand which layers/prompts work best
    - Enables reproducibility
    - Useful for educational visualization
    """
    name: str
    vector: torch.Tensor  # The actual activation difference
    layer_idx: int        # Which layer this was extracted from
    concept_prompt: str   # The prompt that evoked this concept
    baseline_prompt: str  # The neutral baseline
    norm: float          # Vector magnitude (for normalization)
    extraction_date: str


class VectorLibrary:
    """
    Manages a collection of concept vectors.

    Think of this as a "concept dictionary" - we extract concepts once,
    then reuse them for steering in different contexts.
    """

    def __init__(self, save_dir: str = "activation_steering_lab/saved_vectors"):
        """
        Initialize the vector library.

        Args:
            save_dir: Directory to save/load vectors
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage: {concept_name: {layer_idx: ConceptVector}}
        self.vectors: Dict[str, Dict[int, ConceptVector]] = {}

    def compute_concept_vector(
        self,
        model_wrapper,
        concept_prompt: str,
        baseline_prompt: str,
        layer_idx: int,
        concept_name: Optional[str] = None
    ) -> ConceptVector:
        """
        Extract a concept vector by computing the difference between two activations.

        The key insight of activation steering:
        concept_vector = activation(concept_prompt) - activation(baseline_prompt)

        Why subtraction?
        - Removes shared/common features
        - Isolates the specific concept
        - Example: "happy weather" - "weather" = "happiness"

        Args:
            model_wrapper: Loaded model wrapper
            concept_prompt: Text that evokes the desired concept
            baseline_prompt: Neutral text for comparison
            layer_idx: Which layer to extract from
            concept_name: Name for this concept (auto-generated if None)

        Returns:
            ConceptVector object
        """
        from datetime import datetime

        # Register capture hook for this layer
        model_wrapper.clear_hooks()
        model_wrapper.register_capture_hook(layer_idx)

        # Forward pass with concept prompt
        _ = model_wrapper.forward_pass(concept_prompt)
        concept_activation = model_wrapper.captured_activations[layer_idx]

        # Forward pass with baseline prompt
        model_wrapper.clear_hooks()
        model_wrapper.register_capture_hook(layer_idx)
        _ = model_wrapper.forward_pass(baseline_prompt)
        baseline_activation = model_wrapper.captured_activations[layer_idx]

        # Compute difference vector
        # Take the last token's activation (position -1)
        # Shape: [batch=1, seq_len, hidden_dim] -> [hidden_dim]
        concept_vec = concept_activation[0, -1, :] - baseline_activation[0, -1, :]

        # Compute statistics
        vector_norm = torch.norm(concept_vec).item()

        # Auto-generate name if not provided
        if concept_name is None:
            concept_name = self._generate_name(concept_prompt, baseline_prompt)

        # Create ConceptVector object
        concept_vector = ConceptVector(
            name=concept_name,
            vector=concept_vec.cpu(),  # Move to CPU for storage
            layer_idx=layer_idx,
            concept_prompt=concept_prompt,
            baseline_prompt=baseline_prompt,
            norm=vector_norm,
            extraction_date=datetime.now().isoformat()
        )

        # Clean up hooks
        model_wrapper.clear_hooks()

        return concept_vector

    def add_vector(self, concept_vector: ConceptVector) -> None:
        """
        Add a concept vector to the library.

        Vectors are organized by concept name and layer:
        library[name][layer] = vector
        """
        name = concept_vector.name
        layer = concept_vector.layer_idx

        if name not in self.vectors:
            self.vectors[name] = {}

        self.vectors[name][layer] = concept_vector

    def get_vector(self, concept_name: str, layer_idx: Optional[int] = None) -> Optional[ConceptVector]:
        """
        Retrieve a concept vector.

        Args:
            concept_name: Name of the concept
            layer_idx: Specific layer (if None, returns the first available)

        Returns:
            ConceptVector or None if not found
        """
        if concept_name not in self.vectors:
            return None

        if layer_idx is None:
            # Return first available layer
            layer_idx = list(self.vectors[concept_name].keys())[0]

        return self.vectors[concept_name].get(layer_idx)

    def get_vector_nearest_layer(
        self,
        concept_name: str,
        layer_idx: Optional[int] = None,
    ) -> Optional[Tuple[ConceptVector, int]]:
        """Return concept vector from requested or nearest available layer."""
        if concept_name not in self.vectors or not self.vectors[concept_name]:
            return None

        available_layers = sorted(self.vectors[concept_name].keys())

        if layer_idx is None or layer_idx in self.vectors[concept_name]:
            target_layer = layer_idx if layer_idx is not None else available_layers[0]
            vector = self.vectors[concept_name][target_layer]
            return vector, target_layer

        nearest_layer = min(available_layers, key=lambda idx: abs(idx - layer_idx))
        return self.vectors[concept_name][nearest_layer], nearest_layer

    def list_concepts(self) -> list[str]:
        """Get list of all concept names in the library."""
        return list(self.vectors.keys())

    def get_layers_for_concept(self, concept_name: str) -> list[int]:
        """Get all layers where this concept has been extracted."""
        if concept_name not in self.vectors:
            return []
        return list(self.vectors[concept_name].keys())

    def save_vector(self, concept_name: str, layer_idx: int) -> None:
        """
        Save a single concept vector to disk.

        File format: JSON + separate .pt file for tensor
        Why separate files?
        - JSON is human-readable (metadata)
        - .pt is efficient for tensors
        """
        vector = self.get_vector(concept_name, layer_idx)
        if vector is None:
            raise ValueError(f"Vector {concept_name} at layer {layer_idx} not found")

        # Create filename-safe name
        safe_name = self._safe_filename(concept_name)
        base_path = self.save_dir / f"{safe_name}_layer{layer_idx}"

        # Save metadata as JSON
        metadata = {
            'name': vector.name,
            'layer_idx': vector.layer_idx,
            'concept_prompt': vector.concept_prompt,
            'baseline_prompt': vector.baseline_prompt,
            'norm': vector.norm,
            'extraction_date': vector.extraction_date
        }

        with open(f"{base_path}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save tensor
        torch.save(vector.vector, f"{base_path}_tensor.pt")

    def load_vector(self, concept_name: str, layer_idx: int) -> ConceptVector:
        """
        Load a concept vector from disk.

        Returns:
            ConceptVector object
        """
        safe_name = self._safe_filename(concept_name)
        base_path = self.save_dir / f"{safe_name}_layer{layer_idx}"

        # Load metadata
        with open(f"{base_path}_meta.json", 'r') as f:
            metadata = json.load(f)

        # Load tensor
        vector_tensor = torch.load(f"{base_path}_tensor.pt")

        # Create ConceptVector
        concept_vector = ConceptVector(
            name=metadata['name'],
            vector=vector_tensor,
            layer_idx=metadata['layer_idx'],
            concept_prompt=metadata['concept_prompt'],
            baseline_prompt=metadata['baseline_prompt'],
            norm=metadata['norm'],
            extraction_date=metadata['extraction_date']
        )

        # Add to library
        self.add_vector(concept_vector)

        return concept_vector

    def save_all(self) -> None:
        """Save all vectors in the library to disk."""
        for concept_name in self.vectors:
            for layer_idx in self.vectors[concept_name]:
                self.save_vector(concept_name, layer_idx)

    def load_all(self) -> None:
        """
        Load all saved vectors from disk.

        Searches for all *_meta.json files in save_dir.
        """
        for meta_file in self.save_dir.glob("*_meta.json"):
            # Parse filename to get concept name and layer
            # Format: conceptname_layer5_meta.json
            base_name = meta_file.stem.replace("_meta", "")

            # Extract layer number
            parts = base_name.split("_layer")
            if len(parts) != 2:
                continue

            safe_name = parts[0]
            layer_idx = int(parts[1])

            # Load metadata to get real name
            with open(meta_file, 'r') as f:
                metadata = json.load(f)

            concept_name = metadata['name']

            # Load the vector
            try:
                self.load_vector(concept_name, layer_idx)
            except Exception as e:
                print(f"Warning: Could not load {concept_name} layer {layer_idx}: {e}")

    def get_vector_stats(self, concept_name: str, layer_idx: int) -> Dict:
        """
        Get statistics about a concept vector.

        Useful for educational purposes and debugging.
        """
        vector = self.get_vector(concept_name, layer_idx)
        if vector is None:
            return {}

        vec_np = vector.vector.numpy()

        return {
            'name': vector.name,
            'layer': vector.layer_idx,
            'norm': vector.norm,
            'mean': float(np.mean(vec_np)),
            'std': float(np.std(vec_np)),
            'min': float(np.min(vec_np)),
            'max': float(np.max(vec_np)),
            'sparsity': float(np.mean(np.abs(vec_np) < 0.01)),  # % of near-zero values
            'top_5_dims': np.argsort(np.abs(vec_np))[-5:].tolist()  # Most important dimensions
        }

    def normalize_vector(self, concept_vector: ConceptVector, target_norm: float = 1.0) -> ConceptVector:
        """
        Normalize a vector to a target magnitude.

        Why normalize?
        - Makes strength parameter more intuitive
        - Enables fair comparison between concepts
        - Prevents numerical instability

        Args:
            concept_vector: Vector to normalize
            target_norm: Desired L2 norm

        Returns:
            New ConceptVector with normalized vector
        """
        current_norm = torch.norm(concept_vector.vector)

        if current_norm == 0:
            return concept_vector

        normalized = concept_vector.vector * (target_norm / current_norm)

        # Create new ConceptVector with normalized vector
        return ConceptVector(
            name=concept_vector.name,
            vector=normalized,
            layer_idx=concept_vector.layer_idx,
            concept_prompt=concept_vector.concept_prompt,
            baseline_prompt=concept_vector.baseline_prompt,
            norm=target_norm,
            extraction_date=concept_vector.extraction_date
        )

    def combine_vectors(
        self,
        vectors: list[Tuple[ConceptVector, float]],
        new_name: str
    ) -> ConceptVector:
        """
        Combine multiple concept vectors with weights.

        This enables mixing emotions/concepts!
        Example: 0.7 * happy + 0.3 * excited

        Args:
            vectors: List of (ConceptVector, weight) tuples
            new_name: Name for the combined concept

        Returns:
            New combined ConceptVector
        """
        if not vectors:
            raise ValueError("Need at least one vector to combine")

        # Check all vectors are from same layer
        layer = vectors[0][0].layer_idx
        if not all(v.layer_idx == layer for v, _ in vectors):
            raise ValueError("All vectors must be from the same layer")

        # Weighted sum
        combined = torch.zeros_like(vectors[0][0].vector)
        concept_prompts = []
        weights_str = []

        for vector, weight in vectors:
            combined += weight * vector.vector
            concept_prompts.append(f"{weight:.1f}*{vector.name}")
            weights_str.append(f"{weight:.1f}*{vector.concept_prompt}")

        # Create combined concept
        from datetime import datetime
        return ConceptVector(
            name=new_name,
            vector=combined,
            layer_idx=layer,
            concept_prompt=" + ".join(weights_str),
            baseline_prompt=vectors[0][0].baseline_prompt,
            norm=torch.norm(combined).item(),
            extraction_date=datetime.now().isoformat()
        )

    @staticmethod
    def _generate_name(concept_prompt: str, baseline_prompt: str) -> str:
        """Generate a concept name from prompts."""
        # Extract key difference
        concept_words = concept_prompt.lower().split()
        baseline_words = set(baseline_prompt.lower().split())

        # Find unique words in concept
        unique_words = [w for w in concept_words if w not in baseline_words]

        if unique_words:
            return "_".join(unique_words[:3])
        else:
            return "concept_" + "_".join(concept_words[:2])

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Convert concept name to filesystem-safe string."""
        # Replace spaces and special characters
        safe = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove other problematic characters
        safe = "".join(c for c in safe if c.isalnum() or c in "_-")
        return safe.lower()


def create_default_concepts(model_wrapper, library: VectorLibrary, layers: list[int] = None) -> None:
    """
    Pre-compute common concept vectors for the library.

    This saves users time and provides good starting examples.

    Args:
        model_wrapper: Loaded model
        library: VectorLibrary to populate
        layers: Which layers to extract from (if None, uses recommended)
    """
    from datetime import datetime

    if layers is None:
        layers = model_wrapper.get_recommended_layers()
        if not layers:
            # Fallback to middle layer
            layers = [model_wrapper.num_layers // 2]

    # Define concept pairs: (name, concept_prompt, baseline_prompt)
    concepts = [
        ("happy", "I feel so happy and joyful today!", "I feel neutral today."),
        ("sad", "I feel very sad and melancholic.", "I feel neutral today."),
        ("angry", "I am extremely angry and furious!", "I feel neutral today."),
        ("fearful", "I am terrified and afraid!", "I feel neutral today."),
        ("formal", "In accordance with established protocols and regulations.", "This is how things work."),
        ("casual", "Hey dude, like, whatever ya know?", "This is how things work."),
        ("pirate", "Arrr matey, 'tis a fine day to sail the seven seas!", "It is a good day."),
        ("shakespeare", "Forsooth, what light through yonder window breaks!", "What is that over there?"),
        ("enthusiastic", "This is AMAZING! I'm SO excited about this incredible opportunity!", "This is interesting."),
        ("brief", "Short.", "This is a sentence that conveys information in a clear manner.")
    ]

    print(f"Extracting {len(concepts)} concepts at {len(layers)} layers...")

    for name, concept_prompt, baseline_prompt in concepts:
        for layer_idx in layers:
            try:
                print(f"  Extracting '{name}' at layer {layer_idx}...")
                vector = library.compute_concept_vector(
                    model_wrapper=model_wrapper,
                    concept_prompt=concept_prompt,
                    baseline_prompt=baseline_prompt,
                    layer_idx=layer_idx,
                    concept_name=name
                )
                library.add_vector(vector)
            except Exception as e:
                print(f"    Error: {e}")

    print(f"✓ Created {len(library.list_concepts())} concepts")
