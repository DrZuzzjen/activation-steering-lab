"""
Injection Engine for Activation Steering
Handles the logic of steering model behavior through activation injection.
"""

import torch
from typing import List, Dict, Optional, Tuple
from .vector_library import ConceptVector, VectorLibrary
from .model_wrapper import ModelWrapper


class InjectionEngine:
    """
    Manages activation injection during text generation.

    This is the core of the steering mechanism - it coordinates:
    1. Which vectors to inject
    2. At which layers
    3. With what strength
    4. How to combine multiple injections
    """

    def __init__(self, model_wrapper: ModelWrapper, vector_library: VectorLibrary):
        """
        Initialize the injection engine.

        Args:
            model_wrapper: Loaded model wrapper
            vector_library: Library of concept vectors
        """
        self.model = model_wrapper
        self.library = vector_library

    def generate_with_steering(
        self,
        prompt: str,
        concept_name: str,
        layer_idx: int,
        strength: float = 2.0,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        position: int = -1
    ) -> str:
        """
        Generate text with activation steering applied.

        This is the main function users will interact with.

        Args:
            prompt: Input text
            concept_name: Which concept to inject
            layer_idx: Which layer to inject at
            strength: Injection strength multiplier
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            position: Token position for injection (-1 = last token)

        Returns:
            Generated text
        """
        # Get the concept vector
        concept_vector = self.library.get_vector(concept_name, layer_idx)

        if concept_vector is None:
            raise ValueError(
                f"Concept '{concept_name}' not found at layer {layer_idx}. "
                f"Available concepts: {self.library.list_concepts()}"
            )

        # Clear any existing hooks
        self.model.clear_hooks()

        # Register injection hook
        self.model.register_injection_hook(
            layer_idx=layer_idx,
            vector=concept_vector.vector,
            strength=strength,
            position=position
        )

        # Generate text
        try:
            output = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        finally:
            # Always clean up hooks, even if generation fails
            self.model.clear_hooks()

        return output

    def generate_comparison(
        self,
        prompt: str,
        concept_name: str,
        layer_idx: int,
        strength: float = 2.0,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, str]:
        """
        Generate both normal and steered outputs for comparison.

        This is crucial for the educational experience - seeing the difference!

        Returns:
            Dictionary with 'normal' and 'steered' outputs
        """
        # Generate normal output (no steering)
        self.model.clear_hooks()
        normal_output = self.model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # Generate steered output
        steered_output = self.generate_with_steering(
            prompt=prompt,
            concept_name=concept_name,
            layer_idx=layer_idx,
            strength=strength,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        return {
            'normal': normal_output,
            'steered': steered_output
        }

    def generate_with_multiple_concepts(
        self,
        prompt: str,
        concept_injections: List[Tuple[str, int, float]],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Inject multiple concepts at different layers simultaneously.

        This enables complex steering scenarios:
        - Inject "formal" at layer 10, "happy" at layer 15
        - Mix multiple emotions
        - Layer-specific concept fighting

        Args:
            prompt: Input text
            concept_injections: List of (concept_name, layer_idx, strength) tuples
            max_new_tokens: Tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Generated text with all concepts applied
        """
        # Clear existing hooks
        self.model.clear_hooks()

        # Register all injection hooks
        for concept_name, layer_idx, strength in concept_injections:
            concept_vector = self.library.get_vector(concept_name, layer_idx)

            if concept_vector is None:
                print(f"Warning: Skipping {concept_name} at layer {layer_idx} (not found)")
                continue

            self.model.register_injection_hook(
                layer_idx=layer_idx,
                vector=concept_vector.vector,
                strength=strength,
                position=-1
            )

        # Generate
        try:
            output = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        finally:
            self.model.clear_hooks()

        return output

    def analyze_layer_effects(
        self,
        prompt: str,
        concept_name: str,
        strength: float = 2.0,
        max_new_tokens: int = 30,
        layer_step: int = 5
    ) -> Dict[int, str]:
        """
        Test the same concept across multiple layers.

        Educational function to show:
        - Where concepts have the most effect
        - How layer choice matters
        - Why middle layers work best

        Args:
            prompt: Input text
            concept_name: Concept to test
            strength: Injection strength
            max_new_tokens: Tokens to generate
            layer_step: Test every N layers

        Returns:
            Dictionary mapping layer_idx -> generated_text
        """
        results = {}

        # Get recommended layer for this concept
        available_layers = self.library.get_layers_for_concept(concept_name)

        if not available_layers:
            raise ValueError(f"Concept '{concept_name}' not found in library")

        # Use the first available layer as reference
        reference_layer = available_layers[0]
        concept_vector = self.library.get_vector(concept_name, reference_layer)

        # Test across all model layers
        for layer_idx in range(0, self.model.num_layers, layer_step):
            print(f"Testing layer {layer_idx}/{self.model.num_layers}...")

            self.model.clear_hooks()
            self.model.register_injection_hook(
                layer_idx=layer_idx,
                vector=concept_vector.vector,
                strength=strength,
                position=-1
            )

            try:
                output = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9
                )
                results[layer_idx] = output
            except Exception as e:
                results[layer_idx] = f"Error: {e}"
            finally:
                self.model.clear_hooks()

        return results

    def test_strength_range(
        self,
        prompt: str,
        concept_name: str,
        layer_idx: int,
        strengths: List[float] = None,
        max_new_tokens: int = 30
    ) -> Dict[float, str]:
        """
        Test different injection strengths.

        Shows the relationship between strength and effect magnitude.

        Args:
            prompt: Input text
            concept_name: Concept to test
            layer_idx: Layer to inject at
            strengths: List of strengths to test (default: [0.5, 1.0, 2.0, 5.0])
            max_new_tokens: Tokens to generate

        Returns:
            Dictionary mapping strength -> generated_text
        """
        if strengths is None:
            strengths = [0.5, 1.0, 2.0, 3.0, 5.0]

        results = {}

        for strength in strengths:
            print(f"Testing strength {strength}...")

            output = self.generate_with_steering(
                prompt=prompt,
                concept_name=concept_name,
                layer_idx=layer_idx,
                strength=strength,
                max_new_tokens=max_new_tokens
            )

            results[strength] = output

        return results

    def create_mixed_emotion(
        self,
        emotion_weights: Dict[str, float],
        layer_idx: int,
        name: str = "mixed_emotion"
    ) -> ConceptVector:
        """
        Create a new concept by mixing multiple emotions.

        Example usage:
            mixed = engine.create_mixed_emotion(
                {'happy': 0.7, 'excited': 0.3},
                layer_idx=15,
                name='cheerful'
            )

        Args:
            emotion_weights: Dictionary of {emotion_name: weight}
            layer_idx: Layer to mix at
            name: Name for the new concept

        Returns:
            New combined ConceptVector
        """
        # Get vectors for each emotion
        vectors_with_weights = []

        for emotion_name, weight in emotion_weights.items():
            vector = self.library.get_vector(emotion_name, layer_idx)

            if vector is None:
                raise ValueError(f"Emotion '{emotion_name}' not found at layer {layer_idx}")

            vectors_with_weights.append((vector, weight))

        # Combine using library method
        mixed_vector = self.library.combine_vectors(
            vectors=vectors_with_weights,
            new_name=name
        )

        # Add to library
        self.library.add_vector(mixed_vector)

        return mixed_vector

    def demonstrate_concept_fighting(
        self,
        prompt: str,
        concept1: Tuple[str, int, float],  # (name, layer, strength)
        concept2: Tuple[str, int, float],
        max_new_tokens: int = 50
    ) -> str:
        """
        Demonstrate what happens when opposing concepts are injected.

        Educational example of layer interaction.

        Args:
            prompt: Input text
            concept1: First concept (name, layer, strength)
            concept2: Second concept (name, layer, strength)
            max_new_tokens: Tokens to generate

        Returns:
            Generated text with both concepts fighting
        """
        name1, layer1, strength1 = concept1
        name2, layer2, strength2 = concept2

        print(f"Injecting '{name1}' at layer {layer1} (strength {strength1})")
        print(f"Injecting '{name2}' at layer {layer2} (strength {strength2})")

        return self.generate_with_multiple_concepts(
            prompt=prompt,
            concept_injections=[(name1, layer1, strength1), (name2, layer2, strength2)],
            max_new_tokens=max_new_tokens
        )

    def get_injection_summary(
        self,
        concept_name: str,
        layer_idx: int,
        strength: float
    ) -> Dict:
        """
        Get a summary of what will be injected.

        Useful for UI display and educational purposes.
        """
        vector = self.library.get_vector(concept_name, layer_idx)

        if vector is None:
            return {
                'error': f"Concept '{concept_name}' not found at layer {layer_idx}"
            }

        stats = self.library.get_vector_stats(concept_name, layer_idx)

        return {
            'concept': concept_name,
            'layer': layer_idx,
            'strength': strength,
            'effective_norm': stats['norm'] * strength,
            'vector_stats': stats,
            'description': self.model.get_layer_description(layer_idx)
        }
