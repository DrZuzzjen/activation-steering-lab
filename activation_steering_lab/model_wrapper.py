"""
Model Wrapper for Activation Steering
This module handles model loading, hook registration, and memory management for Apple Silicon.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Callable, Optional, Tuple
import gc
import os
from pathlib import Path


class ModelWrapper:
    """
    Wraps a transformer model with activation capture and injection capabilities.

    Key concepts:
    - Hooks: PyTorch functions that intercept forward passes at specific layers
    - Activations: Internal neural network values at each layer
    - MPS: Metal Performance Shaders - Apple's GPU acceleration
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "auto", cache_dir: str = None):
        """
        Initialize the model wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('mps' for Apple Silicon, 'cpu' for fallback)
            cache_dir: Directory to cache models (defaults to local project cache)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hooks = []  # Keep track of registered hooks for cleanup
        self.num_layers = 0

        # Set up cache directory (use local project cache by default)
        if cache_dir is None:
            # Use local cache directory within the project
            project_root = Path(__file__).parent.parent
            self.cache_dir = str(project_root / "activation_steering_lab" / "models_cache")
        else:
            self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Storage for captured activations
        self.captured_activations: Dict[int, torch.Tensor] = {}

    def load_model(self):
        """
        Load the model with appropriate settings for M4 chip.

        Why float16?
        - Reduces memory usage by 50% (16-bit vs 32-bit floats)
        - Faster computation on Apple Silicon
        - Minimal accuracy loss for our educational purposes
        """
        print(f"Loading {self.model_name}...")

        try:
            # Load tokenizer with local cache
            print(f"  Cache directory: {self.cache_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=False  # Allow download if not cached
            )

            # Ensure tokenizer has a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with float16 for memory efficiency
            print(f"  Loading model from cache (or downloading if first time)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir,
                local_files_only=False  # Allow download if not cached
            )

            # Set to evaluation mode (disables dropout, etc.)
            self.model.eval()

            # Get actual device where model was loaded
            # When device_map="auto", we need to find where the model actually is
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                # Get device from first parameter
                self.device = next(self.model.parameters()).device

            # Determine number of layers
            # Most models store layers in model.layers or model.transformer.h
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                self.num_layers = len(self.model.model.layers)
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                self.num_layers = len(self.model.transformer.h)
            else:
                # Fallback: try to find layers
                for attr in dir(self.model):
                    obj = getattr(self.model, attr)
                    if hasattr(obj, 'layers'):
                        self.num_layers = len(obj.layers)
                        break

            print(f"✓ Model loaded: {self.num_layers} layers")
            print(f"✓ Device: {self.device}")
            print(f"✓ Memory footprint: ~{self._estimate_memory_mb():.0f}MB")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                cache_dir=self.cache_dir
            )
            self.model.eval()

            # Set device to cpu
            self.device = torch.device("cpu")

    def _estimate_memory_mb(self) -> float:
        """Estimate model memory usage in MB."""
        if self.model is None:
            return 0

        param_count = sum(p.numel() for p in self.model.parameters())
        # float16 = 2 bytes per parameter
        return (param_count * 2) / (1024 * 1024)

    def get_layer_module(self, layer_idx: int):
        """
        Get the actual PyTorch module for a specific layer.

        This is needed because different models organize layers differently:
        - Phi models: model.model.layers[i]
        - GPT models: model.transformer.h[i]
        - Llama models: model.model.layers[i]
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Could not find layer {layer_idx} in model architecture")

    def register_capture_hook(self, layer_idx: int) -> None:
        """
        Register a hook to capture activations at a specific layer.

        How hooks work:
        - Forward hooks intercept data flowing through the network
        - They receive: (module, input, output)
        - We store the output for later analysis

        Args:
            layer_idx: Which layer to capture from (0 to num_layers-1)
        """
        layer_module = self.get_layer_module(layer_idx)

        def capture_hook(module, input, output):
            """
            This function runs during the forward pass.

            Why output[0]?
            - Transformer layers return tuples: (hidden_states, attention_weights, ...)
            - We want the hidden_states (first element)
            """
            if isinstance(output, tuple):
                self.captured_activations[layer_idx] = output[0].detach().clone()
            else:
                self.captured_activations[layer_idx] = output.detach().clone()

        hook = layer_module.register_forward_hook(capture_hook)
        self.hooks.append(hook)

    def register_injection_hook(self, layer_idx: int, vector: torch.Tensor,
                               strength: float = 1.0, position: int = -1) -> None:
        """
        Register a hook to inject an activation vector during generation.

        This is where the "steering" happens!

        Args:
            layer_idx: Which layer to inject into
            vector: The concept vector to inject (shape: [hidden_dim])
            strength: Multiplier for injection strength (1.0 = use vector as-is)
            position: Token position to inject at (-1 = last token, where generation happens)
        """
        layer_module = self.get_layer_module(layer_idx)

        def injection_hook(module, input, output):
            """
            Modify the activation by adding our concept vector.

            Why addition instead of replacement?
            - Addition preserves the original meaning
            - Replacement would destroy context
            - Think of it as "nudging" the model's thinking

            Output shape: [batch_size, sequence_length, hidden_dim]
            Vector shape: [hidden_dim]
            """
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Clone to avoid modifying the original
                modified = hidden_states.clone()

                # Inject at the specified position
                # Position -1 means the last token (where next-token prediction happens)
                modified[:, position, :] += strength * vector.to(modified.device)

                # Return modified output in the same format
                return (modified,) + output[1:]
            else:
                modified = output.clone()
                modified[:, position, :] += strength * vector.to(modified.device)
                return modified

        hook = layer_module.register_forward_hook(injection_hook)
        self.hooks.append(hook)

    def clear_hooks(self) -> None:
        """
        Remove all registered hooks.

        CRITICAL: Always call this after generation!
        Hooks persist across forward passes and can cause:
        - Memory leaks
        - Unexpected behavior in subsequent generations
        - GPU memory accumulation
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.captured_activations = {}

    def generate(self, prompt: str, max_new_tokens: int = 50,
                temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text with current hooks active.

        Args:
            prompt: Input text
            max_new_tokens: How many tokens to generate
            temperature: Randomness (higher = more creative)
            top_p: Nucleus sampling threshold

        Returns:
            Generated text (prompt + completion)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with no gradient computation (saves memory)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def forward_pass(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Run a forward pass and return captured activations.

        This is used for extracting concept vectors.
        Capture hooks must be registered before calling this!

        Returns:
            Dictionary mapping layer_idx -> activation tensor
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Clear previous captures
        self.captured_activations = {}

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (hooks will capture activations)
        with torch.no_grad():
            _ = self.model(**inputs)

        return self.captured_activations.copy()

    def cleanup_memory(self) -> None:
        """
        Clean up GPU/MPS memory.

        Apple Silicon memory management:
        - MPS doesn't have cuda.empty_cache()
        - We rely on Python's garbage collector
        - Explicitly delete tensors and run gc.collect()
        """
        self.clear_hooks()
        self.captured_activations = {}
        gc.collect()

        # For MPS, there's no explicit cache clearing
        # The Metal framework manages memory automatically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_recommended_layers(self) -> List[int]:
        """
        Get recommended layers for injection based on model depth.

        General guidance:
        - Early layers (0-25%): Token embeddings, syntax
        - Mid layers (25-75%): Concepts, semantics ← BEST FOR STEERING
        - Late layers (75-100%): Output formatting, token prediction

        Returns:
            List of recommended layer indices
        """
        if self.num_layers == 0:
            return []

        # Target the middle 50% of layers
        start = int(self.num_layers * 0.25)
        end = int(self.num_layers * 0.75)

        # Return 3-5 recommended layers
        step = max(1, (end - start) // 4)
        return list(range(start, end, step))[:5]

    def get_layer_description(self, layer_idx: int) -> str:
        """
        Get a human-readable description of what happens at this layer.

        Educational function to help users understand layer roles.
        """
        if self.num_layers == 0:
            return "Model not loaded"

        position = layer_idx / self.num_layers

        if position < 0.25:
            return "🔤 Token & Syntax Processing\nEarly layers handle basic token meanings and grammatical structure."
        elif position < 0.5:
            return "💡 Concept Formation\nMiddle-early layers start forming higher-level concepts and relationships."
        elif position < 0.75:
            return "🎯 Abstract Reasoning (Recommended)\nMiddle-late layers handle abstract concepts - ideal for steering!"
        else:
            return "📝 Output Decision Making\nLate layers focus on choosing specific output tokens."
