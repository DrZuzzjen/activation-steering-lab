"""Test activation visualization functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.figure
import plotly.graph_objects as go
import torch

# Ensure package imports resolve when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_steering_lab.visualization import ActivationVisualizer  # noqa: E402


class DummyTokenizer:
    """Minimal tokenizer stub for tests."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt"):
        tokens = self.tokenize(text)
        ids = torch.arange(len(tokens), dtype=torch.long)
        if len(tokens) == 0:
            ids = torch.tensor([0], dtype=torch.long)
        return {
            "input_ids": ids.unsqueeze(0),
            "attention_mask": torch.ones((1, ids.shape[0]), dtype=torch.long),
        }

    def tokenize(self, text: str) -> list[str]:
        tokens = text.strip().split()
        return tokens if tokens else ["<blank>"]

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [f"tok_{int(item)}" for item in ids]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(self.convert_ids_to_tokens(ids))


class DummyModel:
    """Simple model stub that records activations."""

    def __init__(self, wrapper: "DummyModelWrapper") -> None:
        self.wrapper = wrapper
        self.config = SimpleNamespace(hidden_size=4)
        self._call_count = 0

    def generate(self, *, input_ids, attention_mask, max_new_tokens, **_: dict) -> torch.Tensor:
        base_len = input_ids.shape[1]
        seq_len = base_len + max_new_tokens
        if seq_len <= 0:
            seq_len = 1

        token_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        base_value = 1.0 + self._call_count
        if self.wrapper._injection is not None:
            # Boost activation values to simulate steering impact
            base_value += 1.0

        hidden_size = self.config.hidden_size
        for layer_idx in range(self.wrapper.num_layers):
            layer_value = base_value + layer_idx
            tensor = torch.full((1, seq_len, hidden_size), layer_value, dtype=torch.float32)
            self.wrapper.captured_activations[layer_idx] = tensor

        self._call_count += 1
        return token_ids


class DummyLibrary:
    """Stub concept library for tests."""

    def __init__(self) -> None:
        self.vectors: dict[str, dict[int, SimpleNamespace]] = {}

    def add_concept(self, name: str, layer_idx: int, vector: torch.Tensor) -> None:
        self.vectors.setdefault(name, {})[layer_idx] = SimpleNamespace(vector=vector)

    def get_vector(self, name: str, layer_idx: int):
        return self.vectors.get(name, {}).get(layer_idx)

    def get_vector_nearest_layer(self, name: str, layer_idx: int | None):
        layers = self.vectors.get(name)
        if not layers:
            return None
        available = sorted(layers.keys())
        if layer_idx is None or layer_idx in layers:
            target = layer_idx if layer_idx is not None else available[0]
            return layers[target], target
        nearest = min(available, key=lambda idx: abs(idx - layer_idx))
        return layers[nearest], nearest

    def list_concepts(self) -> list[str]:
        return list(self.vectors.keys())


class DummyModelWrapper:
    """Model wrapper stub exposing required interface."""

    def __init__(self) -> None:
        self.model = DummyModel(self)
        self.tokenizer = DummyTokenizer()
        self.device = torch.device("cpu")
        self.num_layers = 3
        self.captured_activations: dict[int, torch.Tensor] = {}
        self._injection: tuple[int, float] | None = None

    def clear_hooks(self) -> None:
        self.captured_activations = {}
        self._injection = None

    def capture_all_layers(self) -> None:
        # No-op: activations are populated directly in DummyModel
        pass

    def register_injection_hook(self, layer_idx: int, vector: torch.Tensor, strength: float = 1.0, position: int = -1) -> None:
        self._injection = (layer_idx, strength, position, vector)


def build_visualizer() -> tuple[ActivationVisualizer, DummyModelWrapper, DummyLibrary]:
    wrapper = DummyModelWrapper()
    library = DummyLibrary()
    visualizer = ActivationVisualizer(wrapper, library)
    return visualizer, wrapper, library


def _simple_activation_dict(num_layers: int, seq_len: int, hidden_size: int, base: float) -> dict[int, torch.Tensor]:
    activations: dict[int, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        activations[layer_idx] = torch.full((1, seq_len, hidden_size), base + layer_idx, dtype=torch.float32)
    return activations


def test_visualizer_initialization() -> None:
    visualizer, wrapper, library = build_visualizer()
    assert visualizer.model is wrapper
    assert visualizer.library is library


def test_capture_activations() -> None:
    visualizer, wrapper, library = build_visualizer()
    vector = torch.ones(wrapper.model.config.hidden_size, dtype=torch.float32)
    library.add_concept("happy", 1, vector)

    normal_text, steered_text, normal_acts, steered_acts, tokens = visualizer.capture_activations_for_comparison(
        prompt="Hello world",
        concept_name="happy",
        layer_idx=1,
        strength=2.0,
        max_new_tokens=3,
        temperature=0.5,
    )

    assert isinstance(normal_text, str)
    assert isinstance(steered_text, str)
    assert set(normal_acts.keys()) == set(range(wrapper.num_layers))
    assert set(steered_acts.keys()) == set(range(wrapper.num_layers))
    for tensor in normal_acts.values():
        assert tensor.shape[0] == 1
    for tensor in steered_acts.values():
        assert tensor.shape[0] == 1
    prompt_tokens = len(wrapper.tokenizer.tokenize("Hello world"))
    expected_length = prompt_tokens + 3
    assert normal_acts[1].shape[1] == expected_length
    assert steered_acts[1].shape[1] == expected_length
    assert len(tokens) > 0


def test_heatmap_creation() -> None:
    visualizer, wrapper, _ = build_visualizer()
    seq_len = 4
    hidden = wrapper.model.config.hidden_size
    normal_acts = _simple_activation_dict(wrapper.num_layers, seq_len, hidden, base=1.0)
    steered_acts = _simple_activation_dict(wrapper.num_layers, seq_len, hidden, base=2.0)
    fig = visualizer.create_token_layer_heatmap(normal_acts, steered_acts, injection_layer=1, tokens=["a", "b", "c", "d"], concept_name="happy")
    assert isinstance(fig, matplotlib.figure.Figure)
    # 3 main heatmaps + 3 colorbars = 6 axes (seaborn adds colorbar axes)
    assert len(fig.axes) >= 3


def test_concept_space_creation() -> None:
    visualizer, wrapper, library = build_visualizer()
    hidden = wrapper.model.config.hidden_size
    for idx, name in enumerate(["happy", "sad", "calm"], start=1):
        vector = torch.full((hidden,), float(idx))
        library.add_concept(name, 1, vector)
    library.add_concept("excited", 2, torch.ones(hidden) * 4)

    seq_len = 3
    normal_acts = {1: torch.ones((1, seq_len, hidden))}
    steered_acts = {1: torch.ones((1, seq_len, hidden)) * 2}

    fig = visualizer.create_concept_space_2d(normal_acts, steered_acts, layer_idx=1, concept_name="happy")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3
    concept_trace = fig.data[0]
    assert any("excited" in label for label in concept_trace.text)


def test_with_insufficient_concepts() -> None:
    visualizer, wrapper, library = build_visualizer()
    hidden = wrapper.model.config.hidden_size
    library.add_concept("happy", 1, torch.ones(hidden))
    library.add_concept("sad", 1, torch.ones(hidden) * -1)

    seq_len = 2
    normal_acts = {1: torch.ones((1, seq_len, hidden))}
    steered_acts = {1: torch.ones((1, seq_len, hidden))}

    fig = visualizer.create_concept_space_2d(normal_acts, steered_acts, layer_idx=1, concept_name="happy")
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) > 0


def test_clean_token_replaces_artifacts() -> None:
    noisy_tokens = ["▁Hello", "Ġworld", "token▁with▁spaces", "Ċ"]
    cleaned = [ActivationVisualizer._clean_token(tok) for tok in noisy_tokens]
    assert cleaned[0] == "Hello"
    assert cleaned[1] == "world"
    assert "with" in cleaned[2] and "spaces" in cleaned[2]
    # Empty after stripping returns the fallback
    assert cleaned[3] == "▁"
