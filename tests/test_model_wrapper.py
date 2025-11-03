"""Tests for ModelWrapper activation capture behavior."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from activation_steering_lab.model_wrapper import ModelWrapper


class PassthroughLayer(nn.Module):
    """Layer that returns its input, used to exercise hooks."""

    def forward(self, hidden_states):  # type: ignore[override]
        return hidden_states


def _build_wrapper_with_layers(num_layers: int = 1, hidden_size: int = 4) -> tuple[ModelWrapper, list[PassthroughLayer]]:
    wrapper = ModelWrapper()
    layers = [PassthroughLayer() for _ in range(num_layers)]
    wrapper.model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    wrapper.tokenizer = None
    wrapper.device = torch.device("cpu")
    wrapper.num_layers = num_layers
    return wrapper, layers


def test_capture_hooks_accumulate_across_forward_passes() -> None:
    wrapper, layers = _build_wrapper_with_layers()
    wrapper.register_capture_hook(0)

    first_chunk = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    second_chunk = torch.full((1, 2, 4), 10.0)

    _ = layers[0](first_chunk)
    _ = layers[0](second_chunk)

    captured = wrapper.captured_activations[0]
    expected = torch.cat([first_chunk.cpu(), second_chunk.cpu()], dim=1)
    assert torch.allclose(captured, expected)

    wrapper.clear_hooks()


def test_clear_hooks_resets_buffers() -> None:
    wrapper, layers = _build_wrapper_with_layers()
    wrapper.register_capture_hook(0)
    _ = layers[0](torch.ones((1, 1, 4)))

    assert 0 in wrapper.captured_activations
    wrapper.clear_hooks()
    assert wrapper.captured_activations == {}
    assert wrapper._activation_buffers == {}