"""Performance smoke test for visualization helpers."""

from __future__ import annotations

import time
import tracemalloc

import torch
from matplotlib.figure import Figure as MatplotlibFigure
import plotly.graph_objects as go

from tests.test_visualization import build_visualizer


def test_visualization_performance() -> None:
    visualizer, wrapper, library = build_visualizer()

    hidden = wrapper.model.config.hidden_size
    for idx, name in enumerate(["happy", "sad", "calm"], start=1):
        library.add_concept(name, 1, torch.ones(hidden) * float(idx))

    tracemalloc.start()
    start = time.time()

    normal_text, steered_text, normal_acts, steered_acts, tokens = visualizer.capture_activations_for_comparison(
        prompt="Performance test",
        concept_name="happy",
        layer_idx=1,
        strength=1.5,
        max_new_tokens=5,
        temperature=0.7,
    )

    heatmap_fig = visualizer.create_token_layer_heatmap(
        normal_acts=normal_acts,
        steered_acts=steered_acts,
        injection_layer=1,
        tokens=tokens,
        concept_name="happy",
    )
    concept_fig = visualizer.create_concept_space_2d(
        normal_acts=normal_acts,
        steered_acts=steered_acts,
        layer_idx=1,
        concept_name="happy",
    )

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Basic sanity checks to ensure objects were created
    assert isinstance(heatmap_fig, MatplotlibFigure)
    assert isinstance(concept_fig, go.Figure)

    peak_mb = peak / (1024 * 1024)
    assert elapsed < 3.0, f"Visualization took too long: {elapsed:.2f}s"
    assert peak_mb < 20.0, f"Visualization used too much memory: {peak_mb:.2f}MB"
