"""Visualization utilities for activation steering outcomes."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

from activation_steering_lab.model_wrapper import ModelWrapper
from activation_steering_lab.vector_library import VectorLibrary


class ActivationVisualizer:
    """Handles activation capture and visualization helpers."""

    def __init__(self, model_wrapper: ModelWrapper, vector_library: VectorLibrary) -> None:
        self.model = model_wrapper
        self.library = vector_library

    def capture_activations_for_comparison(
        self,
        prompt: str,
        concept_name: str,
        layer_idx: int,
        strength: float,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> Tuple[str, str, Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[str]]:
        """Capture activations for normal vs. steered generations."""
        if self.model.model is None or self.model.tokenizer is None:
            raise ValueError("Model is not initialized. Initialize the model before capturing activations.")

        concept = self.library.get_vector(concept_name, int(layer_idx))
        if concept is None:
            raise ValueError(f"Concept '{concept_name}' not found at layer {layer_idx}.")

        layer_idx = int(layer_idx)
        strength = float(strength)
        max_new_tokens = int(max_new_tokens)
        temperature = float(temperature)

        # Normal generation
        self.model.clear_hooks()
        self.model.capture_all_layers()
        normal_text, normal_tokens, normal_acts = self._run_generation(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )

        # Steered generation
        self.model.clear_hooks()
        self.model.capture_all_layers()
        self.model.register_injection_hook(layer_idx, concept.vector, strength=strength)
        steered_text, steered_tokens, steered_acts = self._run_generation(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )

        self.model.clear_hooks()

        # Use tokens from normal run to keep axis consistent
        tokens = self._reconcile_tokens(normal_tokens, steered_tokens)

        return normal_text, steered_text, normal_acts, steered_acts, tokens

    def create_token_layer_heatmap(
        self,
        normal_acts: Dict[int, torch.Tensor],
        steered_acts: Dict[int, torch.Tensor],
        injection_layer: int,
        tokens: List[str],
        concept_name: str,
    ) -> go.Figure:
        """Create fMRI-style brain scan visualization showing activation hotspots across layers."""
        if not normal_acts or not steered_acts:
            fig = go.Figure()
            fig.add_annotation(
                text="No activation data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(width=1200, height=600)
            return fig

        layer_indices = sorted(set(normal_acts.keys()) & set(steered_acts.keys()))
        if not layer_indices:
            fig = go.Figure()
            fig.add_annotation(
                text="No overlapping layer activations",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(width=1200, height=600)
            return fig

        seq_len = self._determine_sequence_length(normal_acts, steered_acts)
        token_labels = self._prepare_token_labels(tokens, seq_len)
        token_idx = seq_len - 1  # Last token

        # Downsample hidden dimensions into regions (like brain regions)
        num_regions = 64  # Phi-3: 3072 dims → 64 regions (~48 dims per region)
        hidden_size = normal_acts[layer_indices[0]].shape[-1]
        region_size = max(1, hidden_size // num_regions)

        # Build 2D heatmap matrix: [layers × regions]
        heatmap_matrix = []
        layer_vectors_full = []

        for layer_idx in layer_indices:
            normal_vec = normal_acts[layer_idx][0, token_idx, :].detach().cpu().numpy()
            steered_vec = steered_acts[layer_idx][0, token_idx, :].detach().cpu().numpy()
            diff_vec = steered_vec - normal_vec
            diff_vec = np.nan_to_num(diff_vec, nan=0.0, posinf=0.0, neginf=0.0)

            # Downsample into regions by averaging
            regions = []
            for r in range(num_regions):
                start = r * region_size
                end = min(start + region_size, hidden_size)
                if start < hidden_size:
                    region_activation = np.mean(np.abs(diff_vec[start:end]))
                    regions.append(region_activation)
                else:
                    regions.append(0.0)

            heatmap_matrix.append(regions)
            layer_vectors_full.append(diff_vec)

        heatmap_matrix = np.array(heatmap_matrix)  # Shape: [num_layers, num_regions]

        # Calculate layer correlations for annotations
        injection_idx = layer_indices.index(injection_layer) if injection_layer in layer_indices else None
        correlations = {}
        if injection_idx is not None:
            injection_vector = layer_vectors_full[injection_idx]
            for idx, layer_idx in enumerate(layer_indices):
                if idx == injection_idx:
                    correlations[layer_idx] = 1.0
                else:
                    corr = np.corrcoef(injection_vector, layer_vectors_full[idx])[0, 1]
                    corr = np.nan_to_num(corr, nan=0.0)
                    correlations[layer_idx] = float(corr)

        top_layers = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        top_layers = [(l, c) for l, c in top_layers if l != injection_layer][:3]

        # Robust percentile scaling
        vmin = np.percentile(heatmap_matrix, 5)
        vmax = np.percentile(heatmap_matrix, 95)

        # Create 2D heatmap (fMRI brain scan style)
        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=heatmap_matrix.T,  # Transpose: regions on Y-axis, layers on X-axis
                x=[f"L{i}" for i in layer_indices],
                y=[f"R{i}" for i in range(num_regions)],
                colorscale=[
                    [0.0, "rgb(60, 60, 60)"],    # Dark gray (like brain tissue)
                    [0.3, "rgb(100, 0, 0)"],     # Dark red
                    [0.5, "rgb(200, 0, 0)"],     # Red
                    [0.7, "rgb(255, 100, 0)"],   # Orange
                    [0.85, "rgb(255, 200, 0)"],  # Yellow
                    [1.0, "rgb(255, 255, 255)"], # White hotspot
                ],
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(
                    title=dict(text="Activation<br>Intensity", side="right"),
                    tickfont=dict(color="white"),
                ),
                hovertemplate="Layer: %{x}<br>Region: %{y}<br>Intensity: %{z:.2f}<extra></extra>",
                showscale=True,
            )
        )

        # Add injection layer marker
        if injection_idx is not None:
            fig.add_vline(
                x=injection_idx,
                line_color="cyan",
                line_width=3,
                line_dash="dash",
                annotation=dict(
                    text="⚡",
                    font=dict(size=24, color="cyan"),
                    showarrow=False,
                    y=1.02,
                    yref="paper",
                ),
            )

        # Add correlation annotations for top affected layers
        if top_layers and injection_idx is not None:
            for layer_idx, corr in top_layers:
                if layer_idx in layer_indices:
                    target_idx = layer_indices.index(layer_idx)
                    # Add annotation arrow
                    fig.add_annotation(
                        x=target_idx,
                        y=num_regions * 0.9,
                        ax=injection_idx,
                        ay=num_regions * 0.9,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowwidth=2,
                        arrowcolor="cyan",
                        text=f"r={corr:.2f}",
                        font=dict(size=10, color="cyan"),
                        bgcolor="rgba(0, 0, 0, 0.7)",
                    )

        # Calculate summary statistics
        total_activation = float(np.sum(np.abs(heatmap_matrix)))
        most_active_layer_idx = np.argmax(np.sum(np.abs(heatmap_matrix), axis=1))
        most_active_layer = layer_indices[most_active_layer_idx]

        # Add metrics box (yellow, fMRI style)
        metrics_lines = [
            f"<b>Token:</b> '{token_labels[token_idx] if token_idx < len(token_labels) else token_idx}'",
            f"<b>Injection:</b> L{injection_layer}",
            "",
            "<b>Correlation with Injection:</b>",
        ]
        for layer_idx, corr in top_layers:
            metrics_lines.append(f"  L{layer_idx}: r = {corr:.3f}")
        metrics_lines.append("")
        metrics_lines.append(f"<b>Peak Activity:</b> L{most_active_layer}")
        metrics_lines.append(f"<b>Total Intensity:</b> {total_activation:.1f}")

        fig.add_annotation(
            text="<br>".join(metrics_lines),
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="black", family="Arial"),
            align="left",
            bgcolor="rgba(255, 255, 100, 0.9)",
            bordercolor="black",
            borderwidth=2,
            xanchor="left",
            yanchor="top",
        )

        fig.update_layout(
            title=dict(
                text=f"Transformer Activation Map: {concept_name}",
                font=dict(size=18, color="white"),
            ),
            xaxis=dict(
                title="Layer",
                side="bottom",
                tickfont=dict(color="white"),
                gridcolor="rgba(100, 100, 100, 0.2)",
            ),
            yaxis=dict(
                title="Neural Region",
                tickfont=dict(color="white"),
                showticklabels=False,  # Hide region labels (too many)
                gridcolor="rgba(100, 100, 100, 0.2)",
            ),
            width=1400,
            height=700,
            plot_bgcolor="rgb(20, 20, 20)",  # Very dark background (like MRI scan)
            paper_bgcolor="rgb(10, 10, 10)",
            font=dict(color="white"),
            hovermode="closest",
        )

        return fig

    def _run_generation(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[str, List[str], Dict[int, torch.Tensor]]:
        """Generate text while capture hooks are active."""
        tokenizer = self.model.tokenizer
        model = self.model.model

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        self.model.captured_activations = {}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=pad_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        raw_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
        tokens = [self._clean_token(token) for token in raw_tokens]

        seq_len = output_ids.shape[1]
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None and self.model.captured_activations:
            first_tensor = next(iter(self.model.captured_activations.values()))
            hidden_size = first_tensor.shape[-1]
        if hidden_size is None:
            hidden_size = inputs["input_ids"].shape[-1]

        activations: Dict[int, torch.Tensor] = {}
        for idx in range(self.model.num_layers):
            tensor = self.model.captured_activations.get(idx)
            if tensor is None:
                activations[idx] = torch.zeros((1, seq_len, hidden_size), dtype=torch.float32)
            else:
                activations[idx] = tensor.detach().cpu()

        return generated_text, tokens, activations

    @staticmethod
    def _prepare_token_labels(tokens: List[str], seq_len: int) -> List[str]:
        labels = tokens[:seq_len]
        if len(labels) < seq_len:
            labels = labels + [""] * (seq_len - len(labels))
        if len(labels) > 30:
            labels = [label if idx % 2 == 0 else "" for idx, label in enumerate(labels)]
        return labels

    @staticmethod
    def _determine_sequence_length(
        normal_acts: Dict[int, torch.Tensor],
        steered_acts: Dict[int, torch.Tensor],
    ) -> int:
        normal_lengths = [tensor.shape[1] for tensor in normal_acts.values()]
        steered_lengths = [tensor.shape[1] for tensor in steered_acts.values()]
        if not normal_lengths or not steered_lengths:
            return 0
        return min(min(normal_lengths), min(steered_lengths))

    @staticmethod
    def _clean_token(token: str) -> str:
        cleaned = token.replace("▁", " ").replace("Ġ", " ")
        cleaned = cleaned.replace("Ċ", "\n").replace("Ď", "")
        cleaned = cleaned.strip()
        return cleaned or "▁"

    @staticmethod
    def _reconcile_tokens(normal_tokens: List[str], steered_tokens: List[str]) -> List[str]:
        if len(normal_tokens) == len(steered_tokens):
            return normal_tokens
        min_len = min(len(normal_tokens), len(steered_tokens))
        return normal_tokens[:min_len]
