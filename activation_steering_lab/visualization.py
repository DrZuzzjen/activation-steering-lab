"""Visualization utilities for activation steering outcomes."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
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
    ) -> plt.Figure:
        """Create side-by-side heatmaps of activation magnitudes."""
        if not normal_acts or not steered_acts:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No activation data available", ha="center", va="center", fontsize=14)
            ax.axis("off")
            plt.tight_layout()
            return fig

        layer_indices = sorted(set(normal_acts.keys()) & set(steered_acts.keys()))
        if not layer_indices:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No overlapping layer activations", ha="center", va="center", fontsize=14)
            ax.axis("off")
            plt.tight_layout()
            return fig

        seq_len = self._determine_sequence_length(normal_acts, steered_acts)
        token_labels = self._prepare_token_labels(tokens, seq_len)

        normal_matrix = self._activation_matrix(normal_acts, layer_indices, seq_len)
        steered_matrix = self._activation_matrix(steered_acts, layer_indices, seq_len)
        diff_matrix = steered_matrix - normal_matrix

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        inj_idx = layer_indices.index(injection_layer) if injection_layer in layer_indices else None

        self._plot_heatmap(
            axes[0],
            normal_matrix,
            token_labels,
            layer_indices,
            title="Normal Activations",
            cmap="YlOrRd",
            line_idx=inj_idx,
            line_color="cyan",
        )

        title = f"Steered Activations (+{concept_name})"
        self._plot_heatmap(
            axes[1],
            steered_matrix,
            token_labels,
            layer_indices,
            title=title,
            cmap="YlOrRd",
            line_idx=inj_idx,
            line_color="cyan",
            annotation="← Injection" if inj_idx is not None else None,
        )

        max_abs = np.max(np.abs(diff_matrix)) or 1e-6
        self._plot_heatmap(
            axes[2],
            diff_matrix,
            token_labels,
            layer_indices,
            title="Difference (Steered − Normal)",
            cmap="RdBu_r",
            line_idx=inj_idx,
            line_color="yellow",
            vmin=-max_abs,
            vmax=max_abs,
            center=0.0,
        )

        plt.tight_layout()
        return fig

    def create_concept_space_2d(
        self,
        normal_acts: Dict[int, torch.Tensor],
        steered_acts: Dict[int, torch.Tensor],
        layer_idx: int,
        concept_name: str,
    ) -> go.Figure:
        """Create a PCA-based concept space visualization."""
        layer_idx = int(layer_idx)

        concept_vectors: List[np.ndarray] = []
        concept_names: List[str] = []
        concept_labels: List[str] = []
        concept_layers: List[int] = []
        for name in self.library.list_concepts():
            result = self.library.get_vector_nearest_layer(name, layer_idx)
            if result is None:
                continue
            vector, actual_layer = result
            concept_vectors.append(vector.vector.detach().cpu().numpy())
            concept_names.append(name)
            concept_layers.append(actual_layer)
            if actual_layer == layer_idx:
                concept_labels.append(name)
            else:
                concept_labels.append(f"{name} (L{actual_layer})")

        if len(concept_vectors) < 3 or layer_idx not in normal_acts or layer_idx not in steered_acts:
            fig = go.Figure()
            fig.add_annotation(
                text="Add at least 3 concepts for this layer to view the concept space.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(
                title=f"Concept Space Unavailable (Layer {layer_idx})",
                width=800,
                height=600,
                plot_bgcolor="rgba(240,240,240,0.5)",
            )
            return fig

        normal_vec = normal_acts[layer_idx][0, -1, :].detach().cpu().numpy()
        steered_vec = steered_acts[layer_idx][0, -1, :].detach().cpu().numpy()

        all_vectors = concept_vectors + [normal_vec, steered_vec]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(np.stack(all_vectors))

        concept_coords = coords[: len(concept_vectors)]
        normal_coord = coords[-2]
        steered_coord = coords[-1]

        fig = go.Figure()
        marker_colors = ["red" if layer == layer_idx else "#ff8c8c" for layer in concept_layers]
        hover_text = [
            f"{label}<br>Stored Layer: L{stored}" if stored != layer_idx else f"{label}<br>Layer: L{stored}"
            for label, stored in zip(concept_labels, concept_layers)
        ]
        fig.add_trace(
            go.Scatter(
                x=concept_coords[:, 0],
                y=concept_coords[:, 1],
                mode="markers+text",
                name="Concepts",
                text=concept_labels,
                textposition="top center",
                marker=dict(symbol="diamond", color=marker_colors, size=12),
                hovertext=hover_text,
            )
        )

        if concept_name in concept_names:
            idx = concept_names.index(concept_name)
            fig.add_trace(
                go.Scatter(
                    x=[concept_coords[idx, 0]],
                    y=[concept_coords[idx, 1]],
                    mode="markers+text",
                    name="Target Concept",
                    text=[concept_name],
                    textposition="top center",
                    marker=dict(symbol="star", color="gold", size=18, line=dict(color="black", width=1)),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[normal_coord[0]],
                y=[normal_coord[1]],
                mode="markers",
                name="Normal Output",
                marker=dict(symbol="circle", color="blue", size=14),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[steered_coord[0]],
                y=[steered_coord[1]],
                mode="markers",
                name="Steered Output",
                marker=dict(symbol="star", color="green", size=16, line=dict(color="black", width=1)),
            )
        )

        fig.add_annotation(
            x=steered_coord[0],
            y=steered_coord[1],
            ax=normal_coord[0],
            ay=normal_coord[1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="purple",
        )

        fig.update_layout(
            title=f"Concept Space (Layer {layer_idx})",
            xaxis_title="PCA Dimension 1",
            yaxis_title="PCA Dimension 2",
            width=800,
            height=600,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor="rgba(240,240,240,0.5)",
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
    def _activation_matrix(
        activations: Dict[int, torch.Tensor],
        layer_indices: List[int],
        seq_len: int,
    ) -> np.ndarray:
        rows: List[np.ndarray] = []
        for idx in layer_indices:
            tensor = activations[idx][:, :seq_len, :]
            norms = torch.linalg.norm(tensor, dim=-1)
            rows.append(norms.squeeze(0).cpu().numpy())
        return np.stack(rows)

    @staticmethod
    def _plot_heatmap(
        ax: plt.Axes,
        matrix: np.ndarray,
        tokens: List[str],
        layer_indices: List[int],
        *,
        title: str,
        cmap: str,
        line_idx: int | None,
        line_color: str,
        vmin: float | None = None,
        vmax: float | None = None,
        center: float | None = None,
        annotation: str | None = None,
    ) -> None:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            xticklabels=tokens,
            yticklabels=[f"L{idx}" for idx in layer_indices],
            vmin=vmin,
            vmax=vmax,
            center=center,
        )
        ax.set_title(title)
        ax.set_xlabel("Token")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=10)

        if line_idx is not None:
            ax.axhline(line_idx + 0.5, color=line_color, linewidth=2)
            if annotation:
                ax.text(
                    0,
                    line_idx + 0.15,
                    annotation,
                    color=line_color,
                    fontsize=12,
                    fontweight="bold",
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                )

    @staticmethod
    def _reconcile_tokens(normal_tokens: List[str], steered_tokens: List[str]) -> List[str]:
        if len(normal_tokens) == len(steered_tokens):
            return normal_tokens
        min_len = min(len(normal_tokens), len(steered_tokens))
        return normal_tokens[:min_len]
