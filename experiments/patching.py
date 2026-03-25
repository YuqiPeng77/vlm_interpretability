from __future__ import annotations

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from experiments.base import BaseExperiment
from shared.data_loader import (
    load_image_safe,
    load_records,
    parse_concept_specs,
    resolve_csv_path,
    resolve_image_path,
    select_class_aware_pairs,
    select_global_pairs,
)
from shared.hook_manager import HookManager
from shared.metrics import get_yes_no_probabilities
from shared.model_loader import ModelBundle, load_model_bundle
from shared.visualizer import plot_p_yes


COARSE_COMPONENT_COLORS = {
    "attn_only": "#4040B0",
    "mlp_only": "#D05030",
    "both": "#5F6368",
}
ATTENTION_SUBMODULE_COLORS = {
    "q": "#4056C8",
    "k": "#2E86AB",
    "v": "#2F9E44",
    "proj": "#C06C2B",
}
REFERENCE_LINE_COLOR = "#808080"
REFERENCE_LINE_ALPHA = 0.4
COMPONENT_FILL_ALPHA = 0.08
COMPONENT_LINEWIDTH = 2.4
METRIC_REGISTRY = {
    "probability_change": {
        "plot_dir": "probability_change",
    }
}


class PatchingExperiment(BaseExperiment):
    def __init__(self, config: dict, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self.bundle: ModelBundle | None = None
        self.dataset_root = Path(config["dataset"]["root"])
        self.concepts = parse_concept_specs(config["dataset"], require_group=False)
        self.patching_config = config["patching"]
        self.runtime_config = config.get("runtime", {})
        self.method = self.patching_config["method"]
        self.stage = self.patching_config["stage"]
        self.num_pairs = int(self.config["dataset"].get("num_pairs", 0))
        self.modes = list(self.patching_config.get("modes", ["noise", "zero", "counterfactual"]))
        self.metrics = list(self.patching_config.get("metrics", ["probability_change"]))
        self.metric_registry = METRIC_REGISTRY
        self.component_nested_config = self.patching_config.get("component", {})
        self.levels = self._get_levels()
        self.selected_layers_config = self._get_component_value("selected_layers", None)
        self.components = self._get_components()
        self.attention_submodules = self._get_attention_submodules()
        self._validate_combo()

    def _get_component_value(self, key: str, default):
        if key in self.patching_config:
            return self.patching_config[key]
        legacy_map = {
            "selected_layers": "selected_layers",
            "components": "conditions",
            "attention_submodules": "attention_modules",
            "levels": "levels",
        }
        legacy_key = legacy_map.get(key, key)
        return self.component_nested_config.get(legacy_key, default)

    def _get_levels(self) -> list[int]:
        levels = self._get_component_value("levels", [1, 2, 3])
        return sorted({int(level) for level in levels})

    def _get_components(self) -> list[str]:
        components = self._get_component_value("components", ["attn_only", "mlp_only", "both"])
        return [str(component) for component in components]

    def _get_attention_submodules(self) -> list[str]:
        requested = self._get_component_value("attention_submodules", ["q", "k", "v", "proj"])
        normalized = []
        for name in requested:
            token = str(name).lower()
            if token in {"out", "project"}:
                token = "proj"
            normalized.append(token)
        return normalized

    def _validate_combo(self) -> None:
        valid = {
            ("activation", "decoder"),
            ("component", "encoder"),
            ("component", "decoder"),
        }
        if (self.method, self.stage) not in valid:
            raise ValueError(
                f"Unsupported patching combination method={self.method!r}, stage={self.stage!r}."
            )

        invalid_metrics = sorted(set(self.metrics) - set(self.metric_registry))
        if invalid_metrics:
            raise ValueError(f"Unsupported patching metrics: {invalid_metrics}")

        if self.method == "activation" and self.stage == "decoder":
            invalid_modes = sorted(set(self.modes) - {"noise", "zero", "counterfactual"})
            if invalid_modes:
                raise ValueError(f"Unsupported patching modes: {invalid_modes}")

        if self.method == "component":
            invalid_levels = sorted(set(self.levels) - {1, 2, 3})
            if invalid_levels:
                raise ValueError(f"Unsupported component levels: {invalid_levels}")

            invalid_components = sorted(set(self.components) - {"attn_only", "mlp_only", "both"})
            if invalid_components:
                raise ValueError(f"Unsupported component conditions: {invalid_components}")

            invalid_submodules = sorted(set(self.attention_submodules) - {"q", "k", "v", "proj"})
            if invalid_submodules:
                raise ValueError(f"Unsupported attention_submodules: {invalid_submodules}")

            if self.stage not in {"encoder", "decoder"}:
                raise ValueError("Component patching supports stage=encoder|decoder.")

            if self.selected_layers_config is None:
                raise ValueError("patching.selected_layers is required for method=component.")

    def setup(self) -> None:
        self.bundle = load_model_bundle(self.config["model"])
        self.log(
            "Loaded model for patching: "
            f"method={self.method}, stage={self.stage}, "
            f"enc_layers={self.bundle.num_enc_layers}, dec_layers={self.bundle.num_dec_layers}"
        )

    def _empty_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _metric_plot_dir(self, metric_name: str, *parts: str) -> Path:
        path = self.plots_dir / self.metric_registry[metric_name]["plot_dir"]
        for part in parts:
            path = path / part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_forward(self, inputs: dict) -> torch.Tensor:
        assert self.bundle is not None
        with torch.no_grad():
            outputs = self.bundle.model(**inputs, return_dict=True, use_cache=False)
        return outputs.logits[0, -1, :]

    def _prepare_inputs(self, prompt: str, image_path: Path) -> dict:
        assert self.bundle is not None
        image = load_image_safe(image_path, logger=self.log)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        inputs = self.bundle.processor(images=image, text=prompt, return_tensors="pt")
        return {key: value.to(self.bundle.input_device) for key, value in inputs.items()}

    def _build_prompt(self, concept_name: str) -> str:
        assert self.bundle is not None
        text = self.patching_config["prompt"].format(attribute=concept_name)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        return self.bundle.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _get_stage_layers(self):
        assert self.bundle is not None
        return self.bundle.decoder_layers if self.stage == "decoder" else self.bundle.encoder_blocks

    def _selected_layers(self, n_layers: int) -> list[int]:
        if self.selected_layers_config is None:
            return list(range(n_layers))
        selected = sorted({int(layer_idx) for layer_idx in self.selected_layers_config})
        invalid = [layer_idx for layer_idx in selected if layer_idx < 0 or layer_idx >= n_layers]
        if invalid:
            raise ValueError(f"Selected layers out of range: {invalid}")
        return selected

    def _collect_layer_outputs(self, layers, inputs: dict) -> dict[int, torch.Tensor]:
        with HookManager() as hooks:
            hooks.register_layer_output_hooks(layers)
            _ = self._run_forward(inputs)
            return dict(hooks.hidden_cache)

    def _infer_image_mask_from_hidden(self, first_layer, clean_inputs: dict, corrupted_inputs: dict):
        capture: dict[str, torch.Tensor] = {}

        def pre_hook(_module, inputs):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            capture["hidden"] = hidden.detach().clone()

        handle = first_layer.register_forward_pre_hook(pre_hook)
        _ = self._run_forward(clean_inputs)
        handle.remove()
        clean_hidden = capture.get("hidden")

        capture.clear()
        handle = first_layer.register_forward_pre_hook(pre_hook)
        _ = self._run_forward(corrupted_inputs)
        handle.remove()
        corrupted_hidden = capture.get("hidden")

        if clean_hidden is None or corrupted_hidden is None:
            return None
        if clean_hidden.shape != corrupted_hidden.shape:
            return None

        diff = (clean_hidden - corrupted_hidden).float().norm(dim=-1)[0]
        if diff.numel() == 0:
            return None

        # The decoder sequence mixes image and text tokens. We treat positions with
        # unusually large clean-vs-corrupted hidden differences as image-token candidates.
        threshold = diff.mean() + 0.5 * diff.std()
        mask = diff > threshold
        if int(mask.sum().item()) < 8:
            topk = min(diff.numel(), max(8, int(diff.numel() * 0.15)))
            mask = torch.zeros_like(diff, dtype=torch.bool)
            mask[torch.topk(diff, k=topk).indices] = True
        return mask

    def _get_image_mask(self, clean_inputs: dict, corrupted_inputs: dict):
        assert self.bundle is not None
        inferred = self._infer_image_mask_from_hidden(
            self.bundle.decoder_layers[0], clean_inputs, corrupted_inputs
        )
        if inferred is not None and int(inferred.sum().item()) > 0:
            return inferred

        image_token_id = getattr(self.bundle.processor.tokenizer, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.bundle.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = 151655
        fallback = corrupted_inputs["input_ids"][0] == image_token_id
        return fallback if int(fallback.sum().item()) > 0 else None

    def _patched_activation_forward(
        self,
        layer,
        inputs: dict,
        mode: str,
        image_mask: torch.Tensor,
        seed: int,
        clean_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with HookManager() as hooks:
            hooks.register_activation_patching_hook(
                layer=layer,
                mode=mode,
                image_mask=image_mask,
                seed=seed,
                clean_hidden=clean_hidden,
            )
            return self._run_forward(inputs)

    def _patched_residual_forward(
        self,
        layer,
        inputs: dict,
        mode: str,
        seed: int,
        clean_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with HookManager() as hooks:
            hooks.register_residual_patching_hooks(
                layer=layer,
                mode=mode,
                seed=seed,
                clean_delta=clean_delta,
            )
            return self._run_forward(inputs)

    def _select_pairs(self, records: list[dict], seed: int) -> list[tuple[dict, dict]]:
        if self.method == "activation":
            return select_global_pairs(records, self.num_pairs, seed)
        return select_class_aware_pairs(records, self.num_pairs, seed)

    def _get_attention_module(self, block):
        for attr in ("attn", "self_attn", "attention"):
            module = getattr(block, attr, None)
            if module is not None:
                return module
        raise RuntimeError(f"Could not locate attention module on block type {type(block)}")

    def _get_mlp_module(self, block):
        for attr in ("mlp", "ffn", "feed_forward"):
            module = getattr(block, attr, None)
            if module is not None:
                return module
        raise RuntimeError(f"Could not locate MLP module on block type {type(block)}")

    def _get_attention_proj_module(self, attention_module):
        for attr in ("proj", "out_proj", "o_proj"):
            module = getattr(attention_module, attr, None)
            if module is not None:
                return module
        raise RuntimeError(f"Could not locate attention projection module for {type(attention_module)}")

    def _get_decoder_attention_submodule(self, attention_module, component_name: str):
        candidate_map = {
            "q": ("q_proj", "query_proj", "q"),
            "k": ("k_proj", "key_proj", "k"),
            "v": ("v_proj", "value_proj", "v"),
        }
        if component_name == "proj":
            return self._get_attention_proj_module(attention_module)

        for attr in candidate_map[component_name]:
            module = getattr(attention_module, attr, None)
            if module is not None:
                return module
        raise RuntimeError(
            f"Could not locate decoder attention submodule {component_name!r} on {type(attention_module)}"
        )

    def _align_like(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        aligned = source.to(target.device, target.dtype)
        if aligned.shape == target.shape:
            return aligned

        result = target.clone()
        if aligned.ndim == 3 and target.ndim == 3:
            batch = min(aligned.shape[0], target.shape[0])
            tokens = min(aligned.shape[1], target.shape[1])
            width = min(aligned.shape[2], target.shape[2])
            result[:batch, :tokens, :width] = aligned[:batch, :tokens, :width]
            return result
        if aligned.ndim == 2 and target.ndim == 2:
            tokens = min(aligned.shape[0], target.shape[0])
            width = min(aligned.shape[1], target.shape[1])
            result[:tokens, :width] = aligned[:tokens, :width]
            return result
        return aligned.reshape(target.shape)

    def _patch_qkv_component(self, hidden: torch.Tensor, clean_qkv: torch.Tensor, component_name: str) -> torch.Tensor:
        patched = hidden.clone()
        aligned = self._align_like(clean_qkv, hidden)
        # Encoder attention uses a fused qkv projection, so q/k/v patching is implemented
        # by swapping only the corresponding third of the fused output tensor.
        split_size = hidden.shape[-1] // 3
        offsets = {"q": 0, "k": 1, "v": 2}
        start = offsets[component_name] * split_size
        end = start + split_size
        patched[..., start:end] = aligned[..., start:end]
        return patched

    def _collect_clean_component_cache(
        self,
        block,
        inputs: dict,
        requested_submodules: list[str],
    ) -> dict[str, torch.Tensor]:
        cache: dict[str, torch.Tensor] = {}
        block_input: dict[str, torch.Tensor] = {}
        attention_module = self._get_attention_module(block)
        mlp_module = self._get_mlp_module(block)
        handles = []

        def make_hook(name: str):
            def _hook(_module, _inputs, outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                cache[name] = hidden.detach().clone()

            return _hook

        def block_pre_hook(_module, inputs_tuple):
            hidden = inputs_tuple[0] if isinstance(inputs_tuple, tuple) else inputs_tuple
            block_input["hidden"] = hidden.detach().clone()

        def block_fwd_hook(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            cache["block_output"] = hidden.detach().clone()

        handles.append(block.register_forward_pre_hook(block_pre_hook))
        handles.append(block.register_forward_hook(block_fwd_hook))
        handles.append(attention_module.register_forward_hook(make_hook("attn_only")))
        handles.append(mlp_module.register_forward_hook(make_hook("mlp_only")))

        if requested_submodules:
            if self.stage == "encoder":
                # Encoder q/k/v are conceptual slices of a fused qkv module.
                if any(name in {"q", "k", "v"} for name in requested_submodules):
                    qkv_module = getattr(attention_module, "qkv", None)
                    if qkv_module is None:
                        raise RuntimeError(f"Could not locate fused qkv module on {type(attention_module)}")
                    handles.append(qkv_module.register_forward_hook(make_hook("qkv")))
                if "proj" in requested_submodules:
                    handles.append(
                        self._get_attention_proj_module(attention_module).register_forward_hook(
                            make_hook("proj")
                        )
                    )
            else:
                # Decoder attention exposes q/k/v/proj as separate projection modules.
                for name in requested_submodules:
                    module = self._get_decoder_attention_submodule(attention_module, name)
                    handles.append(module.register_forward_hook(make_hook(name)))

        _ = self._run_forward(inputs)
        for handle in reversed(handles):
            handle.remove()

        cache["both"] = cache["block_output"] - block_input["hidden"].to(cache["block_output"].device)
        return cache

    def _patched_component_forward(
        self,
        block,
        inputs: dict,
        component: str,
        clean_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if component == "both":
            # "both" is the block-level residual baseline: replace the full block delta
            # while preserving the incoming residual stream.
            return self._patched_residual_forward(
                layer=block,
                inputs=inputs,
                mode="counterfactual",
                seed=0,
                clean_delta=clean_cache["both"],
            )

        attention_module = self._get_attention_module(block)
        mlp_module = self._get_mlp_module(block)
        proj_module = self._get_attention_proj_module(attention_module)

        if component == "attn_only":
            target_module = attention_module

            def patch_output(outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                return self._align_like(clean_cache["attn_only"], hidden)

        elif component == "mlp_only":
            target_module = mlp_module

            def patch_output(outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                return self._align_like(clean_cache["mlp_only"], hidden)

        elif component in {"q", "k", "v"}:
            if self.stage == "encoder":
                qkv_module = getattr(attention_module, "qkv", None)
                if qkv_module is None:
                    raise RuntimeError(f"Could not locate fused qkv module on {type(attention_module)}")
                target_module = qkv_module

                def patch_output(outputs):
                    hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                    return self._patch_qkv_component(hidden, clean_cache["qkv"], component)

            else:
                # Decoder q/k/v live in distinct projection modules, so we can patch
                # each submodule directly instead of slicing a fused tensor.
                target_module = self._get_decoder_attention_submodule(attention_module, component)

                def patch_output(outputs):
                    hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                    return self._align_like(clean_cache[component], hidden)

        elif component == "proj":
            target_module = proj_module

            def patch_output(outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                return self._align_like(clean_cache["proj"], hidden)

        else:
            raise ValueError(f"Unsupported component patch request: {component}")

        def hook_fn(_module, _inputs, outputs):
            patched = patch_output(outputs)
            if isinstance(outputs, tuple):
                return (patched,) + outputs[1:]
            return patched

        handle = target_module.register_forward_hook(hook_fn)
        try:
            return self._run_forward(inputs)
        finally:
            handle.remove()

    def _init_layer_store(self, layer_ids: list[int], names: list[str]) -> dict[str, dict[int, list[float]]]:
        return {name: {layer_idx: [] for layer_idx in layer_ids} for name in names}

    def _append_metric(self, store: dict[str, dict[int, list[float]]], name: str, layer_idx: int, value: float) -> None:
        store[name][layer_idx].append(float(value))

    def _summarize_series(
        self,
        p_yes_store: dict[str, dict[int, list[float]]],
        effect_store: dict[str, dict[int, list[float]]],
        selected_layers: list[int],
    ) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for name, layer_map in p_yes_store.items():
            p_yes_means = np.array([np.mean(layer_map[layer]) if layer_map[layer] else np.nan for layer in selected_layers])
            p_yes_stds = np.array([np.std(layer_map[layer]) if layer_map[layer] else np.nan for layer in selected_layers])
            effect_means = np.array(
                [np.mean(effect_store[name][layer]) if effect_store[name][layer] else np.nan for layer in selected_layers]
            )
            effect_stds = np.array(
                [np.std(effect_store[name][layer]) if effect_store[name][layer] else np.nan for layer in selected_layers]
            )
            summary[name] = {
                "p_yes_mean": [None if np.isnan(value) else float(value) for value in p_yes_means],
                "p_yes_std": [None if np.isnan(value) else float(value) for value in p_yes_stds],
                "effect_mean": [None if np.isnan(value) else float(value) for value in effect_means],
                "effect_std": [None if np.isnan(value) else float(value) for value in effect_stds],
                "count": [len(layer_map[layer]) for layer in selected_layers],
            }
        return summary

    def _compute_ratio_series(
        self,
        numerator_series: dict[str, dict],
        denominator_series: dict,
        selected_layers: list[int],
        label_suffix: str,
    ) -> dict[str, dict]:
        ratios: dict[str, dict] = {}
        denominator_values = denominator_series["effect_mean"]
        for name, payload in numerator_series.items():
            values: list[float | None] = []
            counts: list[int] = []
            for offset, _layer_idx in enumerate(selected_layers):
                numerator = payload["effect_mean"][offset]
                denominator = denominator_values[offset]
                # Ratios are defined over intervention effects rather than raw P(Yes),
                # so they describe relative contribution within a layer.
                if numerator is None or denominator is None or abs(denominator) < 1e-8:
                    values.append(None)
                else:
                    values.append(float(numerator / denominator))
                counts.append(payload["count"][offset])
            ratios[f"{name}_{label_suffix}"] = {
                "ratio_mean": values,
                "count": counts,
            }
        return ratios

    def _plot_overlay_probability(
        self,
        save_path: Path,
        selected_layers: list[int],
        series_order: list[str],
        series_summary: dict[str, dict],
        title: str,
        baseline_name: str | None = None,
        baseline_label: str | None = None,
        label_map: dict[str, str] | None = None,
    ) -> None:
        x_values = np.array(selected_layers)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor("white")

        for name in series_order:
            if name == baseline_name or name not in series_summary:
                continue
            means = np.array([
                np.nan if value is None else float(value)
                for value in series_summary[name]["p_yes_mean"]
            ])
            stds = np.array([
                np.nan if value is None else float(value)
                for value in series_summary[name]["p_yes_std"]
            ])
            ax.plot(
                x_values,
                means,
                linewidth=COMPONENT_LINEWIDTH,
                color=(ATTENTION_SUBMODULE_COLORS | COARSE_COMPONENT_COLORS).get(name, "black"),
                label=(label_map or {}).get(name, name),
                solid_capstyle="round",
            )
            lower = means - stds
            upper = means + stds
            ax.fill_between(
                x_values,
                lower,
                upper,
                color=(ATTENTION_SUBMODULE_COLORS | COARSE_COMPONENT_COLORS).get(name, "black"),
                alpha=COMPONENT_FILL_ALPHA,
                linewidth=0.0,
            )

        if baseline_name and baseline_name in series_summary:
            means = np.array([
                np.nan if value is None else float(value)
                for value in series_summary[baseline_name]["p_yes_mean"]
            ])
            ax.plot(
                x_values,
                means,
                linestyle="--",
                linewidth=2.0,
                color=REFERENCE_LINE_COLOR,
                alpha=REFERENCE_LINE_ALPHA,
                label=baseline_label or (label_map or {}).get(baseline_name, baseline_name),
                solid_capstyle="round",
            )

        ax.set_xlabel(f"{self.stage.capitalize()} Layer")
        ax.set_ylabel("P(Yes)")
        ax.set_title(title)
        ax.set_xticks(selected_layers)
        ax.grid(axis="y", alpha=0.2, color="#C8CCD4", linewidth=0.8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)

    def _metric_rows_for_standard(
        self,
        concept_name: str,
        concept_group: str | None,
        series_summary: dict[str, dict],
        metric_name: str,
    ) -> list[dict]:
        rows: list[dict] = []
        for mode, payload in series_summary.items():
            for layer_idx, p_yes_mean in enumerate(payload["p_yes_mean"]):
                rows.append(
                    {
                        "metric": metric_name,
                        "view": "standard",
                        "series": mode,
                        "concept": concept_name,
                        "group": concept_group,
                        "method": self.method,
                        "stage": self.stage,
                        "layer": layer_idx,
                        "p_yes_mean": p_yes_mean,
                        "p_yes_std": payload["p_yes_std"][layer_idx],
                        "effect_mean": payload["effect_mean"][layer_idx],
                        "effect_std": payload["effect_std"][layer_idx],
                        "count": payload["count"][layer_idx],
                    }
                )
        return rows

    def _metric_rows_for_component(
        self,
        concept_name: str,
        concept_group: str | None,
        selected_layers: list[int],
        section_name: str,
        section_summary: dict[str, dict],
        metric_name: str,
        ratio: bool = False,
    ) -> list[dict]:
        rows: list[dict] = []
        for series_name, payload in section_summary.items():
            for offset, layer_idx in enumerate(selected_layers):
                row = {
                    "metric": metric_name,
                    "view": section_name,
                    "series": series_name,
                    "concept": concept_name,
                    "group": concept_group,
                    "method": self.method,
                    "stage": self.stage,
                    "layer": layer_idx,
                    "count": payload["count"][offset],
                }
                if ratio:
                    row["ratio_mean"] = payload["ratio_mean"][offset]
                else:
                    row["p_yes_mean"] = payload["p_yes_mean"][offset]
                    row["p_yes_std"] = payload["p_yes_std"][offset]
                    row["effect_mean"] = payload["effect_mean"][offset]
                    row["effect_std"] = payload["effect_std"][offset]
                rows.append(row)
        return rows

    def _run_activation_patching(self) -> None:
        assert self.bundle is not None
        layers = self._get_stage_layers()
        n_layers = len(layers)
        summary: dict[str, dict] = {}

        for concept_idx, concept in enumerate(self.concepts):
            csv_path = resolve_csv_path(self.dataset_root, concept.csv_stem)
            records = load_records(csv_path)
            pairs = self._select_pairs(records, int(self.runtime_config.get("seed", 42)) + concept_idx)
            prompt = self._build_prompt(concept.name)

            p_yes_values = self._init_layer_store(list(range(n_layers)), self.modes)
            effect_values = self._init_layer_store(list(range(n_layers)), self.modes)
            clean_baselines: list[float] = []
            corrupted_baselines: list[float] = []
            valid_pairs = 0

            progress = tqdm(pairs, desc=f"patching-{concept.name}", unit="pair")
            for pair_idx, (positive, negative) in enumerate(progress):
                positive_path = resolve_image_path(self.dataset_root, positive["image_path"])
                negative_path = resolve_image_path(self.dataset_root, negative["image_path"])
                if not positive_path.exists() or not negative_path.exists():
                    self.log(f"Skipping missing pair ({positive_path}, {negative_path})")
                    continue

                try:
                    positive_inputs = self._prepare_inputs(prompt, positive_path)
                    negative_inputs = self._prepare_inputs(prompt, negative_path)
                except Exception as exc:
                    self.log(f"Skipping pair for {concept.name}: {exc}")
                    continue

                clean_logits = self._run_forward(positive_inputs)
                corrupted_logits = self._run_forward(negative_inputs)
                p_yes_clean, _ = get_yes_no_probabilities(clean_logits, self.bundle.processor.tokenizer)
                p_yes_corrupted, _ = get_yes_no_probabilities(
                    corrupted_logits, self.bundle.processor.tokenizer
                )
                clean_baselines.append(p_yes_clean)
                corrupted_baselines.append(p_yes_corrupted)

                image_mask = self._get_image_mask(positive_inputs, negative_inputs)
                if image_mask is None or int(image_mask.sum().item()) == 0:
                    self.log(f"Skipping pair for {concept.name}: could not infer image token mask.")
                    continue
                clean_states = self._collect_layer_outputs(layers, positive_inputs)

                for layer_idx in range(n_layers):
                    for mode in self.modes:
                        seed = int(self.runtime_config.get("seed", 42)) + pair_idx * 1000 + layer_idx
                        patched_logits = self._patched_activation_forward(
                            layer=layers[layer_idx],
                            inputs=negative_inputs,
                            mode=mode,
                            image_mask=image_mask,
                            seed=seed,
                            clean_hidden=clean_states[layer_idx] if mode == "counterfactual" else None,
                        )
                        p_yes_patched, _ = get_yes_no_probabilities(
                            patched_logits, self.bundle.processor.tokenizer
                        )
                        self._append_metric(p_yes_values, mode, layer_idx, p_yes_patched)
                        self._append_metric(effect_values, mode, layer_idx, p_yes_patched - p_yes_corrupted)

                valid_pairs += 1
                self._empty_cache()

            p_yes_clean_mean = float(np.mean(clean_baselines)) if clean_baselines else 0.0
            p_yes_corrupted_mean = float(np.mean(corrupted_baselines)) if corrupted_baselines else 0.0
            metric_rows: list[dict] = []
            metric_results: dict[str, dict] = {}

            for metric_name in self.metrics:
                if metric_name != "probability_change":
                    continue
                probability_summary = self._summarize_series(
                    p_yes_values,
                    effect_values,
                    list(range(n_layers)),
                )
                metric_results[metric_name] = {
                    "p_yes_clean_mean": p_yes_clean_mean,
                    "p_yes_corrupted_mean": p_yes_corrupted_mean,
                    "modes": probability_summary,
                }
                metric_rows.extend(
                    self._metric_rows_for_standard(concept.name, concept.group, probability_summary, metric_name)
                )
                metric_dir = self._metric_plot_dir(metric_name)
                for mode in self.modes:
                    payload = probability_summary[mode]
                    p_yes_means = np.array([np.nan if value is None else float(value) for value in payload["p_yes_mean"]])
                    p_yes_stds = np.array([np.nan if value is None else float(value) for value in payload["p_yes_std"]])
                    plot_p_yes(
                        p_yes_means,
                        p_yes_stds,
                        stage=self.stage,
                        mode=mode,
                        concept_name=concept.name,
                        p_yes_clean_mean=p_yes_clean_mean,
                        p_yes_corrupted_mean=p_yes_corrupted_mean,
                        save_path=metric_dir / f"{concept.name}_{mode}_p_yes.png",
                    )

            concept_result = {
                "group": concept.group,
                "method": self.method,
                "stage": self.stage,
                "selected_pairs": len(pairs),
                "valid_pairs": valid_pairs,
                "metrics": metric_results,
            }
            summary[concept.name] = concept_result
            self.save_json(concept_result, f"{concept.name}_results.json")
            self.save_csv(metric_rows, f"{concept.name}_results.csv")
            self.log(f"Finished patching for concept {concept.name}")

        self.save_json(
            {
                "meta": {
                    "model_path": self.config["model"]["path"],
                    "method": self.method,
                    "stage": self.stage,
                    "modes": self.modes,
                    "metrics": self.metrics,
                    "num_pairs": self.num_pairs,
                    "seed": int(self.runtime_config.get("seed", 42)),
                },
                "results": summary,
            },
            "summary.json",
        )
        self.log("Activation patching experiment completed.")

    def _required_component_sets(self) -> tuple[list[str], list[str]]:
        coarse = set(self.components)
        submodules = set()

        if 1 in self.levels:
            coarse.add("both")
        if 2 in self.levels:
            coarse.update({"attn_only", "mlp_only", "both"})
        if 3 in self.levels:
            coarse.add("attn_only")
            submodules.update(self.attention_submodules)

        ordered_coarse = [name for name in ("attn_only", "mlp_only", "both") if name in coarse]
        ordered_submodules = [name for name in ("q", "k", "v", "proj") if name in submodules]
        return ordered_coarse, ordered_submodules

    def _run_component_patching(self) -> None:
        assert self.bundle is not None
        layers = self._get_stage_layers()
        selected_layers = self._selected_layers(len(layers))
        required_coarse, required_submodules = self._required_component_sets()
        summary: dict[str, dict] = {}

        for concept_idx, concept in enumerate(self.concepts):
            csv_path = resolve_csv_path(self.dataset_root, concept.csv_stem)
            records = load_records(csv_path)
            pairs = self._select_pairs(records, int(self.runtime_config.get("seed", 42)) + concept_idx)
            prompt = self._build_prompt(concept.name)

            coarse_p_yes = self._init_layer_store(selected_layers, required_coarse)
            coarse_effect = self._init_layer_store(selected_layers, required_coarse)
            submodule_p_yes = self._init_layer_store(selected_layers, required_submodules)
            submodule_effect = self._init_layer_store(selected_layers, required_submodules)
            clean_baselines: list[float] = []
            corrupted_baselines: list[float] = []
            valid_pairs = 0

            progress = tqdm(pairs, desc=f"component-{concept.name}", unit="pair")
            for _, (positive, negative) in enumerate(progress):
                positive_path = resolve_image_path(self.dataset_root, positive["image_path"])
                negative_path = resolve_image_path(self.dataset_root, negative["image_path"])
                if not positive_path.exists() or not negative_path.exists():
                    self.log(f"Skipping missing pair ({positive_path}, {negative_path})")
                    continue

                try:
                    positive_inputs = self._prepare_inputs(prompt, positive_path)
                    negative_inputs = self._prepare_inputs(prompt, negative_path)
                except Exception as exc:
                    self.log(f"Skipping pair for {concept.name}: {exc}")
                    continue

                clean_logits = self._run_forward(positive_inputs)
                corrupted_logits = self._run_forward(negative_inputs)
                p_yes_clean, _ = get_yes_no_probabilities(clean_logits, self.bundle.processor.tokenizer)
                p_yes_corrupted, _ = get_yes_no_probabilities(
                    corrupted_logits, self.bundle.processor.tokenizer
                )
                clean_baselines.append(p_yes_clean)
                corrupted_baselines.append(p_yes_corrupted)

                for layer_idx in selected_layers:
                    block = layers[layer_idx]
                    clean_cache = self._collect_clean_component_cache(
                        block,
                        positive_inputs,
                        requested_submodules=required_submodules,
                    )

                    for condition in required_coarse:
                        patched_logits = self._patched_component_forward(
                            block=block,
                            inputs=negative_inputs,
                            component=condition,
                            clean_cache=clean_cache,
                        )
                        p_yes_patched, _ = get_yes_no_probabilities(
                            patched_logits, self.bundle.processor.tokenizer
                        )
                        self._append_metric(coarse_p_yes, condition, layer_idx, p_yes_patched)
                        self._append_metric(coarse_effect, condition, layer_idx, p_yes_patched - p_yes_corrupted)

                    for submodule in required_submodules:
                        patched_logits = self._patched_component_forward(
                            block=block,
                            inputs=negative_inputs,
                            component=submodule,
                            clean_cache=clean_cache,
                        )
                        p_yes_patched, _ = get_yes_no_probabilities(
                            patched_logits, self.bundle.processor.tokenizer
                        )
                        self._append_metric(submodule_p_yes, submodule, layer_idx, p_yes_patched)
                        self._append_metric(submodule_effect, submodule, layer_idx, p_yes_patched - p_yes_corrupted)

                valid_pairs += 1
                self._empty_cache()

            p_yes_clean_mean = float(np.mean(clean_baselines)) if clean_baselines else 0.0
            p_yes_corrupted_mean = float(np.mean(corrupted_baselines)) if corrupted_baselines else 0.0
            metric_results: dict[str, dict] = {}
            metric_rows: list[dict] = []

            for metric_name in self.metrics:
                if metric_name != "probability_change":
                    continue

                coarse_summary = self._summarize_series(coarse_p_yes, coarse_effect, selected_layers)
                submodule_summary = self._summarize_series(
                    submodule_p_yes,
                    submodule_effect,
                selected_layers,
                ) if required_submodules else {}

                block_residual = {}
                component_decomposition = {}
                attn_submodule_decomposition = {}

                if 1 in self.levels:
                    block_residual = {
                        "layers": selected_layers,
                        "series": {"both": coarse_summary["both"]},
                    }
                    metric_rows.extend(
                        self._metric_rows_for_component(
                            concept.name,
                            concept.group,
                            selected_layers,
                            "block_residual",
                            {"both": coarse_summary["both"]},
                            metric_name,
                        )
                    )
                    metric_dir = self._metric_plot_dir(metric_name, "level_1_block_residual")
                    self._plot_overlay_probability(
                        metric_dir / f"{concept.name}_block_residual_p_yes.png",
                        selected_layers,
                        ["both"],
                        {"both": coarse_summary["both"]},
                        f"Block Residual P(Yes): {concept.name}",
                        label_map={"both": "block residual"},
                    )

                if 2 in self.levels:
                    component_decomposition = {
                        "layers": selected_layers,
                        "series": {
                            "attn_only": coarse_summary["attn_only"],
                            "mlp_only": coarse_summary["mlp_only"],
                            "both_reference": coarse_summary["both"],
                        },
                    }
                    metric_rows.extend(
                        self._metric_rows_for_component(
                            concept.name,
                            concept.group,
                            selected_layers,
                            "component_decomposition",
                            {
                                "attn_only": coarse_summary["attn_only"],
                                "mlp_only": coarse_summary["mlp_only"],
                                "both_reference": coarse_summary["both"],
                            },
                            metric_name,
                        )
                    )
                    metric_dir = self._metric_plot_dir(metric_name, "level_2_component_decomposition")
                    self._plot_overlay_probability(
                        metric_dir / f"{concept.name}_component_decomposition_p_yes.png",
                        selected_layers,
                        ["attn_only", "mlp_only", "both"],
                        {
                            "attn_only": coarse_summary["attn_only"],
                            "mlp_only": coarse_summary["mlp_only"],
                            "both": coarse_summary["both"],
                        },
                        f"Component Decomposition P(Yes): {concept.name}",
                        baseline_name="both",
                        baseline_label="block residual",
                        label_map={
                            "attn_only": "attention",
                            "mlp_only": "MLP",
                            "both": "block residual",
                        },
                    )

                if 3 in self.levels and required_submodules:
                    visible_submodules = [name for name in required_submodules if name != "proj"]
                    attn_submodule_decomposition = {
                        "layers": selected_layers,
                        "series": {
                            **{name: submodule_summary[name] for name in visible_submodules},
                            "attn_reference": coarse_summary["attn_only"],
                        },
                    }
                    metric_rows.extend(
                        self._metric_rows_for_component(
                            concept.name,
                            concept.group,
                            selected_layers,
                            "attn_submodule_decomposition",
                            {
                                **{name: submodule_summary[name] for name in visible_submodules},
                                "attn_reference": coarse_summary["attn_only"],
                            },
                            metric_name,
                        )
                    )
                    metric_dir = self._metric_plot_dir(metric_name, "level_3_attn_submodule_decomposition")
                    self._plot_overlay_probability(
                        metric_dir / f"{concept.name}_attn_submodule_decomposition_p_yes.png",
                        selected_layers,
                        ["q", "k", "v", "attn_only"],
                        {
                            **{name: submodule_summary[name] for name in visible_submodules},
                            "attn_only": coarse_summary["attn_only"],
                        },
                        f"Attention Submodule Decomposition P(Yes): {concept.name}",
                        baseline_name="attn_only",
                        baseline_label="full attention",
                        label_map={
                            "q": "Q",
                            "k": "K",
                            "v": "V",
                            "attn_only": "full attention",
                        },
                    )

                metric_results[metric_name] = {
                    "p_yes_clean_mean": p_yes_clean_mean,
                    "p_yes_corrupted_mean": p_yes_corrupted_mean,
                    "block_residual": block_residual,
                    "component_decomposition": component_decomposition,
                    "attn_submodule_decomposition": attn_submodule_decomposition,
                }

            concept_result = {
                "group": concept.group,
                "method": self.method,
                "stage": self.stage,
                "selected_pairs": len(pairs),
                "valid_pairs": valid_pairs,
                "selected_layers": selected_layers,
                "levels": self.levels,
                "components": required_coarse,
                "attention_submodules": required_submodules,
                "metrics": metric_results,
            }
            summary[concept.name] = concept_result
            self.save_json(concept_result, f"{concept.name}_results.json")
            self.save_csv(metric_rows, f"{concept.name}_results.csv")
            self.log(f"Finished component patching for concept {concept.name}")

        self.save_json(
            {
                "meta": {
                    "model_path": self.config["model"]["path"],
                    "method": self.method,
                    "stage": self.stage,
                    "metrics": self.metrics,
                    "levels": self.levels,
                    "selected_layers": self.selected_layers_config,
                    "components": self.components,
                    "attention_submodules": self.attention_submodules,
                    "num_pairs": self.num_pairs,
                    "seed": int(self.runtime_config.get("seed", 42)),
                },
                "results": summary,
            },
            "summary.json",
        )
        self.log("Component patching experiment completed.")

    def run(self) -> None:
        if self.bundle is None:
            self.setup()
        assert self.bundle is not None

        if self.method == "component":
            self._run_component_patching()
            return
        self._run_activation_patching()
