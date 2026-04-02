from __future__ import annotations

import gc
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from experiments.base import BaseExperiment
from shared.data_loader import load_image_safe, load_records, resolve_csv_path, resolve_image_path
from shared.model_loader import ModelBundle, load_model_bundle
from shared.visualizer import (
    plot_attention_allocation,
    plot_head_importance_heatmap,
    plot_top_head_detail,
)


@dataclass(frozen=True)
class AttentionConceptSpec:
    name: str
    csv_stem: str
    group: str


class AttentionAnalysisExperiment(BaseExperiment):
    def __init__(self, config: dict, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self.config = config
        self.bundle: ModelBundle | None = None
        self.dataset_root = Path(config["dataset"]["root"])
        self.dataset_config = config["dataset"]
        self.analysis_config = config["attention_analysis"]
        self.runtime_config = config.get("runtime", {})
        self.seed = int(self.runtime_config.get("seed", 42))
        self.num_samples = int(self.dataset_config.get("num_samples", 0))
        self.top_k_heads = int(self.analysis_config.get("top_k_heads", 10))
        self.token_groups = list(self.analysis_config.get("token_groups", ["image", "text", "keyword"]))
        self.prompt_template = self.analysis_config["prompt"]
        self.concepts = self._parse_concepts(self.dataset_config)
        self._logged_eager_switch = False

    def _parse_concepts(self, dataset_config: dict) -> list[AttentionConceptSpec]:
        parsed: list[AttentionConceptSpec] = []
        concepts = dataset_config.get("concepts", [])
        if not concepts:
            raise ValueError("dataset.concepts must contain at least one concept.")
        for item in concepts:
            group = str(item.get("group", "")).strip().lower().replace("-", "_")
            if group == "control":
                group = "non_affective"
            if group not in {"affective", "non_affective"}:
                raise ValueError(
                    f"Concept {item.get('name')!r} must declare group=affective|non_affective|control."
                )
            csv_stem = item.get("csv_stem") or item.get("csv")
            if not csv_stem:
                raise ValueError(f"Concept {item.get('name')!r} must provide csv_stem or csv.")
            parsed.append(
                AttentionConceptSpec(
                    name=item["name"],
                    csv_stem=csv_stem,
                    group=group,
                )
            )
        return parsed

    def setup(self) -> None:
        self.bundle = load_model_bundle(self.config["model"])
        self.log(
            "Loaded model for attention analysis: "
            f"dec_layers={self.bundle.num_dec_layers}, top_k_heads={self.top_k_heads}"
        )

    def _empty_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _plot_dir(self, *parts: str) -> Path:
        path = self.plots_dir
        for part in parts:
            path = path / part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_prompt(self, concept_name: str) -> str:
        assert self.bundle is not None
        text = self.prompt_template.format(attribute=concept_name)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        return self.bundle.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _prepare_inputs(self, prompt: str, image_path: Path) -> dict:
        assert self.bundle is not None
        image = load_image_safe(image_path, logger=self.log)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        inputs = self.bundle.processor(images=image, text=prompt, return_tensors="pt")
        return {key: value.to(self.bundle.input_device) for key, value in inputs.items()}

    def _sample_balanced_records(
        self,
        records: list[dict],
        max_per_class: int,
        seed: int,
    ) -> tuple[list[dict], list[dict]]:
        rng = random.Random(seed)
        positives = [record for record in records if record["attribute_label"] == 1]
        negatives = [record for record in records if record["attribute_label"] == 0]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        if max_per_class > 0:
            positives = positives[:max_per_class]
            negatives = negatives[:max_per_class]
        return positives, negatives

    def _resolve_image_token_id(self) -> int | None:
        assert self.bundle is not None
        tokenizer = self.bundle.processor.tokenizer
        token_id = getattr(tokenizer, "image_token_id", None)
        if token_id is None:
            token_id = getattr(self.bundle.model.config, "image_token_id", None)
        if token_id is None:
            token_id = 151655
        return token_id

    def _infer_image_mask_from_hidden(
        self,
        clean_inputs: dict,
        contrast_inputs: dict,
    ) -> torch.Tensor | None:
        assert self.bundle is not None
        first_layer = self.bundle.decoder_layers[0]
        capture: dict[str, torch.Tensor] = {}

        def pre_hook(_module, inputs):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            capture["hidden"] = hidden.detach().clone()

        handle = first_layer.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            _ = self.bundle.model(**clean_inputs, return_dict=True, use_cache=False)
        handle.remove()
        clean_hidden = capture.get("hidden")

        capture.clear()
        handle = first_layer.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            _ = self.bundle.model(**contrast_inputs, return_dict=True, use_cache=False)
        handle.remove()
        contrast_hidden = capture.get("hidden")

        if clean_hidden is None or contrast_hidden is None:
            return None
        if clean_hidden.shape != contrast_hidden.shape:
            return None

        diff = (clean_hidden - contrast_hidden).float().norm(dim=-1)[0]
        if diff.numel() == 0:
            return None

        threshold = diff.mean() + 0.5 * diff.std()
        mask = diff > threshold
        if int(mask.sum().item()) < 8:
            topk = min(diff.numel(), max(8, int(diff.numel() * 0.15)))
            mask = torch.zeros_like(diff, dtype=torch.bool)
            mask[torch.topk(diff, k=topk).indices] = True
        return mask

    def _get_image_mask(self, inputs: dict, contrast_inputs: dict | None = None) -> torch.Tensor:
        image_token_id = self._resolve_image_token_id()
        if contrast_inputs is not None:
            inferred = self._infer_image_mask_from_hidden(inputs, contrast_inputs)
            if inferred is not None and int(inferred.sum().item()) > 0:
                return inferred.detach().cpu()

        fallback = inputs["input_ids"][0].detach().cpu() == int(image_token_id)
        if int(fallback.sum().item()) > 0:
            return fallback
        return torch.zeros_like(inputs["input_ids"][0].detach().cpu(), dtype=torch.bool)

    def _get_attention_module(self, layer):
        for attr in ("self_attn", "attn", "attention"):
            module = getattr(layer, attr, None)
            if module is not None:
                return module
        raise RuntimeError(f"Could not locate decoder attention module on layer type {type(layer)}")

    def _get_v_proj_module(self, attention_module):
        for attr in ("v_proj", "value_proj", "v"):
            module = getattr(attention_module, attr, None)
            if module is not None:
                return module
        raise RuntimeError(f"Could not locate decoder value projection on {type(attention_module)}")

    def _get_num_heads(self, attention_module) -> int:
        for attr in ("num_heads", "num_attention_heads", "n_heads"):
            value = getattr(attention_module, attr, None)
            if value is not None:
                return int(value)
        config = getattr(attention_module, "config", None)
        if config is not None:
            for attr in ("num_attention_heads", "num_heads", "n_heads"):
                value = getattr(config, attr, None)
                if value is not None:
                    return int(value)
        raise RuntimeError(f"Could not locate num_heads on {type(attention_module)}")

    def _get_num_kv_heads(self, attention_module) -> int:
        for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
            value = getattr(attention_module, attr, None)
            if value is not None:
                return int(value)
        config = getattr(attention_module, "config", None)
        if config is not None:
            for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
                value = getattr(config, attr, None)
                if value is not None:
                    return int(value)
        return self._get_num_heads(attention_module)

    def _get_head_dim(self, attention_module, num_heads: int) -> int:
        for attr in ("head_dim",):
            value = getattr(attention_module, attr, None)
            if value is not None:
                return int(value)
        hidden_size = getattr(attention_module, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size) // int(num_heads)
        proj = self._get_v_proj_module(attention_module)
        if hasattr(proj, "out_features"):
            return int(proj.out_features) // max(self._get_num_kv_heads(attention_module), 1)
        raise RuntimeError(f"Could not infer head_dim on {type(attention_module)}")

    def _expand_to_query_heads(self, tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
        if tensor.shape[0] == target_heads:
            return tensor
        if target_heads % tensor.shape[0] != 0:
            raise RuntimeError(
                f"Cannot expand from {tensor.shape[0]} heads to {target_heads} query heads."
            )
        # Qwen-style GQA can return fewer KV heads than query heads. We repeat each
        # KV-group slice so every downstream statistic is reported at query-head granularity.
        repeats = target_heads // tensor.shape[0]
        return tensor.repeat_interleave(repeats, dim=0)

    def _collect_attention_configs(self, attention_modules: list) -> list:
        assert self.bundle is not None
        configs = []
        seen = set()
        candidates = [self.bundle.model.config, getattr(self.bundle.model.config, "text_config", None)]
        candidates.extend(getattr(module, "config", None) for module in attention_modules)
        for config in candidates:
            if config is None:
                continue
            config_id = id(config)
            if config_id in seen:
                continue
            seen.add(config_id)
            configs.append(config)
        return configs

    @contextmanager
    def _attention_capture_mode(self, attention_modules: list):
        configs = self._collect_attention_configs(attention_modules)
        snapshots: list[tuple[object, object, object]] = []
        switched = False
        try:
            for config in configs:
                previous_impl = getattr(config, "_attn_implementation", None)
                previous_output = getattr(config, "_output_attentions", None)
                snapshots.append((config, previous_impl, previous_output))
                if previous_impl != "eager":
                    setattr(config, "_attn_implementation", "eager")
                    switched = True
                # We set the private slot directly after forcing eager to avoid
                # tripping the config property's validation on sdpa-backed runs.
                setattr(config, "_output_attentions", True)
            if switched and not self._logged_eager_switch:
                self.log("Switching decoder attention capture to eager for attention_analysis.")
                self._logged_eager_switch = True
            yield
        finally:
            for config, previous_impl, previous_output in snapshots:
                setattr(config, "_attn_implementation", previous_impl)
                setattr(config, "_output_attentions", previous_output)

    def _match_keyword_mask(self, input_ids: torch.Tensor, concept_name: str) -> torch.Tensor:
        assert self.bundle is not None
        tokenizer = self.bundle.processor.tokenizer
        base = input_ids.detach().cpu().tolist()
        candidates: list[list[int]] = []
        for text in (concept_name, f" {concept_name}"):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if token_ids and token_ids not in candidates:
                candidates.append(token_ids)

        keyword_mask = torch.zeros(len(base), dtype=torch.bool)
        for token_ids in candidates:
            width = len(token_ids)
            for start in range(0, len(base) - width + 1):
                if base[start : start + width] == token_ids:
                    keyword_mask[start : start + width] = True
        return keyword_mask

    def _identify_token_groups(
        self,
        input_ids: torch.Tensor,
        image_mask: torch.Tensor,
        concept_name: str,
    ) -> dict[str, torch.Tensor]:
        image = image_mask.detach().cpu().bool()
        text = ~image
        keyword = self._match_keyword_mask(input_ids.detach().cpu(), concept_name) & text
        return {
            "image": image,
            "text": text,
            "keyword": keyword,
        }

    def _extract_attention_info(
        self,
        inputs: dict,
        contrast_inputs: dict | None = None,
    ) -> dict:
        assert self.bundle is not None
        decoder_layers = self.bundle.decoder_layers
        attention_modules = [self._get_attention_module(layer) for layer in decoder_layers]
        v_proj_outputs: dict[int, torch.Tensor] = {}
        fallback_attentions: dict[int, torch.Tensor] = {}
        handles = []
        original_forwards: dict[int, object] = {}

        def make_v_hook(layer_idx: int):
            def _hook(_module, _inputs, outputs):
                tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                v_proj_outputs[layer_idx] = tensor.detach().clone()

            return _hook

        def make_wrapped_forward(layer_idx: int, original_forward):
            def _wrapped(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                # Some builds only expose attentions through the module output tuple,
                # so we keep a local fallback cache in addition to output_attentions=True.
                if isinstance(outputs, tuple) and len(outputs) > 1 and torch.is_tensor(outputs[1]):
                    fallback_attentions[layer_idx] = outputs[1].detach().clone()
                return outputs

            return _wrapped

        for layer_idx, attention_module in enumerate(attention_modules):
            handles.append(self._get_v_proj_module(attention_module).register_forward_hook(make_v_hook(layer_idx)))
            original_forward = attention_module.forward
            original_forwards[layer_idx] = original_forward
            attention_module.forward = make_wrapped_forward(layer_idx, original_forward)

        outputs = None
        try:
            with self._attention_capture_mode(attention_modules):
                with torch.no_grad():
                    try:
                        outputs = self.bundle.model(
                            **inputs,
                            return_dict=True,
                            use_cache=False,
                            output_attentions=True,
                        )
                    except (TypeError, ValueError):
                        outputs = self.bundle.model(
                            **inputs,
                            return_dict=True,
                            use_cache=False,
                        )
        finally:
            for layer_idx, attention_module in enumerate(attention_modules):
                attention_module.forward = original_forwards[layer_idx]
            for handle in reversed(handles):
                handle.remove()

        attention_tensors = getattr(outputs, "attentions", None)
        input_ids = inputs["input_ids"][0].detach().cpu()
        image_mask = self._get_image_mask(inputs, contrast_inputs)

        layer_payloads: list[dict] = []
        for layer_idx, attention_module in enumerate(attention_modules):
            weights = None
            if attention_tensors is not None and layer_idx < len(attention_tensors):
                weights = attention_tensors[layer_idx]
            if weights is None:
                weights = fallback_attentions.get(layer_idx)
            if weights is None:
                raise RuntimeError(
                    f"Attention weights were not available for decoder layer {layer_idx}. "
                    "This transformers/Qwen3-VL build may require a deeper attention wrapper."
                )

            weights = weights.detach()
            if weights.ndim == 4:
                weights = weights[0]
            elif weights.ndim != 3:
                raise RuntimeError(
                    f"Unexpected attention weight shape at layer {layer_idx}: {tuple(weights.shape)}"
                )

            num_heads = self._get_num_heads(attention_module)
            num_kv_heads = self._get_num_kv_heads(attention_module)
            if weights.shape[0] != num_heads:
                weights = self._expand_to_query_heads(weights, num_heads)
            last_weights = weights[:, -1, :]

            if layer_idx not in v_proj_outputs:
                raise RuntimeError(f"Missing v_proj output for decoder layer {layer_idx}")
            v_states = v_proj_outputs[layer_idx]
            if v_states.ndim == 3:
                v_states = v_states[0]
            elif v_states.ndim != 2:
                raise RuntimeError(
                    f"Unexpected v_proj output shape at layer {layer_idx}: {tuple(v_states.shape)}"
                )

            head_dim = self._get_head_dim(attention_module, num_heads)
            seq_len = v_states.shape[0]
            inferred_kv_heads = max(v_states.shape[-1] // head_dim, 1)
            v_states = v_states.reshape(seq_len, inferred_kv_heads, head_dim).permute(1, 0, 2)
            v_states = self._expand_to_query_heads(v_states, num_heads)

            aligned_weights = last_weights.to(v_states.device, v_states.dtype)
            # We treat the per-head context vector before output projection as the
            # head's actual contribution candidate to the residual stream.
            weighted_values = torch.einsum("hs,hsd->hd", aligned_weights, v_states)

            layer_payloads.append(
                {
                    "attention_weights_last": last_weights.detach().float().cpu(),
                    "weighted_values_last": weighted_values.detach().float().cpu(),
                    "value_states": v_states.detach().float().cpu(),
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                }
            )

        return {
            "layers": layer_payloads,
            "input_ids": input_ids,
            "image_mask": image_mask,
        }

    def _sample_to_attention_matrix(self, sample: dict) -> np.ndarray:
        return np.stack(
            [layer_payload["attention_weights_last"].numpy() for layer_payload in sample["layers"]],
            axis=0,
        )

    def _sample_to_weighted_matrix(self, sample: dict) -> np.ndarray:
        return np.stack(
            [layer_payload["weighted_values_last"].numpy() for layer_payload in sample["layers"]],
            axis=0,
        )

    def _sample_to_value_state_tensor(self, sample: dict) -> np.ndarray:
        return np.stack(
            [layer_payload["value_states"].numpy() for layer_payload in sample["layers"]],
            axis=0,
        )

    def _compute_head_importance(
        self,
        positive_values: list[np.ndarray],
        negative_values: list[np.ndarray],
    ) -> np.ndarray:
        if not positive_values or not negative_values:
            raise ValueError("Cannot compute head importance without both positive and negative samples.")
        pos_mean = np.mean(np.stack(positive_values, axis=0), axis=0)
        neg_mean = np.mean(np.stack(negative_values, axis=0), axis=0)
        # Importance is defined as how far the mean per-head written content moves
        # between positive and negative images for the same concept.
        return np.linalg.norm(pos_mean - neg_mean, axis=-1)

    def _compute_layer_attention_allocation(self, samples: list[dict]) -> dict[str, np.ndarray]:
        def collect(token_group: str) -> np.ndarray:
            values: list[np.ndarray] = []
            for sample in samples:
                attention = self._sample_to_attention_matrix(sample)
                mask = sample["token_groups"][token_group].numpy().astype(bool)
                if not np.any(mask):
                    values.append(np.zeros(attention.shape[0], dtype=float))
                    continue
                grouped = attention[:, :, mask].sum(axis=-1).mean(axis=1)
                values.append(grouped)
            return np.mean(np.stack(values, axis=0), axis=0) if values else np.array([], dtype=float)

        return {
            "image": collect("image"),
            "text": collect("text"),
        }

    def _compute_weighted_value_norm_allocation(self, samples: list[dict]) -> dict[str, np.ndarray]:
        def collect(token_group: str) -> np.ndarray:
            values: list[np.ndarray] = []
            for sample in samples:
                attention = self._sample_to_attention_matrix(sample)
                value_states = self._sample_to_value_state_tensor(sample)
                mask = sample["token_groups"][token_group].numpy().astype(bool)
                if not np.any(mask):
                    values.append(np.zeros(attention.shape[0], dtype=float))
                    continue
                group_context = np.einsum("lhs,lhsd->lhd", attention[:, :, mask], value_states[:, :, mask, :])
                values.append(np.linalg.norm(group_context, axis=-1).mean(axis=1))
            return np.mean(np.stack(values, axis=0), axis=0) if values else np.array([], dtype=float)

        return {
            "image": collect("image"),
            "text": collect("text"),
        }

    def _compute_mean_weighted_value_norm(self, samples: list[dict]) -> np.ndarray:
        if not samples:
            raise ValueError("Cannot compute weighted value norm without samples.")
        matrices = [np.linalg.norm(self._sample_to_weighted_matrix(sample), axis=-1) for sample in samples]
        return np.mean(np.stack(matrices, axis=0), axis=0)

    def _select_top_heads(self, importance: np.ndarray, top_k: int) -> list[tuple[int, int, float]]:
        flat_indices = np.argsort(np.nan_to_num(importance, nan=-np.inf), axis=None)[::-1]
        selected: list[tuple[int, int, float]] = []
        for flat_idx in flat_indices[:top_k]:
            layer_idx, head_idx = np.unravel_index(flat_idx, importance.shape)
            selected.append((int(layer_idx), int(head_idx), float(importance[layer_idx, head_idx])))
        return selected

    def _compute_top_head_group_stats(
        self,
        samples: list[dict],
        layer_idx: int,
        head_idx: int,
    ) -> dict[str, float]:
        stats: dict[str, list[float]] = {group: [] for group in self.token_groups}
        for sample in samples:
            attention = self._sample_to_attention_matrix(sample)[layer_idx, head_idx]
            for group in self.token_groups:
                mask = sample["token_groups"][group].numpy().astype(bool)
                # We summarize a head by how much last-token attention mass it allocates
                # to each token group, not by per-token distributions.
                value = float(attention[mask].sum()) if np.any(mask) else 0.0
                stats[group].append(value)
        return {
            group: float(np.mean(values)) if values else 0.0
            for group, values in stats.items()
        }

    def _build_summary_heatmap_labels(
        self,
        top_heads: list[tuple[int, int, float]],
    ) -> list[tuple[int, int, str]]:
        return [(layer_idx, head_idx, str(rank)) for rank, (layer_idx, head_idx, _) in enumerate(top_heads, start=1)]

    def run(self) -> None:
        if self.bundle is None:
            self.setup()
        assert self.bundle is not None

        results: dict[str, dict] = {}
        summary_rows: list[dict] = []
        group_heatmaps: dict[str, dict[str, list[np.ndarray]]] = {
            "affective": {"positive": [], "negative": [], "diff": []},
            "non_affective": {"positive": [], "negative": [], "diff": []},
        }

        for concept_idx, concept in enumerate(self.concepts):
            csv_path = resolve_csv_path(self.dataset_root, concept.csv_stem)
            records = load_records(csv_path)
            positives, negatives = self._sample_balanced_records(records, self.num_samples, self.seed + concept_idx)
            usable = min(len(positives), len(negatives))
            positives = positives[:usable]
            negatives = negatives[:usable]
            prompt = self._build_prompt(concept.name)

            positive_samples: list[dict] = []
            negative_samples: list[dict] = []
            positive_values: list[np.ndarray] = []
            negative_values: list[np.ndarray] = []
            valid_pairs = 0

            progress = tqdm(range(usable), desc=f"attention-{concept.name}", unit="pair")
            for pair_idx in progress:
                positive_path = resolve_image_path(self.dataset_root, positives[pair_idx]["image_path"])
                negative_path = resolve_image_path(self.dataset_root, negatives[pair_idx]["image_path"])
                if not positive_path.exists() or not negative_path.exists():
                    self.log(f"Skipping missing pair ({positive_path}, {negative_path})")
                    continue

                try:
                    positive_inputs = self._prepare_inputs(prompt, positive_path)
                    negative_inputs = self._prepare_inputs(prompt, negative_path)
                    positive_info = self._extract_attention_info(positive_inputs, negative_inputs)
                    negative_info = self._extract_attention_info(negative_inputs, positive_inputs)
                except Exception as exc:
                    self.log(f"Skipping attention pair for {concept.name}: {exc}")
                    self._empty_cache()
                    continue

                positive_info["token_groups"] = self._identify_token_groups(
                    positive_info["input_ids"],
                    positive_info["image_mask"],
                    concept.name,
                )
                negative_info["token_groups"] = self._identify_token_groups(
                    negative_info["input_ids"],
                    negative_info["image_mask"],
                    concept.name,
                )

                positive_samples.append(positive_info)
                negative_samples.append(negative_info)
                positive_values.append(self._sample_to_weighted_matrix(positive_info))
                negative_values.append(self._sample_to_weighted_matrix(negative_info))
                valid_pairs += 1
                self._empty_cache()

            if not positive_samples or not negative_samples:
                self.log(f"No valid samples collected for concept {concept.name}")
                continue

            all_samples = positive_samples + negative_samples
            positive_norm = self._compute_mean_weighted_value_norm(positive_samples)
            negative_norm = self._compute_mean_weighted_value_norm(negative_samples)
            diff_norm = self._compute_head_importance(positive_values, negative_values)
            group_heatmaps[concept.group]["positive"].append(positive_norm)
            group_heatmaps[concept.group]["negative"].append(negative_norm)
            group_heatmaps[concept.group]["diff"].append(diff_norm)
            top_heads = self._select_top_heads(diff_norm, min(self.top_k_heads, diff_norm.size))
            attention_score_allocation = self._compute_layer_attention_allocation(all_samples)
            weighted_value_norm_allocation = self._compute_weighted_value_norm_allocation(all_samples)
            layer_indices = list(range(diff_norm.shape[0]))

            attention_alloc_dir = self._plot_dir("attention_allocation")
            head_importance_dir = self._plot_dir("head_importance")
            top_heads_dir = self._plot_dir("top_heads_detail", concept.name)

            plot_attention_allocation(
                layer_indices=layer_indices,
                allocation_summary=attention_score_allocation,
                concept_name=concept.name,
                save_path=attention_alloc_dir / self.plot_filename(f"{concept.name}_attention_score_allocation"),
                title="Layer-wise Attention Allocation",
                ylabel="Mean Attention Weight",
            )
            plot_attention_allocation(
                layer_indices=layer_indices,
                allocation_summary=weighted_value_norm_allocation,
                concept_name=concept.name,
                save_path=attention_alloc_dir / self.plot_filename(f"{concept.name}_weighted_value_norm_allocation"),
                title="Layer-wise Weighted Value Norm",
                ylabel="Mean Weighted Value L2 Norm",
            )
            plot_head_importance_heatmap(
                importance=positive_norm,
                save_path=head_importance_dir / self.plot_filename(f"{concept.name}_positive_weighted_value_norm_heatmap"),
                title=f"Positive Weighted Value Norm: {concept.name}",
                colorbar_label="Mean L2 Norm",
            )
            plot_head_importance_heatmap(
                importance=negative_norm,
                save_path=head_importance_dir / self.plot_filename(f"{concept.name}_negative_weighted_value_norm_heatmap"),
                title=f"Negative Weighted Value Norm: {concept.name}",
                colorbar_label="Mean L2 Norm",
            )
            plot_head_importance_heatmap(
                importance=diff_norm,
                save_path=head_importance_dir / self.plot_filename(f"{concept.name}_diff_weighted_value_norm_heatmap"),
                title=f"Positive vs Negative Diff Norm: {concept.name}",
                colorbar_label="Diff L2 Norm",
            )
            plot_head_importance_heatmap(
                importance=diff_norm,
                save_path=top_heads_dir / self.plot_filename("top_head_summary"),
                title=f"Top Heads Summary: {concept.name}",
                highlights=self._build_summary_heatmap_labels(top_heads),
                colorbar_label="Diff L2 Norm",
            )

            top_head_details: list[dict] = []
            for layer_idx, head_idx, score in top_heads:
                positive_group_stats = self._compute_top_head_group_stats(positive_samples, layer_idx, head_idx)
                negative_group_stats = self._compute_top_head_group_stats(negative_samples, layer_idx, head_idx)
                plot_top_head_detail(
                    group_names=self.token_groups,
                    positive_values=[positive_group_stats[group] for group in self.token_groups],
                    negative_values=[negative_group_stats[group] for group in self.token_groups],
                    concept_name=concept.name,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    save_path=top_heads_dir / self.plot_filename(f"top_head_L{layer_idx}_H{head_idx}"),
                )
                top_head_details.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "importance": score,
                        "positive_group_attention": positive_group_stats,
                        "negative_group_attention": negative_group_stats,
                    }
                )

            concept_result = {
                "group": concept.group,
                "csv_stem": concept.csv_stem,
                "selected_pairs": usable,
                "valid_pairs": valid_pairs,
                "num_layers": int(diff_norm.shape[0]),
                "num_heads": int(diff_norm.shape[1]),
                "attention_score_allocation": {
                    key: [float(value) for value in values]
                    for key, values in attention_score_allocation.items()
                },
                "weighted_value_norm_allocation": {
                    key: [float(value) for value in values]
                    for key, values in weighted_value_norm_allocation.items()
                },
                "positive_weighted_value_norm": positive_norm.tolist(),
                "negative_weighted_value_norm": negative_norm.tolist(),
                "diff_weighted_value_norm": diff_norm.tolist(),
                "head_importance": diff_norm.tolist(),
                "top_heads": top_head_details,
            }
            results[concept.name] = concept_result
            self.save_json(concept_result, f"{concept.name}_results.json")
            concept_rows: list[dict] = []

            for layer_idx in layer_indices:
                concept_rows.extend(
                    [
                        {
                            "view": "attention_score_allocation",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": layer_idx,
                            "head": "",
                            "series": "image",
                            "value": attention_score_allocation["image"][layer_idx],
                        },
                        {
                            "view": "attention_score_allocation",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": layer_idx,
                            "head": "",
                            "series": "text",
                            "value": attention_score_allocation["text"][layer_idx],
                        },
                        {
                            "view": "weighted_value_norm_allocation",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": layer_idx,
                            "head": "",
                            "series": "image",
                            "value": weighted_value_norm_allocation["image"][layer_idx],
                        },
                        {
                            "view": "weighted_value_norm_allocation",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": layer_idx,
                            "head": "",
                            "series": "text",
                            "value": weighted_value_norm_allocation["text"][layer_idx],
                        },
                    ]
                )
                for head_idx in range(diff_norm.shape[1]):
                    concept_rows.append(
                        {
                            "view": "head_positive_weighted_value_norm",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": layer_idx,
                            "head": head_idx,
                            "series": "positive",
                            "value": positive_norm[layer_idx, head_idx],
                        }
                    )
                    concept_rows.append(
                        {
                            "view": "head_negative_weighted_value_norm",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": layer_idx,
                            "head": head_idx,
                            "series": "negative",
                            "value": negative_norm[layer_idx, head_idx],
                        }
                    )
                    concept_rows.append(
                        {
                            "view": "head_diff_weighted_value_norm",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": layer_idx,
                            "head": head_idx,
                            "series": "diff",
                            "value": diff_norm[layer_idx, head_idx],
                        }
                    )

            for detail in top_head_details:
                for group_name in self.token_groups:
                    concept_rows.append(
                        {
                            "view": "top_head_detail",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": detail["layer"],
                            "head": detail["head"],
                            "series": f"positive_{group_name}",
                            "value": detail["positive_group_attention"][group_name],
                        }
                    )
                    concept_rows.append(
                        {
                            "view": "top_head_detail",
                            "concept": concept.name,
                            "group": concept.group,
                            "layer": detail["layer"],
                            "head": detail["head"],
                            "series": f"negative_{group_name}",
                            "value": detail["negative_group_attention"][group_name],
                        }
                    )

            summary_rows.extend(concept_rows)
            self.save_csv(concept_rows, f"{concept.name}_results.csv")
            self.log(f"Finished attention analysis for concept {concept.name}")

        if not results:
            raise RuntimeError(
                "Attention analysis produced zero valid concepts. "
                "Check the experiment log for attention-extraction failures."
            )

        for group_name, group_payload in group_heatmaps.items():
            for stat_name, matrices in group_payload.items():
                if not matrices:
                    continue
                group_mean = np.mean(np.stack(matrices, axis=0), axis=0)
                plot_head_importance_heatmap(
                    importance=group_mean,
                    save_path=self._plot_dir("head_importance") / self.plot_filename(f"{group_name}_{stat_name}_weighted_value_norm_heatmap"),
                    title=f"{group_name.replace('_', ' ').title()} {stat_name.title()} Weighted Value Norm",
                    colorbar_label="Mean L2 Norm" if stat_name in {"positive", "negative"} else "Diff L2 Norm",
                )

        summary_payload = {
            "meta": {
                "model_path": self.config["model"]["path"],
                "prompt": self.prompt_template,
                "top_k_heads": self.top_k_heads,
                "token_groups": self.token_groups,
                "num_samples_per_class": self.num_samples,
                "seed": self.seed,
            },
            "concept_order": [concept.name for concept in self.concepts if concept.name in results],
            "results": results,
        }
        self.save_json(summary_payload, "summary.json")
        self.save_csv(summary_rows, "summary.csv")
