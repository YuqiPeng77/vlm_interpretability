from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.base import BaseExperiment
from shared.data_loader import (
    limit_records,
    load_records,
    parse_concept_specs,
    resolve_csv_path,
)
from shared.feature_extraction import build_image_chat_prompt, collect_component_features
from shared.metrics import compute_pca_projection
from shared.model_loader import ModelBundle, load_model_bundle
from shared.visualizer import get_pca_axis_limits, plot_pca_scatter_grid, plot_single_pca_scatter


def build_prompt(processor, pca_config: dict) -> str:
    return build_image_chat_prompt(
        processor,
        pca_config.get("prompt"),
        "pca_visualization.prompt",
    )


class PCAVisualizationExperiment(BaseExperiment):
    def __init__(self, config: dict, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self.bundle: ModelBundle | None = None
        self.dataset_root = Path(config["dataset"]["root"])
        self.concepts = parse_concept_specs(config["dataset"], require_group=True)
        self.pca_config = config["pca_visualization"]
        self.runtime_config = config.get("runtime", {})
        self.components = self.pca_config.get("components", ["encoder"])
        self.max_samples = int(self.config["dataset"].get("max_samples_per_attr", 0))
        self.selected_layers: dict[str, list[int]] = {}

    def setup(self) -> None:
        invalid_components = sorted(set(self.components) - {"encoder", "decoder"})
        if invalid_components:
            raise ValueError(f"Unsupported pca_visualization components: {invalid_components}")
        if not self.pca_config.get("prompt"):
            raise ValueError("pca_visualization.prompt is required and must be a non-empty string.")

        self.bundle = load_model_bundle(self.config["model"])
        self.selected_layers = self._parse_selected_layers()
        self.log(
            "Loaded model with "
            f"{self.bundle.num_enc_layers} encoder blocks and {self.bundle.num_dec_layers} decoder layers."
        )

    def _parse_selected_layers(self) -> dict[str, list[int]]:
        raw_selected_layers = self.pca_config.get("selected_layers")
        if not isinstance(raw_selected_layers, dict):
            raise ValueError("pca_visualization.selected_layers must be a mapping from component to layer list.")

        assert self.bundle is not None
        parsed: dict[str, list[int]] = {}
        for component in self.components:
            raw_layers = raw_selected_layers.get(component)
            if not isinstance(raw_layers, list) or not raw_layers:
                raise ValueError(
                    f"pca_visualization.selected_layers.{component} must be a non-empty list of layer indices."
                )

            max_layer = self.bundle.num_enc_layers - 1 if component == "encoder" else self.bundle.num_dec_layers - 1
            if any(str(layer_idx).strip().lower() == "all" for layer_idx in raw_layers):
                parsed[component] = list(range(-1, max_layer + 1))
                continue

            normalized = sorted({int(layer_idx) for layer_idx in raw_layers})
            invalid = [layer_idx for layer_idx in normalized if layer_idx < -1 or layer_idx > max_layer]
            if invalid:
                raise ValueError(
                    f"Invalid {component} PCA layers {invalid}; expected integers in [-1, {max_layer}]."
                )
            parsed[component] = normalized
        return parsed

    def _model_display_name(self) -> str:
        model_name = self.config.get("model", {}).get("name")
        if model_name:
            return str(model_name)
        return Path(self.config["model"]["path"]).name

    def _layer_filename_suffix(self, layer_idx: int) -> str:
        return "input" if layer_idx < 0 else f"L{layer_idx}"

    def run(self) -> None:
        if self.bundle is None:
            self.setup()
        assert self.bundle is not None

        prompt = build_prompt(self.bundle.processor, self.pca_config)
        summary_results: dict[str, dict] = {}
        plot_dir = self.plots_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

        for concept_idx, concept in enumerate(self.concepts):
            csv_path = resolve_csv_path(self.dataset_root, concept.csv_stem)
            records = load_records(csv_path)
            seed = int(self.runtime_config.get("seed", 42)) + concept_idx
            records = limit_records(records, self.max_samples, seed)
            collected = collect_component_features(
                bundle=self.bundle,
                dataset_root=self.dataset_root,
                records=records,
                prompt=prompt,
                components=self.components,
                selected_layers=self.selected_layers,
                logger=self.log,
                progress_desc=f"pca-{concept.name}",
            )
            labels_array = collected["labels"]
            if labels_array.size == 0 or len(np.unique(labels_array)) < 2:
                raise RuntimeError(
                    f"Concept {concept.name!r} does not have enough valid positive/negative samples for PCA."
                )

            concept_result = {
                "group": concept.group,
                "csv_file": csv_path.name,
                "num_samples_used": int(labels_array.size),
                "num_skipped": int(collected["skipped"]),
                "label_counts": {
                    "negative": int(np.sum(labels_array == 0)),
                    "positive": int(np.sum(labels_array == 1)),
                },
            }
            point_rows: list[dict] = []

            for component in self.components:
                component_result = {
                    "selected_layers": self.selected_layers[component],
                    "layers": {},
                }
                plot_payloads: list[dict] = []
                concept_plot_dir = plot_dir / concept.name
                component_plot_dir = concept_plot_dir / component
                concept_plot_dir.mkdir(parents=True, exist_ok=True)
                component_plot_dir.mkdir(parents=True, exist_ok=True)

                for layer_idx in self.selected_layers[component]:
                    projected, explained_variance = compute_pca_projection(
                        collected["feature_bank"][component][layer_idx],
                        seed=seed + max(layer_idx, 0),
                    )
                    layer_key = str(layer_idx)
                    component_result["layers"][layer_key] = {
                        "num_samples": int(projected.shape[0]),
                        "num_positive": int(np.sum(labels_array == 1)),
                        "num_negative": int(np.sum(labels_array == 0)),
                        "explained_variance_ratio": explained_variance,
                        "explained_variance_top2_sum": float(sum(explained_variance)),
                    }
                    plot_payloads.append(
                        {
                            "layer": layer_idx,
                            "points": projected,
                            "labels": labels_array,
                            "explained_variance_ratio": explained_variance,
                        }
                    )

                    for sample_idx, point in enumerate(projected):
                        point_rows.append(
                            {
                                "concept": concept.name,
                                "group": concept.group,
                                "component": component,
                                "layer": layer_idx,
                                "image_path": collected["image_paths"][sample_idx],
                                "attribute_label": int(labels_array[sample_idx]),
                                "class_name": "positive" if int(labels_array[sample_idx]) == 1 else "negative",
                                "pc1": float(point[0]),
                                "pc2": float(point[1]),
                                "explained_variance_ratio_1": explained_variance[0],
                                "explained_variance_ratio_2": explained_variance[1],
                            }
                        )

                concept_result[component] = component_result
                x_limits, y_limits = get_pca_axis_limits(plot_payloads)
                for payload in plot_payloads:
                    layer_idx = int(payload["layer"])
                    plot_single_pca_scatter(
                        payload,
                        concept_name=concept.name,
                        component=component,
                        model_display_name=self._model_display_name(),
                        output_path=component_plot_dir
                        / self.plot_filename(f"{component}_{self._layer_filename_suffix(layer_idx)}"),
                        x_limits=x_limits,
                        y_limits=y_limits,
                        dpi=int(self.pca_config.get("dpi", 300)),
                    )
                plot_pca_scatter_grid(
                    plot_payloads,
                    concept_name=concept.name,
                    component=component,
                    model_display_name=self._model_display_name(),
                    output_path=concept_plot_dir / self.plot_filename(component),
                    dpi=int(self.pca_config.get("dpi", 300)),
                )

            summary_results[concept.name] = concept_result
            self.save_csv(point_rows, f"{concept.name}_pca_points.csv")
            self.log(f"Finished PCA visualization for concept {concept.name}")

        summary = {
            "meta": {
                "model_path": self.config["model"]["path"],
                "model_display_name": self._model_display_name(),
                "seed": int(self.runtime_config.get("seed", 42)),
                "max_samples_per_attr": self.max_samples,
                "components": self.components,
                "selected_layers": self.selected_layers,
                "prompt": self.pca_config["prompt"],
                "num_encoder_layers": self.bundle.num_enc_layers,
                "num_decoder_layers": self.bundle.num_dec_layers,
                "concept_groups": {
                    "non_affective": [
                        concept.name for concept in self.concepts if concept.group == "non_affective"
                    ],
                    "affective": [concept.name for concept in self.concepts if concept.group == "affective"],
                },
            },
            "concept_order": [concept.name for concept in self.concepts],
            "results": summary_results,
        }
        self.save_json(summary, "summary.json")
        self.log("PCA visualization experiment completed.")
