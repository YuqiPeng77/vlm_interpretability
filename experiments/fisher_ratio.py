from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.base import BaseExperiment
from shared.data_loader import limit_records, load_records, parse_concept_specs, resolve_csv_path
from shared.feature_extraction import build_image_chat_prompt, collect_component_features
from shared.metrics import compute_fisher_ratio
from shared.model_loader import ModelBundle, load_model_bundle
from shared.visualizer import (
    plot_fisher_ratio_all_attributes,
    plot_single_fisher_ratio_attribute,
)


def build_prompt(processor, fisher_ratio_config: dict) -> str:
    return build_image_chat_prompt(
        processor,
        fisher_ratio_config.get("prompt"),
        "fisher_ratio.prompt",
    )


class FisherRatioExperiment(BaseExperiment):
    def __init__(self, config: dict, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self.bundle: ModelBundle | None = None
        self.dataset_root = Path(config["dataset"]["root"])
        self.concepts = parse_concept_specs(config["dataset"], require_group=True)
        self.fisher_ratio_config = config["fisher_ratio"]
        self.runtime_config = config.get("runtime", {})
        self.components = self.fisher_ratio_config.get("components", ["encoder", "decoder"])
        self.max_samples = int(self.config["dataset"].get("max_samples_per_attr", 0))

    def setup(self) -> None:
        if not self.fisher_ratio_config.get("prompt"):
            raise ValueError("fisher_ratio.prompt is required and must be a non-empty string.")
        invalid_components = sorted(set(self.components) - {"encoder", "decoder"})
        if invalid_components:
            raise ValueError(f"Unsupported fisher_ratio components: {invalid_components}")

        self.bundle = load_model_bundle(self.config["model"])
        self.log(
            "Loaded model with "
            f"{self.bundle.num_enc_layers} encoder blocks and {self.bundle.num_dec_layers} decoder layers."
        )

    def _model_display_name(self) -> str:
        model_name = self.config.get("model", {}).get("name")
        if model_name:
            return str(model_name)
        return Path(self.config["model"]["path"]).name

    def run(self) -> None:
        if self.bundle is None:
            self.setup()
        assert self.bundle is not None

        prompt = build_prompt(self.bundle.processor, self.fisher_ratio_config)
        selected_layers = {
            component: (
                list(range(-1, self.bundle.num_enc_layers))
                if component == "encoder"
                else list(range(-1, self.bundle.num_dec_layers))
            )
            for component in self.components
        }
        epsilon = float(self.fisher_ratio_config.get("epsilon", 1e-10))
        dpi = int(self.fisher_ratio_config.get("dpi", 300))
        csv_rows: list[dict] = []
        summary_results: dict[str, dict] = {}

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
                selected_layers=selected_layers,
                logger=self.log,
                progress_desc=f"fisher-{concept.name}",
            )
            labels_array = collected["labels"]
            if labels_array.size == 0 or len(np.unique(labels_array)) < 2:
                raise RuntimeError(
                    f"Concept {concept.name!r} does not have enough valid positive/negative samples for Fisher ratio."
                )

            positive_mask = labels_array == 1
            negative_mask = labels_array == 0
            if not np.any(positive_mask) or not np.any(negative_mask):
                raise RuntimeError(
                    f"Concept {concept.name!r} requires both positive and negative samples for Fisher ratio."
                )

            concept_result = {
                "group": concept.group,
                "csv_file": csv_path.name,
                "num_samples_used": int(labels_array.size),
                "num_skipped": int(collected["skipped"]),
                "label_counts": {
                    "negative": int(np.sum(negative_mask)),
                    "positive": int(np.sum(positive_mask)),
                },
            }
            for component in self.components:
                component_layers: dict[str, dict] = {}
                fisher_curve: list[float] = []
                between_curve: list[float] = []
                within_curve: list[float] = []

                for layer_idx in selected_layers[component]:
                    feature_matrix = np.stack(collected["feature_bank"][component][layer_idx], axis=0)
                    stats = compute_fisher_ratio(
                        positive_features=feature_matrix[positive_mask],
                        negative_features=feature_matrix[negative_mask],
                        epsilon=epsilon,
                    )
                    component_layers[str(layer_idx)] = stats
                    fisher_curve.append(stats["fisher_ratio"])
                    between_curve.append(stats["between_class_variance"])
                    within_curve.append(stats["within_class_variance"])
                    csv_rows.append(
                        {
                            "concept": concept.name,
                            "group": concept.group,
                            "component": component,
                            "layer": layer_idx,
                            "fisher_ratio": stats["fisher_ratio"],
                            "between_class_variance": stats["between_class_variance"],
                            "within_class_variance": stats["within_class_variance"],
                            "num_positive": stats["num_positive"],
                            "num_negative": stats["num_negative"],
                        }
                    )

                concept_result[component] = {
                    "layers": component_layers,
                }
                concept_result[f"{component}_layer_fisher_ratio"] = fisher_curve
                concept_result[f"{component}_between_class_variance"] = between_curve
                concept_result[f"{component}_within_class_variance"] = within_curve

            summary_results[concept.name] = concept_result
            self.log(f"Finished Fisher ratio for concept {concept.name}")

        summary = {
            "meta": {
                "model_path": self.config["model"]["path"],
                "model_display_name": self._model_display_name(),
                "seed": int(self.runtime_config.get("seed", 42)),
                "max_samples_per_attr": self.max_samples,
                "prompt": self.fisher_ratio_config["prompt"],
                "epsilon": epsilon,
                "components": list(self.components),
                "num_encoder_layers": self.bundle.num_enc_layers,
                "num_decoder_layers": self.bundle.num_dec_layers,
                "concept_groups": {
                    "non_affective": [concept.name for concept in self.concepts if concept.group == "non_affective"],
                    "affective": [concept.name for concept in self.concepts if concept.group == "affective"],
                },
            },
            "concept_order": [concept.name for concept in self.concepts],
            "results": summary_results,
        }
        self.save_json(summary, "summary.json")
        self.save_csv(csv_rows, "fisher_ratio_by_layer.csv")

        for concept in self.concepts:
            if concept.name not in summary_results:
                continue
            concept_plot_dir = self.plots_dir / concept.name
            concept_plot_dir.mkdir(parents=True, exist_ok=True)
            for component in self.components:
                plot_single_fisher_ratio_attribute(
                    summary=summary,
                    concept_name=concept.name,
                    component=component,
                    output_path=concept_plot_dir / self.plot_filename(component),
                    dpi=dpi,
                )

        for component in self.components:
            plot_fisher_ratio_all_attributes(
                summary=summary,
                component=component,
                output_path=self.plots_dir / self.plot_filename(component),
                dpi=dpi,
            )
        self.log("Fisher ratio experiment completed.")
