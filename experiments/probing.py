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
from shared.metrics import (
    train_logistic_probe_with_random_baseline,
)
from shared.model_loader import ModelBundle, load_model_bundle
from shared.visualizer import (
    plot_grouped_probing_accuracy,
    plot_single_attribute_probing_accuracy,
)


def build_prompt(processor, probing_config: dict) -> str:
    return build_image_chat_prompt(processor, probing_config.get("prompt"), "probing.prompt")


class ProbingExperiment(BaseExperiment):
    def __init__(self, config: dict, output_dir: Path) -> None:
        super().__init__(config, output_dir)
        self.bundle: ModelBundle | None = None
        self.dataset_root = Path(config["dataset"]["root"])
        self.concepts = parse_concept_specs(config["dataset"], require_group=True)
        self.probing_config = config["probing"]
        self.runtime_config = config.get("runtime", {})
        self.components = self.probing_config.get("components", ["encoder", "decoder"])
        self.max_samples = int(self.config["dataset"].get("max_samples_per_attr", 0))
        self.test_size = float(self.probing_config.get("test_size", 0.2))

    def setup(self) -> None:
        if self.probing_config.get("probe_type", "logistic") != "logistic":
            raise ValueError("Only probing.probe_type=logistic is supported.")
        invalid_components = sorted(set(self.components) - {"encoder", "decoder"})
        if invalid_components:
            raise ValueError(f"Unsupported probing components: {invalid_components}")

        if not self.probing_config.get("prompt"):
            raise ValueError("probing.prompt is required and must be a non-empty string.")

        self.bundle = load_model_bundle(self.config["model"])
        self.log(
            "Loaded model with "
            f"{self.bundle.num_enc_layers} encoder blocks and {self.bundle.num_dec_layers} decoder layers."
        )

    def _resolved_prompt_text(self) -> str:
        return self.probing_config["prompt"]

    def _model_display_name(self) -> str:
        model_name = self.config.get("model", {}).get("name")
        if model_name:
            return str(model_name)
        return Path(self.config["model"]["path"]).name

    def run(self) -> None:
        if self.bundle is None:
            self.setup()
        assert self.bundle is not None

        prompt = build_prompt(self.bundle.processor, self.probing_config)
        # We probe one extra "input" slot before the first block/layer so the
        # resulting curve can compare raw incoming representations against layer outputs.
        enc_slots = self.bundle.num_enc_layers + 1
        dec_slots = self.bundle.num_dec_layers + 1
        selected_layers = {
            component: (list(range(-1, self.bundle.num_enc_layers)) if component == "encoder" else list(range(-1, self.bundle.num_dec_layers)))
            for component in self.components
        }
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
                progress_desc=f"probing-{concept.name}",
            )
            labels_array = collected["labels"]
            if labels_array.size == 0 or len(np.unique(labels_array)) < 2:
                raise RuntimeError(
                    f"Concept {concept.name!r} does not have enough valid samples for probing."
                )

            concept_result = {
                "group": concept.group,
                "csv_file": csv_path.name,
                "num_samples_used": int(labels_array.size),
                "num_skipped": int(collected["skipped"]),
            }
            summary_rows: list[dict] = []

            if "encoder" in self.components:
                # Each slot gets its own linear probe so the accuracy curve remains
                # interpretable as "how decodable is the concept at this stage".
                # [28, 20, 1152]
                enc_features = [
                    collected["feature_bank"]["encoder"][layer_idx]
                    for layer_idx in selected_layers["encoder"]
                ]
                encoder_acc, encoder_stats = train_logistic_probe_with_random_baseline(
                    enc_features,
                    labels_array,
                    enc_slots,
                    self.test_size,
                    int(self.runtime_config.get("seed", 42)),
                    random_repeats=3,
                )
                concept_result["encoder_layer_accuracy"] = encoder_acc
                concept_result["encoder_random_label_runs"] = [
                    stat["random_accuracy_runs"] for stat in encoder_stats
                ]
                concept_result["encoder_random_label_mean"] = [
                    stat["random_accuracy_mean"] for stat in encoder_stats
                ]
                concept_result["encoder_random_label_std"] = [
                    stat["random_accuracy_std"] for stat in encoder_stats
                ]
                concept_result["encoder_stats"] = encoder_stats
                summary_rows.extend(
                    {
                        "concept": concept.name,
                        "component": "encoder",
                        "slot": stat["slot"],
                        "real_accuracy": stat["accuracy"],
                        "random_accuracy_mean": stat["random_accuracy_mean"],
                        "random_accuracy_std": stat["random_accuracy_std"],
                        "n_train": stat["n_train"],
                        "n_test": stat["n_test"],
                    }
                    for stat in encoder_stats
                )

            if "decoder" in self.components:
                # [37, 20, 4096]
                dec_features = [
                    collected["feature_bank"]["decoder"][layer_idx]
                    for layer_idx in selected_layers["decoder"]
                ]
                decoder_acc, decoder_stats = train_logistic_probe_with_random_baseline(
                    dec_features,
                    labels_array,
                    dec_slots,
                    self.test_size,
                    int(self.runtime_config.get("seed", 42)),
                    random_repeats=3,
                )
                concept_result["decoder_layer_accuracy"] = decoder_acc
                concept_result["decoder_random_label_runs"] = [
                    stat["random_accuracy_runs"] for stat in decoder_stats
                ]
                concept_result["decoder_random_label_mean"] = [
                    stat["random_accuracy_mean"] for stat in decoder_stats
                ]
                concept_result["decoder_random_label_std"] = [
                    stat["random_accuracy_std"] for stat in decoder_stats
                ]
                concept_result["decoder_stats"] = decoder_stats
                summary_rows.extend(
                    {
                        "concept": concept.name,
                        "component": "decoder",
                        "slot": stat["slot"],
                        "real_accuracy": stat["accuracy"],
                        "random_accuracy_mean": stat["random_accuracy_mean"],
                        "random_accuracy_std": stat["random_accuracy_std"],
                        "n_train": stat["n_train"],
                        "n_test": stat["n_test"],
                    }
                    for stat in decoder_stats
                )

            summary_results[concept.name] = concept_result
            self.save_csv(summary_rows, f"{concept.name}_probing_stats.csv")
            self.log(f"Finished probing for concept {concept.name}")

        summary = {
            "meta": {
                "model_path": self.config["model"]["path"],
                "seed": int(self.runtime_config.get("seed", 42)),
                "test_size": self.test_size,
                "max_samples_per_attr": self.max_samples,
                "model_display_name": self._model_display_name(),
                "random_label_repeats": 3,
                "concept_groups": {
                    "non_affective": [concept.name for concept in self.concepts if concept.group == "non_affective"],
                    "affective": [concept.name for concept in self.concepts if concept.group == "affective"],
                },
                "num_encoder_layers": self.bundle.num_enc_layers,
                "num_decoder_layers": self.bundle.num_dec_layers,
                "prompt_mode": self.probing_config.get("prompt_mode", "neutral"),
                "prompt": self._resolved_prompt_text(),
            },
            "concept_order": [concept.name for concept in self.concepts],
            "results": summary_results,
        }
        self.save_json(summary, "summary.json")

        if "encoder" in self.components:
            plot_grouped_probing_accuracy(
                summary,
                component="encoder",
                output_path=self.plots_dir / self.plot_filename("encoder"),
                dpi=int(self.probing_config.get("dpi", 300)),
            )
        if "decoder" in self.components:
            plot_grouped_probing_accuracy(
                summary,
                component="decoder",
                output_path=self.plots_dir / self.plot_filename("decoder"),
                dpi=int(self.probing_config.get("dpi", 300)),
            )
        for concept in self.concepts:
            if concept.name not in summary_results:
                continue
            concept_plot_dir = self.plots_dir / concept.name
            concept_plot_dir.mkdir(parents=True, exist_ok=True)
            if "encoder" in self.components:
                plot_single_attribute_probing_accuracy(
                    summary,
                    concept_name=concept.name,
                    component="encoder",
                    output_path=concept_plot_dir / self.plot_filename("encoder"),
                    dpi=int(self.probing_config.get("dpi", 300)),
                )
            if "decoder" in self.components:
                plot_single_attribute_probing_accuracy(
                    summary,
                    concept_name=concept.name,
                    component="decoder",
                    output_path=concept_plot_dir / self.plot_filename("decoder"),
                    dpi=int(self.probing_config.get("dpi", 300)),
                )
        self.log("Probing experiment completed.")
