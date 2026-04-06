from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from shared.data_loader import load_image_safe, resolve_image_path
from shared.hook_manager import HookManager
from shared.metrics import extract_decoder_feature, extract_encoder_feature
from shared.model_loader import ModelBundle


def build_image_chat_prompt(processor, prompt_text: str, config_key: str) -> str:
    if not prompt_text:
        raise ValueError(f"{config_key} is required and must be a non-empty string.")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def prepare_inputs(
    bundle: ModelBundle,
    image_path: Path,
    prompt: str,
    logger: Callable[[str], None] | None = None,
) -> dict:
    image = load_image_safe(image_path, logger=logger)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    inputs = bundle.processor(images=image, text=prompt, return_tensors="pt")
    return {key: value.to(bundle.input_device) for key, value in inputs.items()}


def empty_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_all_component_layers(bundle: ModelBundle, component: str) -> list[int]:
    if component == "encoder":
        return list(range(-1, bundle.num_enc_layers))
    if component == "decoder":
        return list(range(-1, bundle.num_dec_layers))
    raise ValueError(f"Unsupported component: {component}")


def validate_component_hook_count(hooks: HookManager, bundle: ModelBundle, component: str) -> None:
    if component == "encoder":
        expected_count = bundle.num_enc_layers + 1
        actual_count = len([key for key in hooks.hidden_cache if isinstance(key, tuple) and key[0] == "enc"])
    elif component == "decoder":
        expected_count = bundle.num_dec_layers + 1
        actual_count = len([key for key in hooks.hidden_cache if isinstance(key, tuple) and key[0] == "dec"])
    else:
        raise ValueError(f"Unsupported component: {component}")

    if actual_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {component} hooks, but collected {actual_count} activations."
        )


def collect_component_features(
    bundle: ModelBundle,
    dataset_root: Path,
    records: list[dict],
    prompt: str,
    components: list[str],
    selected_layers: dict[str, list[int]] | None,
    logger: Callable[[str], None],
    progress_desc: str,
) -> dict:
    normalized_layers = {
        component: (selected_layers or {}).get(component, get_all_component_layers(bundle, component))
        for component in components
    }
    feature_bank: dict[str, dict[int, list[np.ndarray]]] = {
        component: {layer_idx: [] for layer_idx in normalized_layers[component]}
        for component in components
    }
    labels: list[int] = []
    image_paths: list[str] = []
    skipped = 0

    progress = tqdm(records, desc=progress_desc, unit="img")
    for record in progress:
        image_path = resolve_image_path(dataset_root, record["image_path"])
        if not image_path.exists():
            skipped += 1
            logger(f"Skipping missing image {image_path}")
            continue

        try:
            inputs = prepare_inputs(bundle, image_path, prompt, logger=logger)
        except Exception as exc:
            skipped += 1
            logger(f"Skipping image {image_path}: {exc}")
            continue

        try:
            with HookManager() as hooks:
                if "encoder" in components:
                    hooks.register_encoder_probing_hooks(bundle.encoder_blocks)
                if "decoder" in components:
                    hooks.register_decoder_probing_hooks(bundle.decoder_layers)

                with torch.no_grad():
                    _ = bundle.model(**inputs, return_dict=True, use_cache=False)

                for component in components:
                    validate_component_hook_count(hooks, bundle, component)
                    extractor = extract_encoder_feature if component == "encoder" else extract_decoder_feature
                    for layer_idx in normalized_layers[component]:
                        feature_bank[component][layer_idx].append(
                            extractor(hooks.hidden_cache, layer_idx)
                        )
        except Exception as exc:
            skipped += 1
            logger(f"Skipping image {image_path}: {exc}")
            continue

        labels.append(int(record["attribute_label"]))
        image_paths.append(record["image_path"])
        empty_torch_cache()

    return {
        "feature_bank": feature_bank,
        "labels": np.array(labels, dtype=int),
        "image_paths": image_paths,
        "skipped": skipped,
    }
