from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DTYPE_ALIASES = {
    "auto": "auto",
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class ModelBundle:
    model: Any
    processor: Any
    encoder_blocks: Sequence[Any]
    decoder_layers: Sequence[Any]
    num_enc_layers: int
    num_dec_layers: int
    input_device: torch.device


def resolve_torch_dtype(dtype_name: str | None) -> Any:
    if dtype_name is None:
        return torch.bfloat16
    key = str(dtype_name).lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_ALIASES[key]


def find_encoder_blocks(model: Any) -> Sequence[Any]:
    candidates = [
        ("model.visual.blocks", lambda m: m.model.visual.blocks),
        ("visual.blocks", lambda m: m.visual.blocks),
    ]
    for _label, getter in candidates:
        try:
            blocks = getter(model)
        except AttributeError:
            continue
        if blocks is not None:
            return blocks
    raise RuntimeError("Cannot locate encoder blocks on the loaded model.")


def find_decoder_layers(model: Any) -> Sequence[Any]:
    candidates = [
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
    ]
    for _label, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None:
            return layers
    raise RuntimeError("Cannot locate decoder layers on the loaded model.")


def load_model_bundle(model_config: dict) -> ModelBundle:
    model_path = model_config["path"]
    device_map = model_config.get("device_map", "auto")
    torch_dtype = resolve_torch_dtype(model_config.get("dtype", "bfloat16"))

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    ).eval()

    encoder_blocks = find_encoder_blocks(model)
    decoder_layers = find_decoder_layers(model)
    input_device = model.get_input_embeddings().weight.device

    return ModelBundle(
        model=model,
        processor=processor,
        encoder_blocks=encoder_blocks,
        decoder_layers=decoder_layers,
        num_enc_layers=len(encoder_blocks),
        num_dec_layers=len(decoder_layers),
        input_device=input_device,
    )
