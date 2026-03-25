from __future__ import annotations

from typing import Any

import torch


class HookManager:
    def __init__(self) -> None:
        self.handles: list[Any] = []
        self.hidden_cache: dict[tuple[str, int] | int, torch.Tensor] = {}
        self.inputs_cache: dict[int, torch.Tensor] = {}
        self.outputs_cache: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.clear()

    def add(self, handle: Any) -> None:
        self.handles.append(handle)

    def register_encoder_probing_hooks(self, encoder_blocks) -> None:
        def pre_hook(_module, inputs):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            self.hidden_cache[("enc", -1)] = hidden.detach()

        self.add(encoder_blocks[0].register_forward_pre_hook(pre_hook))

        def make_hook(layer_idx: int):
            def _hook(_module, _inputs, outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                self.hidden_cache[("enc", layer_idx)] = hidden.detach()

            return _hook

        for layer_idx, block in enumerate(encoder_blocks):
            self.add(block.register_forward_hook(make_hook(layer_idx)))

    def register_decoder_probing_hooks(self, decoder_layers) -> None:
        def pre_hook(_module, inputs):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            self.hidden_cache[("dec", -1)] = hidden.detach()

        self.add(decoder_layers[0].register_forward_pre_hook(pre_hook))

        def make_hook(layer_idx: int):
            def _hook(_module, _inputs, outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                self.hidden_cache[("dec", layer_idx)] = hidden.detach()

            return _hook

        for layer_idx, layer in enumerate(decoder_layers):
            self.add(layer.register_forward_hook(make_hook(layer_idx)))

    def register_layer_output_hooks(self, layers) -> None:
        def make_hook(layer_idx: int):
            def _hook(_module, _inputs, outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                self.hidden_cache[layer_idx] = hidden.detach().clone()

            return _hook

        for layer_idx, layer in enumerate(layers):
            self.add(layer.register_forward_hook(make_hook(layer_idx)))

    def register_activation_patching_hook(
        self,
        layer,
        mode: str,
        image_mask: torch.Tensor,
        seed: int,
        clean_hidden: torch.Tensor | None = None,
    ) -> None:
        def hook_fn(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            patched = hidden.clone()

            mask = image_mask.to(patched.device)
            if mask.ndim != 1:
                mask = mask.view(-1)
            if mask.numel() != patched.shape[1]:
                resized = torch.zeros(patched.shape[1], dtype=torch.bool, device=patched.device)
                width = min(mask.numel(), patched.shape[1])
                resized[:width] = mask[:width]
                mask = resized
            if int(mask.sum().item()) == 0:
                return outputs

            generator = torch.Generator(device=patched.device)
            generator.manual_seed(seed)

            if mode == "zero":
                patched[:, mask, :] = 0.0
            elif mode == "noise":
                noise = torch.randn(
                    patched[:, mask, :].shape,
                    generator=generator,
                    device=patched.device,
                    dtype=patched.dtype,
                )
                patched[:, mask, :] = noise
            elif mode == "counterfactual":
                if clean_hidden is None:
                    raise ValueError("clean_hidden is required for counterfactual patching.")
                clean = clean_hidden.to(patched.device, patched.dtype)
                if clean.shape[1] != patched.shape[1]:
                    aligned = patched.clone()
                    width = min(clean.shape[1], patched.shape[1])
                    aligned[:, :width, :] = clean[:, :width, :]
                    clean = aligned
                patched[:, mask, :] = clean[:, mask, :]
            else:
                raise ValueError(f"Unknown activation patching mode: {mode}")

            if isinstance(outputs, tuple):
                return (patched,) + outputs[1:]
            return patched

        self.add(layer.register_forward_hook(hook_fn))

    def register_residual_patching_hooks(
        self,
        layer,
        mode: str,
        seed: int,
        clean_delta: torch.Tensor | None = None,
    ) -> None:
        captured: dict[str, torch.Tensor] = {}

        def pre_hook(_module, inputs):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            captured["input"] = hidden.detach().clone()

        def fwd_hook(_module, _inputs, outputs):
            hidden_out = outputs[0] if isinstance(outputs, tuple) else outputs
            layer_input = captured["input"].to(hidden_out.device, hidden_out.dtype)

            if mode == "zero":
                new_out = layer_input
            elif mode == "noise":
                generator = torch.Generator(device=hidden_out.device)
                generator.manual_seed(seed)
                noise = torch.randn(
                    layer_input.shape,
                    generator=generator,
                    device=hidden_out.device,
                    dtype=hidden_out.dtype,
                )
                original_delta = hidden_out - layer_input
                noise = noise * original_delta.norm() / (noise.norm() + 1e-8)
                new_out = layer_input + noise
            elif mode == "counterfactual":
                if clean_delta is None:
                    raise ValueError("clean_delta is required for counterfactual residual patching.")
                delta = clean_delta.to(hidden_out.device, hidden_out.dtype)
                if delta.shape == layer_input.shape:
                    new_out = layer_input + delta
                else:
                    new_out = hidden_out.clone()
                    if delta.ndim == 3 and layer_input.ndim == 3:
                        batch = min(delta.shape[0], layer_input.shape[0])
                        tokens = min(delta.shape[1], layer_input.shape[1])
                        width = min(delta.shape[2], layer_input.shape[2])
                        new_out[:batch, :tokens, :width] = (
                            layer_input[:batch, :tokens, :width] + delta[:batch, :tokens, :width]
                        )
                    elif delta.ndim == 2 and layer_input.ndim == 2:
                        tokens = min(delta.shape[0], layer_input.shape[0])
                        width = min(delta.shape[1], layer_input.shape[1])
                        new_out[:tokens, :width] = (
                            layer_input[:tokens, :width] + delta[:tokens, :width]
                        )
                    else:
                        new_out = layer_input + delta.reshape(layer_input.shape)
            else:
                raise ValueError(f"Unknown residual patching mode: {mode}")

            if isinstance(outputs, tuple):
                return (new_out,) + outputs[1:]
            return new_out

        self.add(layer.register_forward_pre_hook(pre_hook))
        self.add(layer.register_forward_hook(fwd_hook))

    def clear(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.hidden_cache.clear()
        self.inputs_cache.clear()
        self.outputs_cache.clear()
