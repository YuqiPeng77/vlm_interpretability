# VLM Interpretability Infra

This directory contains a config-driven infrastructure for five experiment families on Qwen3-VL:

- `probing`
- `patching`
- `attention_analysis`
- `pca_visualization`
- `fisher_ratio`

`patching` unifies:

- decoder activation patching
- encoder component patching
- decoder component patching

## Layout

- `configs/`: example YAML configs
- `experiments/`: experiment implementations
- `shared/`: reusable loaders, hooks, metrics, and plotting code
- `output/`: timestamped run outputs
- `run_experiment.py`: main dispatcher

## Example Usage

```bash
python run_experiment.py --config configs/probing_concepts.yaml
python run_experiment.py --config configs/patching_decoder_activation.yaml
python run_experiment.py --config configs/patching_encoder_component.yaml
python run_experiment.py --config configs/patching_decoder_component.yaml
python run_experiment.py --config configs/attention_analysis_decoder.yaml
python run_experiment.py --config configs/pca_visualization_concepts.yaml
python run_experiment.py --config configs/fisher_ratio_encoder.yaml
```

## Config Notes

- `dataset.root` is the shared base directory. CSV files are resolved as `dataset.root / "SUN" / "{csv_stem}.csv"`.
- Probing expects explicit grouped concepts under `dataset.concepts`.
- Probing prompt is provided entirely by config through `probing.prompt`.
- Probing now adds a 3-run random-label baseline for both encoder and decoder plots/results.
- `fisher_ratio` reuses the same encoder/decoder hook paths and pooled features as probing, but replaces probe training with layer-wise Fisher ratio statistics.
- `fisher_ratio.components` supports `encoder` and `decoder`, and each selected component includes the input slot `-1` plus all block/layer outputs.
- `fisher_ratio` saves `results/fisher_ratio_by_layer.csv`, `results/summary.json`, grouped plots at `plots/{encoder|decoder}.{png|pdf}`, and per-concept plots at `plots/{concept}/{encoder|decoder}.{png|pdf}`.
- `pca_visualization` reuses the same concept CSV structure as probing, but projects selected encoder/decoder layers into 2D PCA space and saves per-concept scatter grids plus PCA coordinates in CSV form.
- `pca_visualization.selected_layers` is a per-component mapping such as `encoder: [0, 5, 10, 15, 20, 26]`; layer index `-1` is supported to visualize the input representation before the first block/layer, and `encoder: [all]` / `decoder: [all]` expands to `[-1, 0, ..., last_layer]`.
- `pca_visualization` saves a combined grid under `plots/pca/{concept}/{component}_pca.{png|pdf}` and per-layer plots under `plots/pca/{concept}/{component}/{component}_L{layer}.{png|pdf}` (or `{component}_input` for layer `-1`).
- `patching` keeps the original validated combinations:
  - `method=activation, stage=decoder`
  - `method=component, stage=encoder`
  - `method=component, stage=decoder`
- Patching metrics are config-driven through `patching.metrics`; v1 supports `probability_change`.
- Component patching uses `patching.selected_layers`, `patching.components`, `patching.attention_submodules`, and `patching.levels`.
- Component patching uses three reusable analysis levels:
  - level 1: block residual baseline
  - level 2: attention vs MLP decomposition
  - level 3: attention submodule decomposition with `q`, `k`, and `v` curves plus a full-attention reference
- For encoder level 3, `q`, `k`, and `v` are obtained by splitting the fused ViT `attn.qkv` output along the hidden dimension.
- For decoder level 3, `q`, `k`, and `v` are patched from the decoder attention's direct projection modules such as `q_proj`, `k_proj`, and `v_proj`.
- `proj` is still supported by the internal patching cache, but it is not shown in level 3 outputs because it is numerically identical to the full-attention reference in the current Qwen3-VL attention implementation.
- Component patching saves plots under metric-specific directories such as `plots/probability_change/level_1_block_residual/` and `plots/probability_change/level_3_attn_submodule_decomposition/`.
- `attention_analysis` adds decoder head-level analysis driven by `attention_analysis.prompt`, `attention_analysis.top_k_heads`, and `attention_analysis.token_groups`.
- `attention_analysis` supports `dataset.concepts[*].csv` or `csv_stem`, and accepts `group: control` as an alias for `non_affective`.
- `attention_analysis` interprets `dataset.num_samples` as “up to N positive images and up to N negative images per concept”.
- `output.plot_format` is shared by all experiments and supports `png` or `pdf` (default: `png`).
- `attention_analysis` saves:
  - layer allocation curves under `plots/attention_allocation/`
  - single-concept and grouped heatmaps under `plots/head_importance/`
  - top-head summaries and detail charts under `plots/top_heads_detail/`

## Dependencies

Install these packages in the runtime environment before launching experiments:

- `torch`
- `transformers`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `Pillow`
- `tqdm`
- `pyyaml`

## Outputs

Each run creates:

```text
output/{experiment_name}_{timestamp}/
├── results/
├── plots/
└── logs/
```

`logs/` contains the config snapshot, environment metadata, git metadata, and an experiment log.
