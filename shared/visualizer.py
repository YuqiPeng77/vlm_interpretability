from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


SELECTED_NON_AFFECTIVE = {
    "natural",
    "natural light",
    "rugged scene",
}
DISPLAY_LABELS: dict[str, str] = {}
INDIVIDUAL_ALPHA = 0.55
INDIVIDUAL_LINEWIDTH = 1.8
MEAN_LINEWIDTH = 3.0
STD_BAND_ALPHA = 0.12
FIGURE_SIZE = (13, 7)
Y_AXIS_LIMITS = (0.45, 1.0)
GRID_COLOR = "#D1D5DB"
GRID_LINESTYLE = "-"
GRID_LINEWIDTH = 0.8
GRID_ALPHA = 0.8
LEGEND_FONTSIZE = 11
LEGEND_TITLE_FONTSIZE = 12
NON_AFFECTIVE_COLORS = ["#1D4ED8", "#0891B2", "#0F766E", "#6366F1"]
AFFECTIVE_COLORS = ["#EA580C", "#DC2626", "#BE185D"]
GROUP_MEAN_COLORS = {
    "non_affective": "#1E3A8A",
    "affective": "#991B1B",
}
MODE_COLORS = {
    "noise": "tab:blue",
    "zero": "tab:orange",
    "counterfactual": "tab:green",
    "both": "black",
    "attn_only": "tab:blue",
    "mlp_only": "tab:orange",
    "q": "tab:blue",
    "k": "tab:orange",
    "v": "tab:green",
    "proj": "tab:red",
}


def get_selected_concepts(summary: dict) -> list[str]:
    selected: list[str] = []
    concept_order = summary.get("concept_order", [])
    results = summary["results"]

    for concept in concept_order:
        group = results[concept]["group"]
        if group == "affective" or concept in SELECTED_NON_AFFECTIVE:
            selected.append(concept)

    for concept, concept_result in results.items():
        if concept in selected:
            continue
        group = concept_result["group"]
        if group == "affective" or concept in SELECTED_NON_AFFECTIVE:
            selected.append(concept)

    if selected:
        return selected
    if concept_order:
        return list(concept_order)
    return list(results.keys())


def build_color_map(summary: dict, concept_order: list[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    non_affective_idx = 0
    affective_idx = 0

    for concept in concept_order:
        group = summary["results"][concept]["group"]
        if group == "non_affective":
            color_map[concept] = NON_AFFECTIVE_COLORS[non_affective_idx % len(NON_AFFECTIVE_COLORS)]
            non_affective_idx += 1
        else:
            color_map[concept] = AFFECTIVE_COLORS[affective_idx % len(AFFECTIVE_COLORS)]
            affective_idx += 1

    return color_map


def get_group_curves(summary: dict, component: str, group: str) -> np.ndarray:
    concept_order = get_selected_concepts(summary)
    curves = [
        summary["results"][concept][f"{component}_layer_accuracy"]
        for concept in concept_order
        if summary["results"][concept]["group"] == group
    ]
    return np.array(curves, dtype=float)


def get_group_stats(summary: dict, component: str, group: str) -> tuple[np.ndarray, np.ndarray]:
    curves = get_group_curves(summary, component, group)
    return np.mean(curves, axis=0), np.std(curves, axis=0)


def get_x_positions(num_points: int) -> list[int]:
    return list(range(-1, num_points - 1))


def get_xticks_and_labels(x_positions: list[int]) -> tuple[list[int], list[str]]:
    xticks = [-1]
    if len(x_positions) <= 10:
        xticks.extend(x_positions[1:])
    else:
        step = 5
        max_layer = x_positions[-1]
        xticks.extend(list(range(0, max_layer + 1, step)))
        if xticks[-1] != max_layer:
            xticks.append(max_layer)
    labels = ["Input"] + [f"L{tick}" for tick in xticks[1:]]
    return xticks, labels


def plot_grouped_probing_accuracy(
    summary: dict,
    component: str,
    output_path: Path,
    dpi: int = 300,
) -> None:
    concept_order = get_selected_concepts(summary)
    if not concept_order:
        raise ValueError("No concepts available to plot.")
    results = summary["results"]
    color_map = build_color_map(summary, concept_order)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for concept in concept_order:
        concept_result = results[concept]
        group = concept_result["group"]
        accuracies = concept_result[f"{component}_layer_accuracy"]
        layers = get_x_positions(len(accuracies))
        linestyle = "-" if group == "non_affective" else "--"
        ax.plot(
            layers,
            accuracies,
            color=color_map[concept],
            linestyle=linestyle,
            linewidth=INDIVIDUAL_LINEWIDTH,
            alpha=INDIVIDUAL_ALPHA,
            label=DISPLAY_LABELS.get(concept, concept),
        )

    average_styles = {
        "non_affective": {
            "color": GROUP_MEAN_COLORS["non_affective"],
            "linestyle": "-",
            "label": "non-affective mean",
        },
        "affective": {
            "color": GROUP_MEAN_COLORS["affective"],
            "linestyle": "--",
            "label": "affective mean",
        },
    }
    for group, style in average_styles.items():
        curves = get_group_curves(summary, component, group)
        if curves.size == 0:
            continue
        mean_curve, std_curve = get_group_stats(summary, component, group)
        layers = get_x_positions(len(mean_curve))
        lower = np.clip(mean_curve - std_curve, 0.0, 1.0)
        upper = np.clip(mean_curve + std_curve, 0.0, 1.0)
        ax.fill_between(
            layers,
            lower,
            upper,
            color=style["color"],
            alpha=STD_BAND_ALPHA,
            linewidth=0,
            zorder=3,
        )
        ax.plot(
            layers,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=MEAN_LINEWIDTH,
            label=style["label"],
            zorder=10,
        )

    ax.set_title(f"{component.capitalize()} Probing Accuracy by Layer")
    ax.set_xlabel("Input / Layer Output")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*Y_AXIS_LIMITS)

    x_positions = get_x_positions(len(results[concept_order[0]][f"{component}_layer_accuracy"]))
    xticks, xtick_labels = get_xticks_and_labels(x_positions)
    ax.set_xlim(left=min(x_positions), right=max(x_positions))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )

    non_affective_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[concept],
            linestyle="-",
            linewidth=INDIVIDUAL_LINEWIDTH,
            label=DISPLAY_LABELS.get(concept, concept),
            alpha=INDIVIDUAL_ALPHA,
        )
        for concept in concept_order
        if results[concept]["group"] == "non_affective"
    ]
    affective_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[concept],
            linestyle="--",
            linewidth=INDIVIDUAL_LINEWIDTH,
            label=DISPLAY_LABELS.get(concept, concept),
            alpha=INDIVIDUAL_ALPHA,
        )
        for concept in concept_order
        if results[concept]["group"] == "affective"
    ]
    legend_handles = non_affective_handles + [
        Line2D(
            [0],
            [0],
            color=GROUP_MEAN_COLORS["non_affective"],
            linestyle="-",
            linewidth=MEAN_LINEWIDTH,
            label="non-affective mean",
        )
    ] + affective_handles + [
        Line2D(
            [0],
            [0],
            color=GROUP_MEAN_COLORS["affective"],
            linestyle="--",
            linewidth=MEAN_LINEWIDTH,
            label="affective mean",
        )
    ]
    ax.legend(
        handles=legend_handles,
        title="Attributes",
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def plot_p_yes(
    layer_means: np.ndarray,
    layer_stds: np.ndarray,
    stage: str,
    mode: str,
    concept_name: str,
    p_yes_clean_mean: float,
    p_yes_corrupted_mean: float,
    save_path: Path,
) -> None:
    x_values = np.arange(len(layer_means))
    color = MODE_COLORS.get(mode, "black")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        x_values,
        layer_means,
        marker="o",
        markersize=3,
        color=color,
        label=f"P(Yes) after {mode} patching",
    )
    ax.fill_between(
        x_values,
        layer_means - layer_stds,
        layer_means + layer_stds,
        alpha=0.15,
        color=color,
    )
    ax.axhline(
        p_yes_clean_mean,
        color="green",
        linestyle="--",
        linewidth=0.8,
        label=f"P(Yes) clean = {p_yes_clean_mean:.3f}",
    )
    ax.axhline(
        p_yes_corrupted_mean,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"P(Yes) corrupted = {p_yes_corrupted_mean:.3f}",
    )
    ax.set_xlabel(f"{stage.capitalize()} Layer")
    ax.set_ylabel("P(Yes)")
    ax.set_title(f"P(Yes) after patching ({mode}): {concept_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_attention_allocation(
    layer_indices: list[int],
    allocation_summary: dict[str, np.ndarray],
    concept_name: str,
    save_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x_values = np.array(layer_indices)
    styles = {
        "image": {"color": "#2563EB", "linestyle": "-", "label": "image tokens"},
        "text": {"color": "#0F766E", "linestyle": "-", "label": "text tokens"},
    }

    for key, style in styles.items():
        if key not in allocation_summary:
            continue
        ax.plot(
            x_values,
            allocation_summary[key],
            marker="o",
            markersize=3,
            linewidth=2.0,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: {concept_name}")
    ax.set_xticks(layer_indices)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_head_importance_heatmap(
    importance: np.ndarray,
    save_path: Path,
    title: str,
    highlights: list[tuple[int, int, str]] | None = None,
    colorbar_label: str = "Magnitude",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = float(np.nanmax(importance)) if importance.size > 0 else np.nan
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    image = ax.imshow(importance, aspect="auto", cmap="viridis", origin="upper", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")

    if highlights:
        for layer_idx, head_idx, label in highlights:
            ax.add_patch(
                Rectangle((head_idx - 0.5, layer_idx - 0.5), 1.0, 1.0, fill=False, edgecolor="white", linewidth=1.2)
            )
            ax.text(head_idx, layer_idx, label, color="white", fontsize=8, ha="center", va="center")

    fig.colorbar(image, ax=ax, location="right", shrink=0.8, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_top_head_detail(
    group_names: list[str],
    positive_values: list[float],
    negative_values: list[float],
    concept_name: str,
    layer_idx: int,
    head_idx: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = np.arange(len(group_names))
    width = 0.36
    ax.bar(x_values - width / 2, positive_values, width=width, color="#2563EB", label="positive")
    ax.bar(x_values + width / 2, negative_values, width=width, color="#DC2626", label="negative")
    ax.set_xticks(x_values)
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(f"{concept_name}: Top Head L{layer_idx} H{head_idx}")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
