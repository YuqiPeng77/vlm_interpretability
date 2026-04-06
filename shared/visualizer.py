from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from math import ceil


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
CONTROL_COLOR = "#6B7280"
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
PCA_CLASS_STYLES = {
    0: {"color": "#2563EB", "label": "negative"},
    1: {"color": "#DC2626", "label": "positive"},
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


def get_global_control_mean(summary: dict, component: str) -> np.ndarray:
    concept_order = get_selected_concepts(summary)
    curves = [
        summary["results"][concept][f"{component}_random_label_mean"]
        for concept in concept_order
    ]
    return np.mean(np.array(curves, dtype=float), axis=0)


def get_model_display_name(summary: dict) -> str:
    meta = summary.get("meta", {})
    return meta.get("model_display_name") or meta.get("model_path", "model")


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

    global_control_mean = get_global_control_mean(summary, component)
    layers = get_x_positions(len(global_control_mean))
    ax.plot(
        layers,
        global_control_mean,
        color=CONTROL_COLOR,
        linestyle=":",
        linewidth=MEAN_LINEWIDTH - 0.2,
        label="random-label mean",
        zorder=8,
    )

    ax.set_title(f"{get_model_display_name(summary)} - {component.capitalize()} Probing Accuracy by Layer")
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
    ] + [
        Line2D(
            [0],
            [0],
            color=CONTROL_COLOR,
            linestyle=":",
            linewidth=MEAN_LINEWIDTH - 0.2,
            label="random-label mean",
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


def plot_single_attribute_probing_accuracy(
    summary: dict,
    concept_name: str,
    component: str,
    output_path: Path,
    dpi: int = 300,
) -> None:
    concept_result = summary["results"][concept_name]
    accuracies = np.array(concept_result[f"{component}_layer_accuracy"], dtype=float)
    random_mean = np.array(concept_result[f"{component}_random_label_mean"], dtype=float)
    layers = get_x_positions(len(accuracies))
    color = "#DC2626" if concept_result["group"] == "affective" else "#1D4ED8"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        layers,
        accuracies,
        color=color,
        linewidth=2.5,
        linestyle="-",
        label="real probing",
    )
    ax.plot(
        layers,
        random_mean,
        color=CONTROL_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="random-label baseline",
    )
    xticks, xtick_labels = get_xticks_and_labels(layers)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(left=min(layers), right=max(layers))
    ax.set_xlabel("Input / Layer Output")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*Y_AXIS_LIMITS)
    ax.grid(axis="y", alpha=GRID_ALPHA, color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    ax.set_title(f"{get_model_display_name(summary)} - {concept_name} - {component.capitalize()} Probing")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter_grid(
    layer_payloads: list[dict],
    concept_name: str,
    component: str,
    model_display_name: str,
    output_path: Path,
    dpi: int = 300,
) -> None:
    if not layer_payloads:
        raise ValueError("No PCA payloads available to plot.")

    x_limits, y_limits = get_pca_axis_limits(layer_payloads)
    num_plots = len(layer_payloads)
    ncols = 1 if num_plots == 1 else 2
    nrows = ceil(num_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)
    axes_flat = list(axes.flat)

    for ax, payload in zip(axes_flat, layer_payloads):
        draw_pca_scatter(ax, payload, x_limits, y_limits)

    for ax in axes_flat[num_plots:]:
        ax.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=style["label"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=7,
        )
        for style in PCA_CLASS_STYLES.values()
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.suptitle(f"{model_display_name} - {concept_name} - {component.capitalize()} PCA", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.96))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def get_pca_axis_limits(layer_payloads: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate(
        [np.asarray(payload["points"], dtype=float) for payload in layer_payloads],
        axis=0,
    )
    x_values = all_points[:, 0]
    y_values = all_points[:, 1]
    x_margin = max((x_values.max() - x_values.min()) * 0.08, 0.5)
    y_margin = max((y_values.max() - y_values.min()) * 0.08, 0.5)
    x_limits = (x_values.min() - x_margin, x_values.max() + x_margin)
    y_limits = (y_values.min() - y_margin, y_values.max() + y_margin)
    return x_limits, y_limits


def get_pca_layer_label(layer_idx: int) -> str:
    return "Input" if layer_idx < 0 else f"L{layer_idx}"


def draw_pca_scatter(
    ax,
    payload: dict,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    points = np.asarray(payload["points"], dtype=float)
    labels = np.asarray(payload["labels"], dtype=int)
    layer_idx = int(payload["layer"])
    explained_variance = payload["explained_variance_ratio"]
    layer_label = get_pca_layer_label(layer_idx)

    for class_label, style in PCA_CLASS_STYLES.items():
        mask = labels == class_label
        if not np.any(mask):
            continue
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=34,
            alpha=0.82,
            color=style["color"],
            label=style["label"],
            edgecolors="white",
            linewidths=0.35,
        )

    ax.axhline(0.0, color="#9CA3AF", linewidth=0.8, alpha=0.8)
    ax.axvline(0.0, color="#9CA3AF", linewidth=0.8, alpha=0.8)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(alpha=0.25)
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}%)")
    ax.set_title(f"{layer_label} | top-2 variance = {sum(explained_variance) * 100:.1f}%")


def plot_single_pca_scatter(
    payload: dict,
    concept_name: str,
    component: str,
    model_display_name: str,
    output_path: Path,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    dpi: int = 300,
) -> None:
    layer_idx = int(payload["layer"])
    layer_label = get_pca_layer_label(layer_idx)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    draw_pca_scatter(ax, payload, x_limits, y_limits)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=style["label"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=7,
        )
        for style in PCA_CLASS_STYLES.values()
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=9)
    ax.set_title(f"{model_display_name} - {concept_name} - {component.capitalize()} {layer_label} PCA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def get_fisher_group_curves(summary: dict, component: str, group: str) -> np.ndarray:
    concept_order = get_selected_concepts(summary)
    curves = [
        summary["results"][concept][f"{component}_layer_fisher_ratio"]
        for concept in concept_order
        if summary["results"][concept]["group"] == group
    ]
    return np.array(curves, dtype=float)


def get_fisher_y_limits(curves: list[np.ndarray]) -> tuple[float, float]:
    if not curves:
        return (0.0, 1.0)
    merged = np.concatenate([np.asarray(curve, dtype=float) for curve in curves], axis=0)
    upper = float(np.max(merged))
    if upper <= 0.0:
        return (0.0, 1.0)
    margin = max(upper * 0.08, 1e-6)
    return (0.0, upper + margin)


def plot_single_fisher_ratio_attribute(
    summary: dict,
    concept_name: str,
    component: str,
    output_path: Path,
    dpi: int = 300,
) -> None:
    concept_result = summary["results"][concept_name]
    fisher_ratio = np.array(concept_result[f"{component}_layer_fisher_ratio"], dtype=float)
    layers = get_x_positions(len(fisher_ratio))
    color = "#DC2626" if concept_result["group"] == "affective" else "#1D4ED8"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        layers,
        fisher_ratio,
        color=color,
        linewidth=2.5,
        linestyle="-",
        label=DISPLAY_LABELS.get(concept_name, concept_name),
    )
    xticks, xtick_labels = get_xticks_and_labels(layers)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(left=min(layers), right=max(layers))
    ax.set_ylim(*get_fisher_y_limits([fisher_ratio]))
    ax.set_xlabel("Input / Layer Output")
    ax.set_ylabel("Fisher Ratio")
    ax.grid(axis="y", alpha=GRID_ALPHA, color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    ax.set_title(f"{get_model_display_name(summary)} - {concept_name} - {component.capitalize()} Fisher Ratio")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fisher_ratio_all_attributes(
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
    curves = [np.array(results[concept][f"{component}_layer_fisher_ratio"], dtype=float) for concept in concept_order]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for concept in concept_order:
        concept_result = results[concept]
        fisher_ratio = np.array(concept_result[f"{component}_layer_fisher_ratio"], dtype=float)
        layers = get_x_positions(len(fisher_ratio))
        linestyle = "-" if concept_result["group"] == "non_affective" else "--"
        ax.plot(
            layers,
            fisher_ratio,
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
        group_curves = get_fisher_group_curves(summary, component, group)
        if group_curves.size == 0:
            continue
        mean_curve = np.mean(group_curves, axis=0)
        std_curve = np.std(group_curves, axis=0)
        layers = get_x_positions(len(mean_curve))
        lower = np.clip(mean_curve - std_curve, 0.0, None)
        upper = mean_curve + std_curve
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

    x_positions = get_x_positions(len(curves[0]))
    xticks, xtick_labels = get_xticks_and_labels(x_positions)
    ax.set_xlim(left=min(x_positions), right=max(x_positions))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylim(*get_fisher_y_limits(curves))
    ax.set_xlabel("Input / Layer Output")
    ax.set_ylabel("Fisher Ratio")
    ax.set_title(f"{get_model_display_name(summary)} - {component.capitalize()} Fisher Ratio by Layer")
    ax.grid(axis="y", color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)

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
        loc="upper left",
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
