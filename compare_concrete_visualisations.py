import os
import tempfile
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
DETAILS_DIR = BASE_DIR / "model_details"
OUTPUT_DIR = BASE_DIR / "model_visualisations" / "normal_vs_uhpc"
MPL_CACHE_DIR = OUTPUT_DIR / ".matplotlib-cache"


def configure_matplotlib_cache():
    cache_locations = [
        MPL_CACHE_DIR,
        Path(tempfile.gettempdir()) / "normal_vs_uhpc_matplotlib_cache",
    ]

    for cache_dir in cache_locations:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            test_file = cache_dir / ".write-test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
            os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))
            return
        except OSError:
            continue


configure_matplotlib_cache()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NORMAL_COLOR = "#2f6f73"
UHPC_COLOR = "#b84a62"
ACCENT_COLOR = "#d88c32"
GRID_COLOR = "#d8d6cf"


def style_axis(axis):
    axis.grid(axis="x", color=GRID_COLOR, alpha=0.55)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def save_model_metric_comparison(metrics):
    model_order = metrics.sort_values("RMSE")["model"].drop_duplicates().tolist()
    metric_specs = [
        ("RMSE", "Lower is better"),
        ("MAE", "Lower is better"),
        ("R2", "Higher is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    for axis, (metric, subtitle) in zip(axes, metric_specs):
        pivot = metrics.pivot(index="model", columns="concrete_type", values=metric).reindex(model_order)
        y_positions = range(len(pivot))
        bar_height = 0.36

        axis.barh(
            [position - bar_height / 2 for position in y_positions],
            pivot["normal"],
            height=bar_height,
            color=NORMAL_COLOR,
            label="Normal",
        )
        axis.barh(
            [position + bar_height / 2 for position in y_positions],
            pivot["uhpc"],
            height=bar_height,
            color=UHPC_COLOR,
            label="UHPC",
        )
        axis.set_yticks(list(y_positions))
        axis.set_yticklabels(pivot.index)
        axis.invert_yaxis()
        axis.set_title(f"{metric} ({subtitle})", fontsize=12, weight="bold")
        axis.set_xlabel(metric)
        style_axis(axis)

        for position, value in enumerate(pivot["normal"]):
            axis.text(value, position - bar_height / 2, f" {value:.2f}", va="center", fontsize=9)
        for position, value in enumerate(pivot["uhpc"]):
            axis.text(value, position + bar_height / 2, f" {value:.2f}", va="center", fontsize=9)

    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Normal Concrete vs UHPC Model Performance", fontsize=16, weight="bold")
    fig.savefig(OUTPUT_DIR / "model_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_comparison(importance):
    models = ["Random Forest Regressor", "Gradient Boosting Regressor"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    for axis, model_name in zip(axes, models):
        model_importance = importance[importance["model"] == model_name]
        pivot = model_importance.pivot(
            index="feature",
            columns="concrete_type",
            values="absolute_importance",
        ).fillna(0)
        pivot["difference"] = pivot["uhpc"] - pivot["normal"]
        selected = pivot.reindex(pivot["difference"].abs().sort_values(ascending=False).index).head(10)
        selected = selected.sort_values("difference")
        y_positions = range(len(selected))
        bar_height = 0.36

        axis.barh(
            [position - bar_height / 2 for position in y_positions],
            selected["normal"],
            height=bar_height,
            color=NORMAL_COLOR,
            label="Normal",
        )
        axis.barh(
            [position + bar_height / 2 for position in y_positions],
            selected["uhpc"],
            height=bar_height,
            color=UHPC_COLOR,
            label="UHPC",
        )
        axis.set_yticks(list(y_positions))
        axis.set_yticklabels(selected.index)
        axis.set_xlabel("Absolute feature importance")
        axis.set_title(model_name, fontsize=12, weight="bold")
        style_axis(axis)

    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Largest Feature-Importance Differences", fontsize=16, weight="bold")
    fig.savefig(OUTPUT_DIR / "feature_importance_difference.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_composition_difference(mean_comparison):
    selected = mean_comparison.reindex(
        mean_comparison["uhpc_minus_normal_mean"].abs().sort_values(ascending=False).index
    ).head(12)
    selected = selected.sort_values("uhpc_minus_normal_mean")

    colors = [NORMAL_COLOR if value < 0 else UHPC_COLOR for value in selected["uhpc_minus_normal_mean"]]
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.barh(selected["feature"], selected["uhpc_minus_normal_mean"], color=colors)
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_xlabel("UHPC mean minus normal concrete mean")
    axis.set_title("Largest Mean Composition Differences", fontsize=14, weight="bold")
    style_axis(axis)

    for feature, value in zip(selected["feature"], selected["uhpc_minus_normal_mean"]):
        ha = "left" if value >= 0 else "right"
        offset = 6 if value >= 0 else -6
        axis.annotate(
            f"{value:.1f}",
            xy=(value, feature),
            xytext=(offset, 0),
            textcoords="offset points",
            va="center",
            ha=ha,
            fontsize=9,
        )

    fig.savefig(OUTPUT_DIR / "mean_composition_difference.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_composition_profile(mean_comparison):
    material_features = [
        "cement",
        "binder",
        "water",
        "silica_fume",
        "flyash",
        "ggbs",
        "coarse_agg",
        "fine_agg",
        "superplasticizer",
    ]
    profile = mean_comparison.set_index("feature").loc[material_features]
    y_positions = range(len(profile))
    bar_height = 0.36

    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.barh(
        [position - bar_height / 2 for position in y_positions],
        profile["normal"],
        height=bar_height,
        color=NORMAL_COLOR,
        label="Normal",
    )
    axis.barh(
        [position + bar_height / 2 for position in y_positions],
        profile["uhpc"],
        height=bar_height,
        color=UHPC_COLOR,
        label="UHPC",
    )
    axis.set_yticks(list(y_positions))
    axis.set_yticklabels(profile.index)
    axis.invert_yaxis()
    axis.set_xlabel("Mean dosage / derived quantity")
    axis.set_title("Mean Mix Profile", fontsize=14, weight="bold")
    axis.legend(frameon=False, loc="lower right")
    style_axis(axis)

    fig.savefig(OUTPUT_DIR / "mean_mix_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_distribution_boxplots(normal, uhpc):
    features = ["cs", "water_binder_ratio", "binder", "superplasticizer", "silica_fume", "fine_agg"]
    titles = [
        "Compressive strength",
        "Water-binder ratio",
        "Binder",
        "Superplasticizer",
        "Silica fume",
        "Fine aggregate",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for axis, feature, title in zip(axes, features, titles):
        box = axis.boxplot(
            [normal[feature], uhpc[feature]],
            tick_labels=["Normal", "UHPC"],
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.6},
        )
        box["boxes"][0].set_facecolor(NORMAL_COLOR)
        box["boxes"][1].set_facecolor(UHPC_COLOR)
        for patch in box["boxes"]:
            patch.set_alpha(0.78)
        axis.set_title(title, fontsize=11, weight="bold")
        axis.grid(axis="y", color=GRID_COLOR, alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Distribution Differences Between Normal Concrete and UHPC", fontsize=16, weight="bold")
    fig.savefig(OUTPUT_DIR / "key_distribution_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_visualisation_index():
    lines = [
        "# Normal Concrete vs UHPC Visualisations",
        "",
        "- `model_metric_comparison.png`: model performance split by concrete type.",
        "- `feature_importance_difference.png`: feature-importance differences for Random Forest and Gradient Boosting.",
        "- `mean_composition_difference.png`: largest mean mix-design differences, shown as UHPC minus normal concrete.",
        "- `mean_mix_profile.png`: side-by-side mean composition profile.",
        "- `key_distribution_boxplots.png`: distribution comparison for strength and key mix variables.",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(DETAILS_DIR / "normal_vs_uhpc_model_metrics.csv")
    importance = pd.read_csv(DETAILS_DIR / "normal_vs_uhpc_feature_importance.csv")
    mean_comparison = pd.read_csv(DETAILS_DIR / "normal_vs_uhpc_mean_composition_difference.csv")
    normal = pd.read_csv(DATA_DIR / "updated_normal_concrete.csv")
    uhpc = pd.read_csv(DATA_DIR / "update_uhpc_concrete.csv")

    save_model_metric_comparison(metrics)
    save_feature_importance_comparison(importance)
    save_composition_difference(mean_comparison)
    save_composition_profile(mean_comparison)
    save_distribution_boxplots(normal, uhpc)
    write_visualisation_index()

    print(f"Saved normal-vs-UHPC visualisations to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
