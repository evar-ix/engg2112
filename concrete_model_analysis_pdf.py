import os
import textwrap
from pathlib import Path


OUTPUT_DIR = Path("model_visualisations")
PDF_PATH = OUTPUT_DIR / "concrete_model_analysis.pdf"
MPL_CACHE_DIR = OUTPUT_DIR / ".matplotlib-cache"

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches.
TEXT_COLOR = "#1f2933"
MUTED_COLOR = "#52616b"
ACCENT_COLOR = "#2f6f73"


def add_wrapped_text(axis, text, x, y, width=88, size=10.5, color=TEXT_COLOR, weight="normal", line_spacing=1.35):
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    axis.text(
        x,
        y,
        wrapped,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        color=color,
        weight=weight,
        linespacing=line_spacing,
    )


def add_bullets(axis, bullets, x, y, width=82, size=10.2, gap=0.06):
    current_y = y
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=width)
        axis.text(
            x,
            current_y,
            "- " + wrapped[0],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=size,
            color=TEXT_COLOR,
        )
        for line in wrapped[1:]:
            current_y -= 0.032
            axis.text(
                x + 0.025,
                current_y,
                line,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=size,
                color=TEXT_COLOR,
            )
        current_y -= gap


def new_text_page(title, subtitle=None):
    fig = plt.figure(figsize=PAGE_SIZE)
    axis = fig.add_axes([0, 0, 1, 1])
    axis.axis("off")
    axis.text(
        0.08,
        0.93,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        color=TEXT_COLOR,
        weight="bold",
    )
    if subtitle:
        add_wrapped_text(axis, subtitle, 0.08, 0.88, width=82, size=11.5, color=MUTED_COLOR)
    return fig, axis


def add_chart_page(pdf, image_path, title, bullets):
    fig = plt.figure(figsize=PAGE_SIZE)
    title_axis = fig.add_axes([0.08, 0.88, 0.84, 0.08])
    title_axis.axis("off")
    title_axis.text(
        0,
        0.95,
        title,
        ha="left",
        va="top",
        fontsize=18,
        weight="bold",
        color=TEXT_COLOR,
    )

    image_axis = fig.add_axes([0.08, 0.35, 0.84, 0.48])
    image_axis.imshow(mpimg.imread(image_path))
    image_axis.axis("off")

    note_axis = fig.add_axes([0.08, 0.06, 0.84, 0.24])
    note_axis.axis("off")
    add_bullets(note_axis, bullets, 0, 0.98, width=96, size=10.5, gap=0.085)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_metrics_table(axis, metrics):
    display_metrics = metrics.copy()
    display_metrics["RMSE"] = display_metrics["RMSE"].map(lambda value: f"{value:.2f}")
    display_metrics["MAE"] = display_metrics["MAE"].map(lambda value: f"{value:.2f}")
    display_metrics["R2"] = display_metrics["R2"].map(lambda value: f"{value:.3f}")

    table_axis = axis.inset_axes([0.08, 0.12, 0.84, 0.25])
    table_axis.axis("off")
    table = table_axis.table(
        cellText=display_metrics[["Model", "RMSE", "MAE", "R2"]].values,
        colLabels=["Model", "RMSE", "MAE", "R2"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.55, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dee5")
        if row == 0:
            cell.set_facecolor("#e8f0ef")
            cell.set_text_props(weight="bold", color=TEXT_COLOR)
        elif row == 1:
            cell.set_facecolor("#f3faf7")
        else:
            cell.set_facecolor("white")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv").sort_values("RMSE")
    best = metrics.iloc[0]

    with PdfPages(PDF_PATH) as pdf:
        fig, axis = new_text_page(
            "Analysis of Concrete Strength Model Visualisations",
            "This report explains how the four machine-learning models estimate concrete compressive strength and interprets the generated graphs.",
        )
        add_wrapped_text(
            axis,
            f"Overall conclusion: the {best['Model']} performs best. It has the lowest RMSE ({best['RMSE']:.2f}), lowest MAE ({best['MAE']:.2f}), and highest R2 ({best['R2']:.3f}). This means its predictions are closest to the measured concrete compressive strength values in the test data.",
            0.08,
            0.78,
            width=84,
            size=12,
            weight="bold",
            color=ACCENT_COLOR,
        )
        add_wrapped_text(
            axis,
            "Compressive strength is predicted from mix-design and curing variables such as cement, water, binder, water-binder ratio, aggregates, supplementary cementitious materials, superplasticizer, temperature, age, and whether the sample is UHPC. The target variable is cs, which represents measured compressive strength.",
            0.08,
            0.64,
            width=88,
            size=10.8,
        )
        add_bullets(
            axis,
            [
                "Simple Linear Regression uses one selected predictor, binder, and fits a straight-line relationship between that feature and strength.",
                "Multiple Linear Regression uses all input features in a weighted linear equation, so each feature adds or subtracts from the predicted strength.",
                "Random Forest builds many decision trees and averages their predictions, allowing nonlinear effects and interactions between mix variables.",
                "Gradient Boosting builds trees sequentially, where each new tree tries to correct the errors left by the previous trees.",
            ],
            0.1,
            0.50,
            width=78,
            size=10.5,
            gap=0.075,
        )
        add_metrics_table(axis, metrics)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_chart_page(
            pdf,
            OUTPUT_DIR / "model_metric_comparison.png",
            "Model Performance Comparison",
            [
                "RMSE and MAE measure the size of prediction errors. Lower values mean the predicted compressive strength is closer to the real measured value.",
                "R2 measures how much of the variation in compressive strength is explained by the model. Values closer to 1 indicate stronger predictive performance.",
                "Random Forest is clearly best on this graph because it has the smallest errors and the highest R2. Gradient Boosting is second, while both linear models lose accuracy because they cannot capture enough nonlinear behaviour.",
            ],
        )

        add_chart_page(
            pdf,
            OUTPUT_DIR / "actual_vs_predicted_by_model.png",
            "Actual vs Predicted Strength",
            [
                "The diagonal line represents perfect predictions. Points close to that line mean the model predicted concrete strength accurately.",
                "Random Forest has the tightest cluster around the diagonal line, showing strong agreement between predicted and actual strength values.",
                "The Simple Linear Regression plot is much more spread out because binder alone cannot describe the full behaviour of concrete strength development.",
            ],
        )

        add_chart_page(
            pdf,
            OUTPUT_DIR / "residuals_by_model.png",
            "Residual Analysis",
            [
                "Residuals are calculated as actual strength minus predicted strength. A good model should have residuals scattered closely around zero.",
                "Random Forest shows the smallest residual spread, meaning its errors are generally smaller and less systematic.",
                "The linear models show wider residual patterns, which suggests they miss important nonlinear relationships such as the combined effect of water-binder ratio, age, and cementitious materials.",
            ],
        )

        add_chart_page(
            pdf,
            OUTPUT_DIR / "model_feature_summary.png",
            "Feature Importance and Model Behaviour",
            [
                "The tree-based models rely most strongly on water-binder ratio and age. This agrees with concrete technology: lower water-binder ratio usually increases strength, and curing age strongly affects strength gain.",
                "Multiple Linear Regression spreads influence across many variables because it can only form a weighted sum. Coefficients show direction and relative effect after standardisation, but not complex interactions.",
                "Random Forest and Gradient Boosting can split the data into different regions, so they can represent thresholds and interactions that are difficult for linear regression to model.",
            ],
        )

        fig, axis = new_text_page("Final Interpretation")
        add_wrapped_text(
            axis,
            "The graphs show that concrete compressive strength is not explained well by a single variable or by a simple straight-line equation. Strength depends on several interacting factors, especially water-binder ratio, curing age, and binder composition.",
            0.08,
            0.82,
            width=86,
            size=11.2,
        )
        add_wrapped_text(
            axis,
            "Random Forest performs best because it can model nonlinear relationships and interactions while remaining stable through averaging many trees. Gradient Boosting also performs strongly, but in these results it has slightly larger errors. Multiple Linear Regression is useful for interpretation, but its straight-line structure limits accuracy. Simple Linear Regression is the weakest because binder alone does not contain enough information to predict compressive strength reliably.",
            0.08,
            0.66,
            width=86,
            size=11.2,
        )
        add_wrapped_text(
            axis,
            f"Therefore, based on the visualisations and the test-set metrics, the recommended model for predicting concrete compressive strength is the {best['Model']}.",
            0.08,
            0.42,
            width=84,
            size=12.5,
            weight="bold",
            color=ACCENT_COLOR,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved PDF analysis to {PDF_PATH.resolve()}")


if __name__ == "__main__":
    main()
