import math
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("combined_concrete.csv")
TARGET = "cs"
OUTPUT_DIR = Path("model_visualisations")
MPL_CACHE_DIR = OUTPUT_DIR / ".matplotlib-cache"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def find_best_simple_feature(X_train, y_train):
    best_feature = None
    best_rmse = float("inf")

    for column in X_train.columns:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        cv_rmse = -cross_val_score(
            model,
            X_train[[column]],
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=5,
        ).mean()

        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_feature = column

    return best_feature


def build_models(best_simple_feature):
    return {
        f"Simple Linear Regression ({best_simple_feature})": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        "Multiple Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }


def train_models(X_train, X_test, y_train, y_test, best_simple_feature):
    rows = []
    prediction_frames = []
    fitted_models = {}

    for name, model in build_models(best_simple_feature).items():
        if name.startswith("Simple Linear Regression"):
            train_features = X_train[[best_simple_feature]]
            test_features = X_test[[best_simple_feature]]
        else:
            train_features = X_train
            test_features = X_test

        model.fit(train_features, y_train)
        predictions = model.predict(test_features)

        rows.append({
            "Model": name,
            "RMSE": root_mean_squared_error(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions),
            "R2": r2_score(y_test, predictions),
        })
        prediction_frames.append(pd.DataFrame({
            "Model": name,
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "Residual": y_test.to_numpy() - predictions,
        }))
        fitted_models[name] = model

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions, fitted_models


def save_metric_comparison(metrics):
    colors = ["#2f6f73", "#d88c32", "#5f5a8b", "#b84a62"]
    ordered = metrics.sort_values("RMSE")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    metric_specs = [
        ("RMSE", "Lower is better"),
        ("MAE", "Lower is better"),
        ("R2", "Higher is better"),
    ]

    for axis, (metric, subtitle) in zip(axes, metric_specs):
        bars = axis.barh(ordered["Model"], ordered[metric], color=colors)
        axis.set_title(f"{metric} ({subtitle})", fontsize=12, weight="bold")
        axis.set_xlabel(metric)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        axis.spines[["top", "right", "left"]].set_visible(False)

        for bar in bars:
            width = bar.get_width()
            label = f"{width:.3f}" if metric == "R2" else f"{width:.2f}"
            axis.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {label}",
                va="center",
                fontsize=9,
            )

    fig.suptitle("Concrete Strength Model Performance", fontsize=16, weight="bold")
    fig.savefig(OUTPUT_DIR / "model_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prediction_scatter(metrics, predictions):
    model_names = metrics.sort_values("RMSE")["Model"].tolist()
    actual_min = predictions["Actual"].min()
    actual_max = predictions["Actual"].max()
    padding = (actual_max - actual_min) * 0.05
    limits = [actual_min - padding, actual_max + padding]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    for axis, model_name in zip(axes, model_names):
        model_predictions = predictions[predictions["Model"] == model_name]
        model_metrics = metrics[metrics["Model"] == model_name].iloc[0]

        axis.scatter(
            model_predictions["Actual"],
            model_predictions["Predicted"],
            s=18,
            alpha=0.55,
            color="#2f6f73",
            edgecolors="none",
        )
        axis.plot(limits, limits, color="#b84a62", linewidth=2)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(model_name, fontsize=11, weight="bold")
        axis.set_xlabel("Actual strength")
        axis.set_ylabel("Predicted strength")
        axis.grid(alpha=0.25)
        axis.text(
            0.04,
            0.94,
            f"RMSE: {model_metrics['RMSE']:.2f}\nMAE: {model_metrics['MAE']:.2f}\nR2: {model_metrics['R2']:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            fontsize=9,
        )

    fig.suptitle("Actual vs Predicted Concrete Strength", fontsize=16, weight="bold")
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted_by_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residual_plots(metrics, predictions):
    model_names = metrics.sort_values("RMSE")["Model"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    for axis, model_name in zip(axes, model_names):
        model_predictions = predictions[predictions["Model"] == model_name]

        axis.scatter(
            model_predictions["Predicted"],
            model_predictions["Residual"],
            s=18,
            alpha=0.55,
            color="#d88c32",
            edgecolors="none",
        )
        axis.axhline(0, color="#333333", linewidth=1.5)
        axis.set_title(model_name, fontsize=11, weight="bold")
        axis.set_xlabel("Predicted strength")
        axis.set_ylabel("Residual")
        axis.grid(alpha=0.25)

    fig.suptitle("Residual Patterns by Model", fontsize=16, weight="bold")
    fig.savefig(OUTPUT_DIR / "residuals_by_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_linear_coefficients(model, feature_names):
    coefficients = model.named_steps["lr"].coef_
    return pd.Series(coefficients, index=feature_names).sort_values(key=lambda s: s.abs(), ascending=False)


def save_feature_summary(fitted_models, X_columns, best_simple_feature):
    mlr = get_linear_coefficients(fitted_models["Multiple Linear Regression"], X_columns).head(10)
    rf = pd.Series(
        fitted_models["Random Forest Regressor"].feature_importances_,
        index=X_columns,
    ).sort_values(ascending=False).head(10)
    gb = pd.Series(
        fitted_models["Gradient Boosting Regressor"].feature_importances_,
        index=X_columns,
    ).sort_values(ascending=False).head(10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    axes[0].bar([best_simple_feature], [1], color="#2f6f73")
    axes[0].set_title("Simple Linear Regression Feature", fontsize=11, weight="bold")
    axes[0].set_ylabel("Selected single predictor")
    axes[0].set_ylim(0, 1.2)
    axes[0].set_yticks([])
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].text(
        0,
        1.04,
        "Chosen by 5-fold CV RMSE",
        ha="center",
        fontsize=9,
    )

    mlr_colors = ["#2f6f73" if value >= 0 else "#b84a62" for value in mlr]
    axes[1].barh(mlr.index[::-1], mlr.values[::-1], color=mlr_colors[::-1])
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("Multiple Linear Regression Coefficients", fontsize=11, weight="bold")
    axes[1].set_xlabel("Standardized coefficient")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].spines[["top", "right", "left"]].set_visible(False)

    axes[2].barh(rf.index[::-1], rf.values[::-1], color="#5f5a8b")
    axes[2].set_title("Random Forest Feature Importance", fontsize=11, weight="bold")
    axes[2].set_xlabel("Importance")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].spines[["top", "right", "left"]].set_visible(False)

    axes[3].barh(gb.index[::-1], gb.values[::-1], color="#d88c32")
    axes[3].set_title("Gradient Boosting Feature Importance", fontsize=11, weight="bold")
    axes[3].set_xlabel("Importance")
    axes[3].grid(axis="x", alpha=0.25)
    axes[3].spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("What Each Model Uses Most", fontsize=16, weight="bold")
    fig.savefig(OUTPUT_DIR / "model_feature_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    best_simple_feature = find_best_simple_feature(X_train, y_train)
    metrics, predictions, fitted_models = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        best_simple_feature,
    )

    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)

    save_metric_comparison(metrics)
    save_prediction_scatter(metrics, predictions)
    save_residual_plots(metrics, predictions)
    save_feature_summary(fitted_models, X.columns, best_simple_feature)

    print(f"Saved visualisations to {OUTPUT_DIR.resolve()}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
