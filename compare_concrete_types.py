import math
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "model_details"

DATASETS = {
    "normal": DATA_DIR / "updated_normal_concrete.csv",
    "uhpc": DATA_DIR / "update_uhpc_concrete.csv",
}

TARGET = "cs"
DROP_COLUMNS = {TARGET, "is_uhpc"}
RANDOM_STATE = 42
TEST_SIZE = 0.2


def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def build_models():
    return {
        "Multiple Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }


def load_dataset(path):
    df = pd.read_csv(path)
    feature_columns = [column for column in df.columns if column not in DROP_COLUMNS]
    return df, feature_columns


def model_feature_importance(model_name, fitted_model, feature_columns):
    if model_name == "Multiple Linear Regression":
        values = fitted_model.named_steps["lr"].coef_
        importance_column = "standardized_coefficient"
        importance_values = values
        absolute_values = pd.Series(values).abs().tolist()
    else:
        importance_column = "importance"
        importance_values = fitted_model.feature_importances_
        absolute_values = importance_values

    rows = []
    for feature, value, absolute_value in zip(feature_columns, importance_values, absolute_values):
        rows.append({
            "feature": feature,
            importance_column: value,
            "absolute_importance": absolute_value,
        })
    return pd.DataFrame(rows)


def train_dataset(label, df, feature_columns):
    X = df[feature_columns]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metric_rows = []
    importance_frames = []
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metric_rows.append({
            "concrete_type": label,
            "model": model_name,
            "rows": len(df),
            "features": len(feature_columns),
            "RMSE": root_mean_squared_error(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions),
            "R2": r2_score(y_test, predictions),
        })

        importance = model_feature_importance(model_name, model, feature_columns)
        importance.insert(0, "model", model_name)
        importance.insert(0, "concrete_type", label)
        importance_frames.append(importance)

    return pd.DataFrame(metric_rows), pd.concat(importance_frames, ignore_index=True)


def composition_summary(datasets):
    composition_features = [
        "cement",
        "ggbs",
        "flyash",
        "silica_fume",
        "limestone_powder",
        "quartz_powder",
        "nano_silica",
        "water",
        "superplasticizer",
        "coarse_agg",
        "fine_agg",
        "steel_fiber",
        "sand_binder_ratio",
        "binder",
        "water_binder_ratio",
    ]
    rows = []
    for label, df in datasets.items():
        for feature in composition_features:
            rows.append({
                "concrete_type": label,
                "feature": feature,
                "mean": df[feature].mean(),
                "median": df[feature].median(),
                "std": df[feature].std(),
                "min": df[feature].min(),
                "max": df[feature].max(),
            })
    summary = pd.DataFrame(rows)

    mean_comparison = summary.pivot(index="feature", columns="concrete_type", values="mean")
    mean_comparison["uhpc_minus_normal_mean"] = mean_comparison["uhpc"] - mean_comparison["normal"]
    mean_comparison = mean_comparison.reset_index()
    return summary, mean_comparison


def top_features(importance, model_name, concrete_type, limit=8):
    rows = (
        importance[
            (importance["model"] == model_name)
            & (importance["concrete_type"] == concrete_type)
        ]
        .sort_values("absolute_importance", ascending=False)
        .head(limit)
    )
    return ", ".join(rows["feature"].tolist())


def markdown_table(df):
    table = df.copy()
    for column in table.select_dtypes(include="number").columns:
        table[column] = table[column].map(lambda value: f"{value:.4f}")

    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_report(metrics, importance, mean_comparison):
    best_metrics = metrics.sort_values(["concrete_type", "RMSE"]).groupby("concrete_type").head(1)
    composition_differences = mean_comparison.reindex(
        mean_comparison["uhpc_minus_normal_mean"].abs().sort_values(ascending=False).index
    ).head(8)

    lines = [
        "# Normal Concrete vs UHPC Feature Importance",
        "",
        "Separate models were trained on the updated processed datasets:",
        "- `datasets/updated_normal_concrete.csv`",
        "- `datasets/update_uhpc_concrete.csv`",
        "",
        "The target is `cs`. Predictors include mix composition, curing `temperature`, `age`, and derived ratios; `is_uhpc` is excluded because each model is trained on only one concrete type.",
        "",
        "## Best model by concrete type",
        "",
        markdown_table(best_metrics[["concrete_type", "model", "RMSE", "MAE", "R2"]]),
        "",
        "## Top Gradient Boosting features",
        "",
        f"- Normal concrete: {top_features(importance, 'Gradient Boosting Regressor', 'normal')}",
        f"- UHPC: {top_features(importance, 'Gradient Boosting Regressor', 'uhpc')}",
        "",
        "## Top Random Forest features",
        "",
        f"- Normal concrete: {top_features(importance, 'Random Forest Regressor', 'normal')}",
        f"- UHPC: {top_features(importance, 'Random Forest Regressor', 'uhpc')}",
        "",
        "## Largest mean composition differences",
        "",
        markdown_table(composition_differences),
        "",
    ]
    (OUTPUT_DIR / "normal_vs_uhpc_feature_importance_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = {}
    feature_sets = {}
    for label, path in DATASETS.items():
        df, feature_columns = load_dataset(path)
        loaded[label] = df
        feature_sets[label] = feature_columns

    if feature_sets["normal"] != feature_sets["uhpc"]:
        raise ValueError("Normal and UHPC datasets must have matching feature columns.")

    metric_frames = []
    importance_frames = []
    for label, df in loaded.items():
        metrics, importance = train_dataset(label, df, feature_sets[label])
        metric_frames.append(metrics)
        importance_frames.append(importance)

    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["concrete_type", "RMSE"])
    importance = pd.concat(importance_frames, ignore_index=True)
    composition, mean_comparison = composition_summary(loaded)

    metrics.to_csv(OUTPUT_DIR / "normal_vs_uhpc_model_metrics.csv", index=False)
    importance.sort_values(
        ["model", "feature", "concrete_type"],
    ).to_csv(OUTPUT_DIR / "normal_vs_uhpc_feature_importance.csv", index=False)
    composition.to_csv(OUTPUT_DIR / "normal_vs_uhpc_composition_summary.csv", index=False)
    mean_comparison.to_csv(OUTPUT_DIR / "normal_vs_uhpc_mean_composition_difference.csv", index=False)
    write_report(metrics, importance, mean_comparison)

    print(metrics.to_string(index=False))
    print(f"\nSaved comparison outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
