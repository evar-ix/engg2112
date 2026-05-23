import pandas as pd
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = "combined_concrete.csv"
TARGET = "cs"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
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
    "temperature",
    "age",
    "is_uhpc",
    "binder",
    "water_binder_ratio",
]

MIX_INPUTS = [
    ("cement", "Cement content (kg/m3)", 500.0),
    ("ggbs", "GGBS / slag content (kg/m3)", 0.0),
    ("flyash", "Fly ash content (kg/m3)", 0.0),
    ("silica_fume", "Silica fume content (kg/m3)", 0.0),
    ("limestone_powder", "Limestone powder content (kg/m3)", 0.0),
    ("quartz_powder", "Quartz powder content (kg/m3)", 0.0),
    ("nano_silica", "Nano silica content (kg/m3)", 0.0),
    ("water", "Water content (kg/m3)", 160.0),
    ("superplasticizer", "Superplasticizer content (kg/m3)", 5.0),
    ("coarse_agg", "Coarse aggregate content (kg/m3)", 800.0),
    ("fine_agg", "Fine aggregate content (kg/m3)", 700.0),
    ("temperature", "Curing temperature (C)", 20.0),
    ("age", "Curing age (days)", 28.0),
]


def train_random_forest():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def ask_float(label, default):
    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if raw_value == "":
            return default
        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a number.")


def ask_yes_no(label, default=False):
    default_text = "y" if default else "n"
    while True:
        raw_value = input(f"{label} (y/n) [{default_text}]: ").strip().lower()
        if raw_value == "":
            return int(default)
        if raw_value in {"y", "yes"}:
            return 1
        if raw_value in {"n", "no"}:
            return 0
        print("Please enter y or n.")


def collect_mix_values():
    print("\nEnter the concrete mix characteristics.")
    print("Press Enter to use the default value shown in brackets.\n")

    values = {}
    for key, label, default in MIX_INPUTS:
        values[key] = ask_float(label, default)

    values["is_uhpc"] = ask_yes_no("Is this UHPC?", default=False)

    binder_components = [
        "cement",
        "ggbs",
        "flyash",
        "silica_fume",
        "limestone_powder",
        "quartz_powder",
        "nano_silica",
    ]
    values["binder"] = sum(values[component] for component in binder_components)

    if values["binder"] <= 0:
        raise ValueError("Binder must be greater than 0.")

    values["water_binder_ratio"] = values["water"] / values["binder"]
    return values


def predict_strength(model, values):
    row = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
    return model.predict(row)[0]


def main():
    print("Concrete Compressive Strength Demo")
    print("Model: Random Forest Regressor")
    print("Training model from combined_concrete.csv...")
    model = train_random_forest()

    values = collect_mix_values()
    prediction = predict_strength(model, values)

    print("\nPrediction result")
    print("-----------------")
    print(f"Binder content: {values['binder']:.2f} kg/m3")
    print(f"Water-to-binder ratio: {values['water_binder_ratio']:.3f}")
    print(f"Predicted compressive strength: {prediction:.2f} MPa")
    print("\nNote: This is an estimate only and does not replace certified lab testing.")


if __name__ == "__main__":
    main()
