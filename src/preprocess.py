import pandas as pd

# Column setup
ALL_COLUMNS = [
    "cement", "ggbs", "flyash", "silica_fume",
    "limestone_powder", "quartz_powder", "nano_silica",
    "water", "superplasticizer",
    "coarse_agg", "fine_agg",
    "temperature",
    "age", "cs", "is_uhpc"
]

def align_columns(df):
    """Add any missing columns as 0, then return only ALL_COLUMNS in order."""
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[ALL_COLUMNS].copy()

def add_engineered_features(df):
    """Compute binder total and water-binder ratio."""
    df = df.copy()
    df["binder"] = (
        df["cement"] + df["flyash"] + df["ggbs"] +
        df["silica_fume"] + df["limestone_powder"] +
        df["quartz_powder"] + df["nano_silica"]
    )
    df["water_binder_ratio"] = df["water"] / (df["binder"] + 1e-6)
    return df

# IIT Bhubaneswar dataset (.txt, space-separated, 4-row header)
def load_and_process_iit(path):
    df = pd.read_csv(path, sep=r'\s+', skiprows=4)
    df = df.rename(columns={
        "GGBS":        "ggbs",
        "MK":          "silica_fume",       # Metakaolin mapped to silica_fume slot
        "SP":          "superplasticizer",
        "NCA_20_DOWN": "nca_20",
        "NCA_10_DOWN": "nca_10",
        "RCA_20_Down": "rca_20",
        "RCA_10DOWN":  "rca_10",
        "SAND":        "fine_agg",
        "AGE":         "age",
        "CS":          "cs",
        "flyash":      "flyash",
        "cement":      "cement",
        "water":       "water",
    })
    df["coarse_agg"] = df["nca_20"] + df["nca_10"] + df["rca_20"] + df["rca_10"]
    df["is_uhpc"] = 0
    return align_columns(df)

# UCI dataset: Concrete_Data.csv (long column names, has GGBS)
def load_and_process_uci(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()     # strip whitespace from col names
    df = df.rename(columns={
        "Cement (component 1)(kg in a m^3 mixture)":             "cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "ggbs",
        "Fly Ash (component 3)(kg in a m^3 mixture)":            "flyash",
        "Water  (component 4)(kg in a m^3 mixture)":             "water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)":   "superplasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":  "coarse_agg",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)":     "fine_agg",
        "Age (day)":                                             "age",
        "Concrete compressive strength(MPa, megapascals)":       "cs",
    })
    df["is_uhpc"] = 0
    return align_columns(df)

# Standard dataset: concrete_compressive_strength_dataset.csv
def load_and_process_std(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "cement":       "cement",
        "slag":         "ggbs",
        "ash":          "flyash",
        "water":        "water",
        "superplastic": "superplasticizer",
        "coarseagg":    "coarse_agg",
        "fineagg":      "fine_agg",
        "age":          "age",
        "strength":     "cs",
    })
    df["is_uhpc"] = 0
    return align_columns(df)

# UHPC dataset: Data_UHPC.csv
def load_and_process_uhpc(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "C":   "cement",
        "S":   "ggbs",
        "SF":  "silica_fume",
        "LP":  "limestone_powder",
        "QP":  "quartz_powder",
        "FA":  "flyash",
        "NS":  "nano_silica",
        "A":   "coarse_agg",
        "W":   "water",
        "Fi":  "fine_agg",
        "SP":  "superplasticizer",
        "T":   "temperature",
        "Age": "age",
        "CS":  "cs",
    })
    df["is_uhpc"] = 1
    return align_columns(df)

# Main pipeline
def create_final_datasets(iit_path, uci_path, std_path, uhpc_path):
    df_iit  = load_and_process_iit(iit_path)
    df_uci  = load_and_process_uci(uci_path)
    df_std  = load_and_process_std(std_path)
    df_uhpc = load_and_process_uhpc(uhpc_path)

    df_all = pd.concat([df_iit, df_uci, df_std, df_uhpc], ignore_index=True)

    df_normal   = add_engineered_features(
        df_all[df_all["is_uhpc"] == 0].reset_index(drop=True)
    )
    df_uhpc_out = add_engineered_features(
        df_all[df_all["is_uhpc"] == 1].reset_index(drop=True)
    )
    df_all = add_engineered_features(df_all.reset_index(drop=True))

    return df_normal, df_uhpc_out, df_all

def save_datasets(df_normal, df_uhpc_only, df_all):
    df_normal.to_csv("data/processed/normal_concrete.csv", index=False)
    df_uhpc_only.to_csv("data/processed/uhpc_concrete.csv", index=False)
    df_all.to_csv("data/processed/combined_concrete.csv", index=False)

# Main code
if __name__ == "__main__":
    df_normal, df_uhpc, df_all = create_final_datasets(
        iit_path  = "data/raw/raw_data_concrete_iit.txt",
        uci_path  = "data/raw/Concrete_Data.csv",
        std_path  = "data/raw/concrete_compressive_strength_dataset.csv",
        uhpc_path = "data/raw/Data_UHPC.csv",
    )
    save_datasets(df_normal, df_uhpc, df_all)
    print(f"Saved — Normal: {len(df_normal)} rows | UHPC: {len(df_uhpc)} rows | Combined: {len(df_all)} rows")
