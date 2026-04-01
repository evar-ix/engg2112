import pandas as pd

# Function space

def load_csv_data(path):
    df = pd.read_csv(path)
    return df

def process_iit(df):
    df["coarse_agg"] = df["nca_20"] + df["nca_10"]
    df["fine_agg"] = df["sand"]
    df["rca_agg"] = df["rca_20"] + df["rca_10"]

    df = df.rename(columns={"sp": "superplasticizer"})
    df["is_uhpc"] = 0  # mark as normal concrete
    return df

def standardise_columns(df):
    df = df.rename(columns={
        "Cement": "cement",
        "Fly Ash": "flyash",
        "Water": "water",
        "Age": "age",
        "Concrete compressive strength": "cs"
    })
    df["is_uhpc"] = 0  # mark as normal concrete
    return df

def process_uhpc(df):
    df = df.rename(columns={
        "C": "cement",
        "S": "ggbs",
        "SF": "silica_fume",
        "LP": "limestone_powder",
        "QP": "quartz_powder",
        "FA": "flyash",
        "NS": "nano_silica",
        "A": "coarse_agg",
        "W": "water",
        "Fi": "fine_agg",
        "SP": "superplasticizer",
        "T": "temperature",
        "Age": "age",
        "CS": "cs"
    })

    df["is_uhpc"] = 1  # mark data row as UHPC, to seperate from other concretes
    return df

# Align columns for wide dataset
ALL_COLUMNS = [
    "cement","ggbs","flyash","silica_fume",
    "limestone_powder","quartz_powder","nano_silica",
    "water","superplasticizer",
    "coarse_agg","fine_agg",
    "temperature",
    "age","cs","is_uhpc"
]

def align_columns(df):
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # missing materials/features = 0
    return df[ALL_COLUMNS]

# Merge and separate UHPC
def create_final_datasets(iit_path, standard_path, uhpc_path):
    # Load datasets
    df_iit = load_csv_data(iit_path)
    df_iit = process_iit(df_iit)
    df_iit = align_columns(df_iit)

    df_std = load_csv_data(standard_path)
    df_std = standardise_columns(df_std)
    df_std = align_columns(df_std)

    df_uhpc = load_csv_data(uhpc_path)
    df_uhpc = process_uhpc(df_uhpc)
    df_uhpc = align_columns(df_uhpc)

    # Merge everything
    df_all = pd.concat([df_iit, df_std, df_uhpc], ignore_index=True)

    # Separate datasets
    df_normal = df_all[df_all["is_uhpc"] == 0].reset_index(drop=True)
    df_uhpc_only = df_all[df_all["is_uhpc"] == 1].reset_index(drop=True)

    # Feature to be classified
    for df in [df_normal, df_uhpc_only]:
        df["binder"] = df["cement"] + df["flyash"] + df["ggbs"] + df["silica_fume"] + df["limestone_powder"] + df["quartz_powder"] + df["nano_silica"]
        df["water_binder_ratio"] = df["water"] / (df["binder"] + 1e-6)

    return df_normal, df_uhpc_only, df_all

def save_datasets(df_normal, df_uhpc_only, df_all):
    df_normal.to_csv("data/processed/normal_concrete.csv", index=False)
    df_uhpc_only.to_csv("data/processed/uhpc_concrete.csv", index=False)
    df_all.to_csv("data/processed/combined_concrete.csv", index=False)

# Main code
if __name__ == "__main__":
    df_normal, df_uhpc, df_all = create_final_datasets(
        "data/raw/Concrete_Data.csv",
        "data/raw/concrete_compressive_strength_dataset.csv",
        "data/raw/Data UHPC.csv"
    )
    save_datasets(df_normal, df_uhpc, df_all)
    print("Datasets saved: normal, UHPC, combined.")
