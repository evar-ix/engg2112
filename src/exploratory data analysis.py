import pandas as pd
import matplotlib.pyplot as plt

#   CORRELATION MATRIX

# Load dataset
df = pd.read_csv(
    "C:/Users/kevin/.vscode-shared/engg2112/engg2112/data/processed/update_combined_concrete.csv"
)

# Select only numeric columns
corr = df.corr(numeric_only=True)

# Plot
plt.figure(figsize=(12,10))

plt.imshow(corr, aspect='auto')

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix - Combined Concrete Dataset")

plt.tight_layout()

plt.savefig(
    "C:/Users/kevin/.vscode-shared/engg2112/engg2112/data/processed",
    dpi=300
)

plt.show()

#   DISTRIBUTION PLOT

# Load datasets
df_normal = pd.read_csv(
    "C:/Users/kevin/.vscode-shared/engg2112/engg2112/data/processed/updated_normal_concrete.csv"
)

df_uhpc = pd.read_csv(
    "C:/Users/kevin/.vscode-shared/engg2112/engg2112/data/processed/update_uhpc_concrete.csv"
)

# Plot distributions
plt.figure(figsize=(10,6))

plt.hist(
    df_normal["cs"],
    bins=30,
    alpha=0.7,
    label="Normal Concrete"
)

plt.hist(
    df_uhpc["cs"],
    bins=30,
    alpha=0.7,
    label="UHPC"
)

plt.xlabel("Compressive Strength (MPa)")
plt.ylabel("Frequency")

plt.title(
    "Compressive Strength Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "C:/Users/kevin/.vscode-shared/engg2112/engg2112/data/processed",
    dpi=300
)

plt.show()

#   OUTLIER ANALYSIS
# Create boxplot
plt.figure(figsize=(8,6))

plt.boxplot([
    df_normal["cs"],
    df_uhpc["cs"]
])

plt.xticks(
    [1,2],
    ["Normal Concrete", "UHPC"]
)

plt.ylabel("Compressive Strength (MPa)")

plt.title(
    "Outlier Analysis of Concrete Strength"
)

plt.tight_layout()

plt.savefig(
    "C:/Users/kevin/.vscode-shared/engg2112/engg2112/data/processed",
    dpi=300
)

plt.show()