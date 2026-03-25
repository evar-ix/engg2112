import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("Concrete Compressive Strength.csv")

# 2. Basic dataset scan
print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# 3. Define features and target
X = df.drop(columns=["strength"])
y = df["strength"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Simple Linear Regression (using one feature only)
# This shows basic linear regression clearly
X_simple = df[["cement"]]
X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

simple_lr = LinearRegression()
simple_lr.fit(X_simple_train, y_simple_train)
simple_preds = simple_lr.predict(X_simple_test)

print("\n--- Simple Linear Regression (cement only) ---")
print("MAE:", mean_absolute_error(y_simple_test, simple_preds))
print("RMSE:", mean_squared_error(y_simple_test, simple_preds) ** 0.5)
print("R2:", r2_score(y_simple_test, simple_preds))
print("Coefficient:", simple_lr.coef_[0])
print("Intercept:", simple_lr.intercept_)

# 6. Multiple Linear Regression
multi_lr = LinearRegression()
multi_lr.fit(X_train, y_train)
multi_preds = multi_lr.predict(X_test)

print("\n--- Multiple Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, multi_preds))
print("RMSE:", mean_squared_error(y_test, multi_preds) ** 0.5)
print("R2:", r2_score(y_test, multi_preds))

# 7. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\n--- Random Forest Regressor ---")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("RMSE:", mean_squared_error(y_test, rf_preds) ** 0.5)
print("R2:", r2_score(y_test, rf_preds))

# 8. Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

print("\n--- Gradient Boosting Regressor ---")
print("MAE:", mean_absolute_error(y_test, gb_preds))
print("RMSE:", mean_squared_error(y_test, gb_preds) ** 0.5)
print("R2:", r2_score(y_test, gb_preds))

# 9. Compare model performance
results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        mean_absolute_error(y_simple_test, simple_preds),
        mean_absolute_error(y_test, multi_preds),
        mean_absolute_error(y_test, rf_preds),
        mean_absolute_error(y_test, gb_preds)
    ],
    "RMSE": [
        mean_squared_error(y_simple_test, simple_preds) ** 0.5,
        mean_squared_error(y_test, multi_preds) ** 0.5,
        mean_squared_error(y_test, rf_preds) ** 0.5,
        mean_squared_error(y_test, gb_preds) ** 0.5
    ],
    "R2": [
        r2_score(y_simple_test, simple_preds),
        r2_score(y_test, multi_preds),
        r2_score(y_test, rf_preds),
        r2_score(y_test, gb_preds)
    ]
})

print("\n--- Model Comparison ---")
print(results)

# 10. Correlation with target
print("\nCorrelation with strength:")
print(df.corr(numeric_only=True)["strength"].sort_values(ascending=False))