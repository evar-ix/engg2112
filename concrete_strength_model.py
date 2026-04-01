import math
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA_PATH = 'combined_concrete.csv'  # change if needed
TARGET = 'cs'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Pick the best single feature for simple linear regression
    best_feature = None
    best_rmse = float('inf')
    for col in X.columns:
        simple_model = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        cv_rmse = -cross_val_score(
            simple_model,
            X_train[[col]],
            y_train,
            scoring='neg_root_mean_squared_error',
            cv=5,
        ).mean()
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_feature = col

    models = {
        f'Simple Linear Regression ({best_feature})': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'Multiple Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ]),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'Gradient Boosting Regressor': GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        xtr = X_train[[best_feature]] if 'Simple Linear Regression' in name else X_train
        xte = X_test[[best_feature]] if 'Simple Linear Regression' in name else X_test

        model.fit(xtr, y_train)
        pred = model.predict(xte)
        rows.append({
            'Model': name,
            'RMSE': math.sqrt(mean_squared_error(y_test, pred)),
            'MAE': mean_absolute_error(y_test, pred),
            'R2': r2_score(y_test, pred),
        })
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values('RMSE')
    print('\nModel comparison\n')
    print(results.to_string(index=False))

    mlr = fitted['Multiple Linear Regression']
    mlr_importance = pd.Series(
        mlr.named_steps['lr'].coef_, index=X.columns, name='standardized_coefficient'
    )
    mlr_importance = mlr_importance.reindex(
        mlr_importance.abs().sort_values(ascending=False).index
    )
    print('\nTop multiple linear regression features\n')
    print(mlr_importance.head(10).to_string())

    rf = fitted['Random Forest Regressor']
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print('\nTop random forest features\n')
    print(rf_importance.head(10).to_string())

    gb = fitted['Gradient Boosting Regressor']
    gb_importance = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print('\nTop gradient boosting features\n')
    print(gb_importance.head(10).to_string())


if __name__ == '__main__':
    main()
