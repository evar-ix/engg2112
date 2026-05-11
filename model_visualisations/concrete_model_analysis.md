# Analysis of Concrete Strength Model Visualisations

This analysis explains how the machine-learning models predict concrete compressive strength and interprets the updated visualisations.

The models compared are:

- Simple Linear Regression
- Multiple Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor
- K-Nearest Neighbours Regressor
- Naive Bayes Strength-Range Model

The target variable is `cs`, which represents measured concrete compressive strength. The models use concrete mix and curing variables such as cement, water, binder, water-binder ratio, aggregates, supplementary cementitious materials, superplasticizer, temperature, age, and whether the sample is UHPC.

## Overall Model Performance

![Model metric comparison](model_metric_comparison.png)

The model performance graph compares the models using RMSE, MAE, and R2.

RMSE, or Root Mean Squared Error, measures the average size of prediction errors. A lower RMSE means the predicted compressive strength values are closer to the actual measured values.

MAE, or Mean Absolute Error, measures the average absolute difference between actual and predicted strength. A lower MAE is better.

R2 measures how much of the variation in compressive strength is explained by the model. An R2 value closer to 1 means the model explains more of the data and performs better.

The results are:

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Random Forest Regressor | 4.91 | 2.75 | 0.988 |
| Gradient Boosting Regressor | 6.66 | 4.94 | 0.979 |
| Support Vector Regressor | 9.55 | 6.01 | 0.956 |
| K-Nearest Neighbours Regressor | 9.85 | 4.91 | 0.953 |
| Multiple Linear Regression | 16.74 | 12.26 | 0.865 |
| Simple Linear Regression | 21.56 | 15.54 | 0.776 |
| Naive Bayes Strength-Range Model | 32.72 | 25.57 | 0.485 |

From these results, the Random Forest Regressor performs best overall. It has the lowest RMSE, the lowest MAE, and the highest R2 score.

## Actual vs Predicted Concrete Strength

![Actual vs predicted by model](actual_vs_predicted_by_model.png)

The actual vs predicted graphs show how close each model's predictions are to the real measured compressive strength values.

The diagonal line represents perfect prediction. If a point lies exactly on this line, the predicted concrete strength is equal to the actual concrete strength. Points close to the line indicate accurate predictions, while points far from the line indicate larger errors.

The Random Forest Regressor has the tightest grouping of points around the diagonal line, showing that its predicted strength values are closest to the actual values.

The Gradient Boosting Regressor also performs strongly, but its predictions are slightly more spread out than Random Forest.

The Support Vector Regressor and K-Nearest Neighbours Regressor improve on the linear models. This shows that nonlinear methods are more suitable for this concrete strength dataset than simple straight-line models.

The Multiple Linear Regression and Simple Linear Regression models are more spread out because they cannot fully capture the nonlinear behaviour of concrete strength development.

The Naive Bayes Strength-Range Model performs worst. This is expected because Naive Bayes is naturally a classification method, not a regression method. In this project it was adapted by predicting a strength range first, then converting that range into an approximate numeric strength value.

## Residual Analysis

![Residuals by model](residuals_by_model.png)

Residuals are calculated as:

```text
Residual = Actual compressive strength - Predicted compressive strength
```

A good model should have residuals scattered closely around zero. This means the model is not consistently overpredicting or underpredicting concrete strength.

The Random Forest Regressor has the smallest residual spread. This means its prediction errors are generally smaller and more balanced than the other models.

The Gradient Boosting Regressor also has relatively small residuals, which confirms that it is the second-best model.

The SVR and KNN models have wider residuals than the tree-based models, but they are still better than the linear models. This suggests they capture some nonlinear relationships, but not as effectively as Random Forest or Gradient Boosting.

The linear models and Naive Bayes model show larger residual patterns. This indicates that they miss important relationships in the data, especially interactions between water-binder ratio, curing age, binder composition, and other mix variables.

## Feature Importance and Model Behaviour

![Model feature summary](model_feature_summary.png)

The feature summary graph shows which variables are most influential for the models where feature effects can be interpreted directly.

For the Simple Linear Regression model, the selected feature is binder. This model predicts strength using only binder, so it is easy to understand but too simple for accurate concrete strength prediction.

For the Multiple Linear Regression model, the graph shows standardised coefficients. These coefficients indicate how strongly each feature contributes to the prediction. Positive coefficients increase predicted strength, while negative coefficients reduce predicted strength.

For the Random Forest and Gradient Boosting models, the most important features include water-binder ratio and age. This agrees with concrete technology because lower water-binder ratio generally increases strength, and curing age strongly affects strength gain.

The tree-based models perform best because they can capture nonlinear behaviour and interactions between variables. For example, the effect of age may depend on the water-binder ratio, cementitious material composition, and superplasticizer content.

SVR, KNN, and Naive Bayes are included in the performance, actual-vs-predicted, and residual visualisations. They are not included in the feature-importance graph because their feature effects are not as directly interpretable using the same method as linear coefficients or tree feature importances.

## How the Models Calculate Compressive Strength

The Simple Linear Regression model calculates compressive strength using one feature only. In this case, it uses binder. The model fits a straight-line equation:

```text
Predicted strength = intercept + coefficient x binder
```

The Multiple Linear Regression model calculates compressive strength using all input variables in one linear equation:

```text
Predicted strength = intercept + coefficient1 x feature1 + coefficient2 x feature2 + ...
```

The Random Forest Regressor builds many decision trees. Each tree predicts strength from different splits in the data, such as water-binder ratio, age, cement content, or aggregate values. The final prediction is the average of all tree predictions.

The Gradient Boosting Regressor also uses decision trees, but it builds them one after another. Each new tree tries to correct the errors made by the previous trees.

The Support Vector Regressor uses a kernel method to fit a flexible nonlinear prediction surface. This allows it to model curved relationships between the input variables and compressive strength.

The K-Nearest Neighbours Regressor predicts strength by finding similar concrete mixes in the training data. It then estimates strength based on the strengths of those nearby mixes.

The Naive Bayes Strength-Range Model first divides compressive strength into ranges. It predicts which range a concrete mix belongs to, then converts that predicted range into an approximate strength value. This makes it useful as a comparison baseline, but it is not as natural for continuous strength prediction as the regression models.

## Final Conclusion

Based on the visualisations and the model metrics, the Random Forest Regressor is still the best model for predicting concrete compressive strength.

It performs best because:

- It has the lowest RMSE, meaning it has the smallest overall prediction error.
- It has the lowest MAE, meaning its average prediction error is smallest.
- It has the highest R2 score, meaning it explains the most variation in concrete strength.
- Its actual vs predicted graph shows points closest to the perfect prediction line.
- Its residual plot shows the smallest and most balanced errors.
- It can model nonlinear relationships between mix design, curing age, and compressive strength.

The final ranking is:

1. Random Forest Regressor
2. Gradient Boosting Regressor
3. Support Vector Regressor
4. K-Nearest Neighbours Regressor
5. Multiple Linear Regression
6. Simple Linear Regression
7. Naive Bayes Strength-Range Model

Therefore, the recommended model is:

**Random Forest Regressor**
