# Normal Concrete vs UHPC Feature Importance

Separate models were trained on the updated processed datasets:
- `datasets/updated_normal_concrete.csv`
- `datasets/update_uhpc_concrete.csv`

The target is `cs`. Predictors include mix composition, curing `temperature`, `age`, and derived ratios; `is_uhpc` is excluded because each model is trained on only one concrete type.

## Best model by concrete type

| concrete_type | model | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| normal | Random Forest Regressor | 2.6462 | 1.6130 | 0.9760 |
| uhpc | Random Forest Regressor | 8.0335 | 4.9489 | 0.9843 |

## Top Gradient Boosting features

- Normal concrete: age, water_binder_ratio, binder, cement, flyash, water, superplasticizer, fine_agg
- UHPC: superplasticizer, binder, age, cement, water_binder_ratio, fine_agg, silica_fume, water

## Top Random Forest features

- Normal concrete: water_binder_ratio, age, cement, binder, flyash, coarse_agg, fine_agg, ggbs
- UHPC: superplasticizer, age, binder, water_binder_ratio, fine_agg, coarse_agg, cement, silica_fume

## Largest mean composition differences

| feature | normal | uhpc | uhpc_minus_normal_mean |
| --- | --- | --- | --- |
| fine_agg | 762.9581 | 39.8494 | -723.1086 |
| binder | 413.4996 | 714.4579 | 300.9582 |
| cement | 285.9688 | 524.4657 | 238.4969 |
| coarse_agg | 970.6699 | 817.1834 | -153.4865 |
| silica_fume | 0.9573 | 97.3598 | 96.4025 |
| ggbs | 72.7355 | 17.9298 | -54.8057 |
| water | 180.6373 | 128.0211 | -52.6162 |
| flyash | 53.8381 | 18.6854 | -35.1527 |
