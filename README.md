# Concrete Compressive Strength Prediction

This project predicts concrete compressive strength using machine learning in Python.

## Models Used
- Simple Linear Regression
- Multiple Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor
- K-Nearest Neighbours Regressor
- Neural Network Regressor
- Naive Bayes Strength-Range Model

## Dataset
The dataset contains 1030 rows and 14 columns, including the target variable `strength`.

## Goal
To predict concrete compressive strength based on ingredients and curing age.

## How to Run 

```bash
pip install -r requirements.txt
python concrete_strength_model.py
```

## Interactive Demo

Run the Random Forest prediction demo with:

```bash
python random_forest_strength_demo.py
```

Enter the concrete mix values when prompted. The demo calculates binder content
and water-to-binder ratio, then prints the predicted compressive strength in MPa.

## Visualisations

Generate model comparison charts with:

```bash
python model_visualisations.py
```

The charts and supporting CSV files are saved in `model_visualisations/`.
