import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import load_data, prepare_data, evaluate_model
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def run_regression_with_tuning():
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    models_and_params = {
        "Linear Regression": (LinearRegression(), {}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=42),
            {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
        )
    }

    results = {}

    for name, (model, params) in models_and_params.items():
        print(f"Tuning {name}...")
        best_model, best_params = tune_hyperparameters(model, params, X_train, y_train)
        print(f"Best Params for {name}: {best_params}")
        metrics = evaluate_model(best_model, X_test, y_test)
        results[name] = metrics
        print(f"{name} -> MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.2f}")

    result_df = pd.DataFrame(results).T
    result_df.to_csv("hyperparameter_results.csv")
    print("\nHyperparameter tuning results saved to hyperparameter_results.csv")

if __name__ == "__main__":
    run_regression_with_tuning()