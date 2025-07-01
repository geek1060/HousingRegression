import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import load_data, prepare_data, evaluate_model

def run_regression():
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"{name} -> MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.2f}")

    result_df = pd.DataFrame(results).T
    result_df.to_csv("regression_results.csv")
    print("\nModel comparison saved to regression_results.csv")

if __name__ == "__main__":
    run_regression()