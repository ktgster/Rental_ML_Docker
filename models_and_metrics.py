from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import numpy as np
import pickle
import json


class Model:
    def __init__(self, model_type="linear"):
        # Setting the model_type attribute
        self.model_type = model_type
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            self.model = RandomForestRegressor()
        elif model_type == "xgboost":
            self.model = xgb.XGBRegressor()
        elif model_type == "svr":
            self.model = SVR()
        elif model_type == "decision_tree":
            self.model = DecisionTreeRegressor()
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor()
        elif model_type == "ridge":
            self.model = Ridge()
        elif model_type == "lasso":
            self.model = Lasso()
        else:
            raise ValueError(
                "Invalid model_type. Supported types: linear, random_forest, xgboost, svr, decision_tree, gradient_boosting, ridge, lasso"
            )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename, columns):
        with open(filename, "wb") as file:
            pickle.dump({"model": self.model, "columns": columns}, file)

    def save_structure(self, filename):
        model_structure = {"model_type": self.model_type}
        with open(filename, "w") as file:
            json.dump(model_structure, file)


def train_and_predict(
    models_list, X_train, X_test, y_train, y_test, feature_stats, label_stats
):
    # Scale x_test once before looping through models
    X_test["sq_feet_y"] = (X_test["sq_feet_y"] - feature_stats["sq_feet_y"]["min"]) / (
        feature_stats["sq_feet_y"]["max"] - feature_stats["sq_feet_y"]["min"]
    )

    for model_type in models_list:
        # Train model
        current_model = Model(model_type=model_type)
        current_model.train(X_train, y_train)

        # Predict
        predictions = current_model.predict(X_test)

        # Rescale the predictions using inverse of min-max scaling
        predictions_rescaled = (
            predictions
            * (label_stats["price_y"]["max"] - label_stats["price_y"]["min"])
        ) + label_stats["price_y"]["min"]

        # Compute metrics
        metrics = compute_regression_metrics(
            y_true=y_test.tolist(), y_pred=predictions_rescaled.tolist()
        )

        # print metrics
        print(f"Model: {model_type} - Metrics: {metrics}")
        print(
            f"Model: {model_type} pkl, json, metrics file saved to models/{model_type}"
        )

        # save model pkl file
        current_model.save_model(f"models/{model_type}.pkl", list(X_train.columns))
        # save model weights in json
        current_model.save_structure(f"models/{model_type}.json")
        # save metrics to .py file
        with open(f"models/{model_type}_metrics.py", "w") as f:
            f.write("metrics = " + str(metrics) + "\n")


def compute_regression_metrics(y_true, y_pred):
    """
    Compute common regression metrics: MSE, RMSE, R^2, MAE, MBD, MAPE, and Median Absolute Error.

    Parameters:
    - y_true: List, true target values.
    - y_pred: List, predicted target values.

    Returns:
    - Dictionary containing the metrics.
    """

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # MSE
    mse_val = ((y_true - y_pred) ** 2).mean()

    # RMSE
    rmse_val = np.sqrt(mse_val)

    # R^2
    y_mean = y_true.mean()
    total_variance = ((y_true - y_mean) ** 2).sum()
    r2_val = 1 - (mse_val * len(y_true) / total_variance)

    # MAE
    mae_val = np.abs(y_true - y_pred).mean()

    # MBD
    mbd_val = (y_true - y_pred).mean()

    # MAPE
    mape_val = (np.abs((y_true - y_pred) / y_true).mean()) * 100

    # Median Absolute Error
    medae_val = np.median(np.abs(y_true - y_pred))

    return {
        "MSE": mse_val,
        "RMSE": rmse_val,
        "R^2": r2_val,
        "MAE": mae_val,
        "MBD": mbd_val,
        "MAPE": mape_val,
        "Median Absolute Error": medae_val,
    }
