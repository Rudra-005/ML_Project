import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(eval_metric="rmse"),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.7, 0.8, 0.9],
                    "n_estimators": [50, 100, 200],
                },

                "Linear Regression": {},

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                },

                "XGBoost": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [50, 100, 200],
                },

                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "iterations": [50, 100],
                },

                "AdaBoost": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [50, 100, 200],
                },
            }

            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            logging.info(f"Best Model: {best_model_name} | R2 Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance")

            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model,
            )

            predictions = best_model.predict(X_test)
            return r2_score(y_test, predictions)

        except Exception as e:
            raise CustomException(e, sys)
