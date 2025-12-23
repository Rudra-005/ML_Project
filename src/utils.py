import os
import sys
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params_grid = param.get(model_name, {})  # ✅ SAFE ACCESS

            if params_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params_grid,
                    cv=3,
                    n_jobs=-1,
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            score = r2_score(y_test, y_test_pred)
            report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
        
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)