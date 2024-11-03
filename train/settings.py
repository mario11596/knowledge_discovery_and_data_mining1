from enum import Enum

class ModelTypes(Enum):
    CATBOOST = "Catboost"
    XGBOOST = "XGBoost"
    LINEARREGRESSION = "LinearRegression"
    DECISIONTREEREGRESSOR = "DecisionTreeRegressor"
    MLPREGRESSOR = "MLPRegressor"
    SEQUENTIAL = "Sequential"


class Settings:
    # Model type
    model_type = ModelTypes.CATBOOST

    # Test split
    test_size = 0.1

    # Categorical features
    categorical_features = ["Location", "PoolQuality", "HasPhotovoltaics", "HasFireplace", "HouseColor", "HeatingType",
                            "HasFiberglass", "IsFurnished", "KitchensQuality", "BathroomsQuality", "BedroomsQuality",
                            "LivingRoomsQuality", "WindowMaterial"]

    if model_type == ModelTypes.DECISIONTREEREGRESSOR:
        model_params = {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            "splitter": ['best'],
            "max_depth": [5, 6, 7, 8, 9, 10],
            'max_features': ['sqrt', 'log2', None],
            "random_seed": 42
        }
    elif model_type == ModelTypes.LINEARREGRESSION:
        model_params = {
            "random_seed": 42,
        }

    elif model_type == ModelTypes.MLPREGRESSOR:
        model_params = {
            "random_seed": 42,
        }
    elif model_type == ModelTypes.CATBOOST:
        # Model related settings
        model_params = {
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 8,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "task_type": "GPU",
            "cat_features": categorical_features,
            "od_wait": 100,
            "use_best_model": True,
            "random_seed": 42,
            "verbose": 100
        }

    # Data related settings
    data_params = {
        "data_path": "./files/preprocessed_3_without_DateSinceForSale_HasPhotovoltaics_filled.csv",
        "target_column": "Price",
    }

    # Experiment related settings
    experiment_params = {
        "additional_description": "Experiment ran with NO cross-val. Split training set into train/validation sets. "
                                  "Metrics saved represent the results on the final test set. ",
        "author": "alex"
    }


settings = Settings()
