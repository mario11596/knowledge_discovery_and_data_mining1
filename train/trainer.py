import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from keras import Input
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Concatenate

from .settings import settings, ModelTypes
from sklearn.model_selection import GridSearchCV


class Trainer:
    def __init__(self, model_type, logs_dir):
        self.model_type = model_type
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()
        self.logs_dir = logs_dir

        if model_type == ModelTypes.CATBOOST:
            self.model = CatBoostRegressor()
        elif model_type == ModelTypes.LINEARREGRESSION:
            self.model = LinearRegression()

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        elif model_type == ModelTypes.DECISIONTREEREGRESSOR:
            self.model = DecisionTreeRegressor()

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
        elif model_type == ModelTypes.MLPREGRESSOR:

            self.model = MLPRegressor()

        elif model_type == ModelTypes.XGBOOST:
            self.model = XGBRegressor()
        elif model_type == ModelTypes.SEQUENTIAL:
            self.model = Sequential()
            print(f'{type(self.model) = }')
            # Standardize the features
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
        else:
            raise ValueError(f"Model type not supported. Choose from {ModelTypes.__members__.keys()}")


    def _load_data(self):
        print(f'Current working directory: {os.getcwd()}')
        data = pd.read_csv(settings.data_params["data_path"])
        X = data.drop(settings.data_params["target_column"], axis=1)
        y = data[settings.data_params["target_column"]]

        if self.model_type == ModelTypes.LINEARREGRESSION:
            X['HasPhotovoltaics'] = X['HasPhotovoltaics'].astype(int)
            X['HasFireplace'] = X['HasPhotovoltaics'].astype(int)

            # One-Hot Encoding
            encoder = OneHotEncoder(sparse_output=False)
            encoded_colors = encoder.fit_transform(X[['HouseColor']])

            # Convert the result back to a DataFrame
            encoded_colors = pd.DataFrame(encoded_colors, columns=encoder.get_feature_names_out(['HouseColor']))
            X = pd.concat([X, encoded_colors], axis=1)

            X = X.drop(['HouseColor'], axis=1)
            numerical_columns = X.select_dtypes(include=np.number).columns
            X = X.fillna(X[numerical_columns].mean())

        elif self.model_type == ModelTypes.DECISIONTREEREGRESSOR or self.model_type == ModelTypes.MLPREGRESSOR:

            encoder = OneHotEncoder(sparse_output=False)
            encoded_colors = encoder.fit_transform(X[['HouseColor']])

            encoded_colors = pd.DataFrame(encoded_colors, columns=encoder.get_feature_names_out(['HouseColor']))
            X = pd.concat([X, encoded_colors], axis=1)

            X = X.drop(['HouseColor'], axis=1)
            numerical_columns = X.select_dtypes(include=np.number).columns
            X = X.fillna(X[numerical_columns].mean())

        elif self.model_type == ModelTypes.SEQUENTIAL:
            X_categorical = pd.get_dummies(X[settings.categorical_features], drop_first=True)
            X = pd.concat([X.drop(settings.categorical_features, axis=1), X_categorical], axis=1)

        return train_test_split(X, y, test_size=settings.test_size, random_state=settings.model_params["random_seed"])


    def train(self):
        if isinstance(self.model, CatBoostRegressor):
            '''# Do grid search to find best params
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=settings.test_size,
                                                                                  random_state=settings.model_params[
                                                                                      "random_seed"])

            model = CatBoostRegressor(**settings.model_params)

            grid = {
                'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1],
                'depth': [6, 8, 10],
                'iterations': [100, 500, 1000],
                'od_wait': [10, 50, 100],
            }

            # Record the start time
            start_time = time.time()

            grid_search_result = model.grid_search(
                grid,
                X_train_fold,
                y_train_fold,
                cv=10,
                plot=True,
                verbose=False
            )

            # Record the end time
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            # Convert seconds to hours, minutes, and seconds
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Format the result
            formatted_time = "{:02}:{:02}:{:.3f}".format(int(hours), int(minutes), seconds)

            print(f"Elapsed time for the Grid Search: {formatted_time}")
            print(f"Best params: {grid_search_result['params']}")'''


            # Train CatBoostRegressor without k-fold cross-validation
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(self.X_train, self.y_train, test_size=settings.test_size, random_state=settings.model_params["random_seed"])
            # Initialize and train CatBoostRegressor on the training part of the fold
            model = CatBoostRegressor(**settings.model_params)
            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=
                (
                    X_val_fold,
                    y_val_fold
                ),
                use_best_model=True,
                plot=False
            )

            predictions = model.predict(X_val_fold)
            val_mse = mean_squared_error(y_val_fold, predictions)
            val_mae = mean_absolute_error(y_val_fold, predictions)
            val_r2 = r2_score(y_val_fold, predictions)
            val_rmse = root_mean_squared_error(y_val_fold, predictions)

            # After finding the best model from CV, evaluate it on the test data
            test_predictions = model.predict(self.X_test)
            test_rmse = root_mean_squared_error(self.y_test, test_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)
            test_mae = mean_absolute_error(self.y_test, test_predictions)
            test_r2 = r2_score(self.y_test, test_predictions)

            print("Validation RMSE:", val_rmse)
            print("Validation MSE:", val_mse)
            print("Validation MAE:", val_mae)
            print("Validation R^2:", val_r2)

            print("Test RMSE:", test_rmse)
            print("Test MSE:", test_mse)
            print("Test MAE:", test_mae)
            print("Test R^2:", test_r2)

            metrics = {
                "Root Mean Squared Error": test_rmse,
                "Mean Squared Error": test_mse,
                "Mean Absolute Error": test_mae,
                "R^2 Score": test_r2
            }

            # Save metrics to a JSON file for later comparison
            with open(f"{self.logs_dir}/metrics.json", "w") as f:
                json.dump(metrics, f)

            # Plotting losses during training if needed
            plt.figure(figsize=(10, 5))
            plt.plot(model.evals_result_['learn']['RMSE'], label='Training RMSE')
            plt.plot(model.evals_result_['validation']['RMSE'], label='Validation RMSE')
            plt.title('RMSE Through Training')
            plt.xlabel('Iterations')
            plt.ylabel('RMSE')
            plt.legend()
            plt.savefig(f"{self.logs_dir}/metrics_during_training.png")
            plt.show()

            # Save the best model
            model.save_model(f"{self.logs_dir}/catboost_model.cbm")


            '''# Using k-fold cross-validation to find the best model
            # Initialize the best RMSE and best model
            results = []

            # Perform k-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=settings.model_params["random_seed"])

            # Iterate through each fold
            for train_index, val_index in kf.split(self.X_train):
                X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
                y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

                # Initialize and train CatBoostRegressor on the training part of the fold
                model = CatBoostRegressor(**settings.model_params)
                model.fit(
                    X_train_fold,
                    y_train_fold,
                    eval_set=
                    (
                        X_val_fold,
                        y_val_fold
                    ),
                    use_best_model=True,
                    plot=False
                )

                predictions = model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold, predictions)
                mae = mean_absolute_error(y_val_fold, predictions)
                r2 = r2_score(y_val_fold, predictions)
                rmse = root_mean_squared_error(y_val_fold, predictions)

                results.append({
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse
                })

            avg_rmse = np.mean([res['rmse'] for res in results])
            std_rmse = np.std([res['rmse'] for res in results])
            avg_r2 = np.mean([res['r2'] for res in results])

            # Select model with the best combination of low RMSE and high R²
            best_model = None
            best_rmse = float('inf')
            best_score = float('inf')

            for result in results:
                score = (result['rmse'] - avg_rmse) / std_rmse + (
                            avg_r2 - result['r2'])  # Penalize both high RMSE and low R²
                if score < best_score:
                    best_score = score
                    best_rmse = result['rmse']
                    best_model = result['model']

            # After finding the best model from CV, evaluate it on the test data
            test_predictions = best_model.predict(self.X_test)
            test_rmse = root_mean_squared_error(self.y_test, test_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)
            test_mae = mean_absolute_error(self.y_test, test_predictions)
            test_r2 = r2_score(self.y_test, test_predictions)

            print("Best CV RMSE:", best_rmse)
            print("Test RMSE:", test_rmse)
            print("Test MSE:", test_mse)
            print("Test MAE:", test_mae)
            print("Test R^2:", test_r2)

            metrics = {
                "Best CV RMSE - training phase [used just for comparison]": best_rmse,
                "Root Mean Squared Error": test_rmse,
                "Mean Squared Error": test_mse,
                "Mean Absolute Error": test_mae,
                "R^2 Score": test_r2
            }

            # Save metrics to a JSON file for later comparison
            with open(f"{self.logs_dir}/metrics.json", "w") as f:
                json.dump(metrics, f)

            # Plotting losses during training if needed
            plt.figure(figsize=(10, 5))
            plt.plot(best_model.evals_result_['learn']['RMSE'], label='Training RMSE')
            plt.plot(best_model.evals_result_['validation']['RMSE'], label='Validation RMSE')
            plt.title('RMSE Through Training')
            plt.xlabel('Iterations')
            plt.ylabel('RMSE')
            plt.legend()
            plt.savefig(f"{self.logs_dir}/metrics_during_training.png")
            plt.show()

            # Save the best model
            best_model.save_model(f"{self.logs_dir}/catboost_model.cbm")'''

        elif isinstance(self.model, DecisionTreeRegressor):

            model = DecisionTreeRegressor(random_state=42)
            model_params = {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                "splitter": ['best'],
                "max_depth": [5, 6, 7, 8, 9, 10],
                'max_features': ['sqrt', 'log2', None],
            }
            gs = GridSearchCV(model,model_params, cv=5, verbose=4)

            gs.fit(self.X_train, self.y_train)

            print(gs.best_params_)
            print(gs.best_score_)

            test_predictions = gs.best_estimator_.predict(self.X_test)

            test_rmse = root_mean_squared_error(self.y_test, test_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)
            test_mae = mean_absolute_error(self.y_test, test_predictions)
            test_r2 = r2_score(self.y_test, test_predictions)

            print("Test RMSE:", test_rmse)
            print("Test MSE:", test_mse)
            print("Test MAE:", test_mae)
            print("Test R^2:", test_r2)

            metrics = {
                "Root Mean Squared Error": test_rmse,
                "Mean Squared Error": test_mse,
                "Mean Absolute Error": test_mae,
                "R^2 Score": test_r2
            }

            # Save metrics to a JSON file for later comparison
            with open(f"{self.logs_dir}/metrics.json", "w") as f:
                json.dump(metrics, f)

            # Save the model
            joblib.dump(self.model, f"{self.logs_dir}/decision_tree_regression_model.pkl")

        elif isinstance(self.model, Sequential):
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(self.X_train, self.y_train, test_size=settings.test_size, random_state=settings.model_params["random_seed"])

            # Define the neural network architecture
            model = Sequential([
                Input(shape=(self.X_train.shape[1],)),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)  # Output layer with single neuron for regression
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            # Define callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

            # Train the model
            history = model.fit(
                X_train_fold,
                y_train_fold,
                epochs=settings.model_params["epochs"],
                batch_size=settings.model_params["batch_size"],
                validation_data=(X_val_fold, y_val_fold),
                verbose=0,
                callbacks=[early_stopping, reduce_lr]
            )

            # Compute metrics on validation data
            val_predictions = model.predict(X_val_fold)
            val_mse = mean_squared_error(y_val_fold, val_predictions)
            val_mae = mean_absolute_error(y_val_fold, val_predictions)
            val_r2 = r2_score(y_val_fold, val_predictions)
            val_rmse = root_mean_squared_error(y_val_fold, val_predictions)

            # Compute metrics on test data
            test_predictions = model.predict(self.X_test)
            test_rmse = root_mean_squared_error(self.y_test, test_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)
            test_mae = mean_absolute_error(self.y_test, test_predictions)
            test_r2 = r2_score(self.y_test, test_predictions)

            print("Validation RMSE:", val_rmse)
            print("Validation MSE:", val_mse)
            print("Validation MAE:", val_mae)
            print("Validation R^2:", val_r2)

            print("Test RMSE:", test_rmse)
            print("Test MSE:", test_mse)
            print("Test MAE:", test_mae)
            print("Test R^2:", test_r2)

            metrics = {
                "Root Mean Squared Error": test_rmse,
                "Mean Squared Error": test_mse,
                "Mean Absolute Error": test_mae,
                "R^2 Score": test_r2
            }

            # Save metrics to a JSON file for later comparison
            with open(f"{self.logs_dir}/metrics.json", "w") as f:
                json.dump(metrics, f)

            # Plotting training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"{self.logs_dir}/loss_during_training.png")
            plt.show()
            
        elif isinstance(self.model, MLPRegressor):

            model_params = {
                'solver': ['lbfgs', 'sgd', 'adam'],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "max_iter" : [600]
            }
            gs = GridSearchCV(self.model, model_params, cv=5, verbose=4)

            gs.fit(self.X_train, self.y_train)

            print(gs.best_params_)
            print(gs.best_score_)

            test_predictions = gs.best_estimator_.predict(self.X_test)

            test_rmse = root_mean_squared_error(self.y_test, test_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)
            test_mae = mean_absolute_error(self.y_test, test_predictions)
            test_r2 = r2_score(self.y_test, test_predictions)

            print("Test RMSE:", test_rmse)
            print("Test MSE:", test_mse)
            print("Test MAE:", test_mae)
            print("Test R^2:", test_r2)

            metrics = {
                "Root Mean Squared Error": test_rmse,
                "Mean Squared Error": test_mse,
                "Mean Absolute Error": test_mae,
                "R^2 Score": test_r2
            }

            # Save metrics to a JSON file for later comparison
            with open(f"{self.logs_dir}/metrics.json", "w") as f:
                json.dump(metrics, f)

        elif isinstance(self.model, LinearRegression):

            self.model.fit(self.X_train, self.y_train)
            test_predictions = self.model.predict(self.X_test)

            test_rmse = root_mean_squared_error(self.y_test, test_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)
            test_mae = mean_absolute_error(self.y_test, test_predictions)
            test_r2 = r2_score(self.y_test, test_predictions)

            print("Test RMSE:", test_rmse)
            print("Test MSE:", test_mse)
            print("Test MAE:", test_mae)
            print("Test R^2:", test_r2)

            metrics = {
                "Root Mean Squared Error": test_rmse,
                "Mean Squared Error": test_mse,
                "Mean Absolute Error": test_mae,
                "R^2 Score": test_r2
            }

            # Save metrics to a JSON file for later comparison
            with open(f"{self.logs_dir}/metrics.json", "w") as f:
                json.dump(metrics, f)

            # Save the model
            #joblib.dump(self.model, f"{self.logs_dir}/decision_tree_regression_model.pkl")

            # Plotting training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"{self.logs_dir}/loss_during_training.png")
            plt.show()
        else:
            raise ValueError(f"Model not supported for training. Choose from {ModelTypes.__members__.keys()}")
