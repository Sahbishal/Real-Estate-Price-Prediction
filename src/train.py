import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_model(X_train, y_train):
    """
    Trains a Random Forest Regressor model.
    
    Args:
        X_train: Processed training features.
        y_train: Training target.
        
    Returns:
        model: Trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints metrics.
    
    Args:
        model: Trained model.
        X_test: Processed test features.
        y_test: Test target.
        
    Returns:
        metrics (dict): Dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation Metrics:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    
    return {"mae": mae, "rmse": rmse, "r2": r2}

def save_artifacts(model, preprocessor, model_path='models/model.joblib', preprocessor_path='models/preprocessor.joblib'):
    """
    Saves the trained model and preprocessor.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")
