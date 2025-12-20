import os
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model, evaluate_model, save_artifacts

def main():
    # Define paths
    DATA_PATH = os.path.join(os.getcwd(), 'Housing.csv')
    MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model.joblib')
    PREPROCESSOR_PATH = os.path.join(os.getcwd(), 'models', 'preprocessor.joblib')
    
    print("Starting Real Estate Price Prediction Pipeline...")
    
    # 1. Load Data
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(e)
        return

    # 2. Preprocess Data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # 3. Train Model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # 4. Evaluate Model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # 5. Save Artifacts
    print("Saving artifacts...")
    save_artifacts(model, preprocessor, MODEL_PATH, PREPROCESSOR_PATH)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
