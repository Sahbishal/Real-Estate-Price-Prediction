import joblib
import pandas as pd
import os

class RealEstatePredictor:
    def __init__(self, model_path='models/model.joblib', preprocessor_path='models/preprocessor.joblib'):
        """
        Initializes the predictor by loading the model and preprocessor.
        """
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor not found. Please train the model first.")
            
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
    def predict(self, data: dict):
        """
        Predicts the price for a single house.
        
        Args:
            data (dict): Dictionary containing house features.
            
        Returns:
            float: Predicted price.
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        try:
            processed_data = self.preprocessor.transform(df)
        except Exception as e:
            raise ValueError(f"Error in preprocessing: {e}")
            
        # Predict
        prediction = self.model.predict(processed_data)
        return prediction[0]

if __name__ == "__main__":
    # Example usage
    predictor = RealEstatePredictor(
        model_path=os.path.join(os.getcwd(), 'models', 'model.joblib'),
        preprocessor_path=os.path.join(os.getcwd(), 'models', 'preprocessor.joblib')
    )
    
    # Sample data (taken from dataset head)
    sample_house = {
        'area': 7420,
        'bedrooms': 4,
        'bathrooms': 2,
        'stories': 3,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'furnished'
    }
    
    price = predictor.predict(sample_house)
    print(f"Predicted Price: {price}")
