# Real Estate Price Prediction

This project implements a machine learning pipeline to predict real estate prices based on various features like area, number of bedrooms, bathrooms, etc.

## Project Structure

```
Real-Estate-Price-Prediction/
├── data/                   # Dataset directory (Housing.csv should be here or in root)
├── models/                 # Saved models and preprocessors
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_loader.py      # Data loading logic
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── train.py            # Model training and evaluation
│   └── predict.py          # Inference logic
├── main.py                 # Main script to run the training pipeline
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Dataset

The project uses `Housing.csv` which contains the following features:
- price (Target)
- area
- bedrooms
- bathrooms
- stories
- mainroad
- guestroom
- basement
- hotwaterheating
- airconditioning
- parking
- prefarea
- furnishingstatus

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, evaluate it, and save the artifacts:

```bash
python main.py
```

This will create `models/model.joblib` and `models/preprocessor.joblib`.

### Making Predictions

You can use the `src/predict.py` script to make predictions programmatically.

Example:

```python
from src.predict import RealEstatePredictor
import os

predictor = RealEstatePredictor(
    model_path='models/model.joblib',
    preprocessor_path='models/preprocessor.joblib'
)
data = {
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

price = predictor.predict(data)
print(f"Predicted Price: {price}")
```


