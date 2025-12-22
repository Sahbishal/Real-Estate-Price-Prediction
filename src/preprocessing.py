import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df: pd.DataFrame, target_column: str = 'price'):
    """
    Preprocesses the data: separates target, encodes categorical variables, and scales numerical ones.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        target_column (str): Name of the target column.
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numerical columns
    # We know specific columns from analysis, but let's make it dynamic or explicit based on the dataset knowledge
    # Explicitly defining based on dataset inspection:
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    categorical_cols = ['furnishingstatus']
    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    # Create transformers
    # For binary yes/no, we can map them or use OneHotEncoder. OneHotEncoder is safer for general pipelines.
    # However, yes/no is better as binary 0/1. Let's map them first if possible, or just use OHE.
    # Let's use OHE for all categorical/object columns for simplicity and robustness in the pipeline.
    
    # Actually, let's check dtypes.
    # numerical_cols are int/float.
    # binary and categorical are object.
    
    numerical_transformer = StandardScaler()
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first') # drop='first' to avoid dummy variable trap
    
    # Combine in ColumnTransformer
    # Note: We need to pass the correct column names.
    
    # Let's verify column names exist in X
    # All columns in X that are object type will be treated as categorical
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
