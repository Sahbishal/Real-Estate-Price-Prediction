import os
from src.predict import RealEstatePredictor

def get_user_input():
    print("\n=== Enter House Details ===")
    
    try:
        area = float(input("Area (in sqft): "))
        bedrooms = int(input("Number of bedrooms: "))
        bathrooms = int(input("Number of bathrooms: "))
        stories = int(input("Number of stories: "))
        
        mainroad = input("Main road access (yes/no): ").lower()
        guestroom = input("Guestroom (yes/no): ").lower()
        basement = input("Basement (yes/no): ").lower()
        hotwaterheating = input("Hot water heating (yes/no): ").lower()
        airconditioning = input("Air conditioning (yes/no): ").lower()
        
        parking = int(input("Number of parking spots: "))
        prefarea = input("Preferred area (yes/no): ").lower()
        
        print("Furnishing status options: furnished, semi-furnished, unfurnished")
        furnishingstatus = input("Furnishing status: ").lower()
        
        return {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'parking': parking,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus
        }
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None

def main():
    model_path = os.path.join(os.getcwd(), 'models', 'model.joblib')
    preprocessor_path = os.path.join(os.getcwd(), 'models', 'preprocessor.joblib')
    
    try:
        predictor = RealEstatePredictor(model_path, preprocessor_path)
    except FileNotFoundError:
        print("Error: Model not found. Please run the training pipeline first.")
        return

    while True:
        user_data = get_user_input()
        
        if user_data:
            try:
                price = predictor.predict(user_data)
                print(f"\nEstimated Price: {price:,.2f}")
            except Exception as e:
                print(f"Error during prediction: {e}")
        
        cont = input("\nDo you want to predict another house? (yes/no): ").lower()
        if cont != 'yes':
            break

if __name__ == "__main__":
    main()
