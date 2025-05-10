import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

def load_data():
    try:
        # Read the file line by line
        with open('pima-indians-diabetes.csv', 'r') as file:
            lines = file.readlines()
        
        # Process the data
        data_list = []
        for line in lines:
            # Split the line by comma and convert to numeric values
            values = [x.strip() for x in line.strip().split(',')]
            try:
                # Convert all values to float
                numeric_values = [float(x) if x else np.nan for x in values]
                data_list.append(numeric_values)
            except ValueError:
                continue  # Skip lines that can't be converted to numeric
        
        # Create DataFrame
        columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                  'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
        df = pd.DataFrame(data_list, columns=columns)
        
        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN in the outcome column
        df = df.dropna(subset=['outcome'])
        
        # Convert outcome to int
        df['outcome'] = df['outcome'].astype(int)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_data(df):
    if df is None:
        return None
    
    df_cleaned = df.copy()
    
    # Replace 0 with NaN for columns where 0 is not possible
    zero_not_possible = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
    for col in zero_not_possible:
        df_cleaned.loc[df_cleaned[col] == 0, col] = np.nan
    
    # Fill NaN values with median for each column
    for column in df_cleaned.columns:
        if column != 'outcome':
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
    
    return df_cleaned

def engineer_features(df):
    """Add engineered features to improve model performance"""
    df_engineered = df.copy()
    
    # BMI categories
    df_engineered['bmi_category'] = pd.cut(df_engineered['bmi'], 
        bins=[0, 18.5, 24.9, 29.9, float('inf')],
        labels=[0, 1, 2, 3])
    
    # Age categories
    df_engineered['age_category'] = pd.cut(df_engineered['age'],
        bins=[0, 20, 40, 60, float('inf')],
        labels=[0, 1, 2, 3])
    
    # Glucose categories
    df_engineered['glucose_category'] = pd.cut(df_engineered['glucose'],
        bins=[0, 70, 99, 126, float('inf')],
        labels=[0, 1, 2, 3])
    
    # Interaction features
    df_engineered['glucose_bmi'] = df_engineered['glucose'] * df_engineered['bmi']
    df_engineered['age_bmi'] = df_engineered['age'] * df_engineered['bmi']
    
    # Risk score (combination of important features)
    df_engineered['risk_score'] = (
        df_engineered['glucose'] * 0.3 + 
        df_engineered['bmi'] * 0.2 + 
        df_engineered['age'] * 0.1 + 
        df_engineered['diabetes_pedigree'] * 0.4
    )
    
    return df_engineered

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy - Training: {train_accuracy:.2%}, Testing: {test_accuracy:.2%}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def plot_feature_importance(model, feature_names):
    # Display feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_feature_distributions(data):
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(data.columns[:-1], 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=data, x=feature, hue='outcome', multiple="stack")
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

def make_prediction(model, scaler, input_data):
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0]

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    data = load_data()
    if data is None:
        print("Error: Could not load data. Please check if the data file exists.")
        return
    
    cleaned_data = clean_data(data)
    
    # Prepare features and target
    X = cleaned_data.drop('outcome', axis=1)
    y = cleaned_data['outcome']
    
    # Engineer features
    print("\nEngineering features...")
    X = engineer_features(X)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate model
    print("\nTraining and evaluating model...")
    model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Plot visualizations
    print("\nGenerating visualizations...")
    plot_feature_importance(model, X.columns)
    plot_confusion_matrix(y_test, model.predict(X_test_scaled))
    plot_feature_distributions(cleaned_data)
    
    # Example prediction
    print("\nExample prediction:")
    example_input = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], 
                               columns=['pregnancies', 'glucose', 'blood_pressure', 
                                      'skin_thickness', 'insulin', 'bmi', 
                                      'diabetes_pedigree', 'age'])
    example_input_engineered = engineer_features(example_input)
    prediction, probability = make_prediction(model, scaler, example_input_engineered)
    
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
    print(f"Probability: {probability[1]:.2%} chance of diabetes")

if __name__ == "__main__":
    main() 