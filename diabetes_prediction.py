import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define column names based on the dataset description
columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
           'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']

# Load the data (assuming it's in CSV format, you'll need to adjust the path)
data = pd.read_csv('pima-indians-diabetes.csv', names=columns)

# Data Analysis and Cleaning
def analyze_data(df):
    print("\nData Analysis:")
    print("-" * 50)
    print("\nShape:", df.shape)
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nClass Distribution:")
    print(df['outcome'].value_counts(normalize=True))

def clean_data(df):
    # Replace 0 values with NaN for columns where 0 is not possible
    zero_not_possible = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
    df[zero_not_possible] = df[zero_not_possible].replace(0, np.nan)
    
    # Fill NaN values with median
    for column in df.columns[:-1]:  # Exclude the outcome column
        df[column] = df[column].fillna(df[column].median())
    
    return df

# Data Preprocessing
def preprocess_data(df):
    # Separate features and target
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Create and train the model
def create_and_train_model(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16, 8),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

# Evaluate and plot results
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Training Loss')
    plt.title('Model Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and analyze data
    print("Loading and analyzing data...")
    analyze_data(data)
    
    # Clean data
    print("\nCleaning data...")
    cleaned_data = clean_data(data)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(cleaned_data)
    
    # Create and train model
    print("\nCreating and training model...")
    model = create_and_train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main() 