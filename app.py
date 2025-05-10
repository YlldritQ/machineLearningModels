import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction Model",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("Diabetes Prediction Model")
st.markdown("""
This application predicts the likelihood of diabetes based on various health metrics.
The model is trained on the Pima Indians Diabetes Database.
""")

# Load and preprocess data
@st.cache_data
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
        st.error(f"Error loading data: {str(e)}")
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
            # Updated way to fill NaN values without using inplace
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

# Load and prepare data
try:
    data = load_data()
    if data is not None:
        cleaned_data = clean_data(data)
        
        # Sidebar for data exploration
        st.sidebar.header("Data Exploration")
        if st.sidebar.checkbox("Show Raw Data"):
            st.subheader("Raw Data")
            st.write(data)
        
        if st.sidebar.checkbox("Show Data Statistics"):
            st.subheader("Data Statistics")
            st.write(data.describe())
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "Data Visualization"])
        
        with tab1:
            st.header("Make a Prediction")
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
                glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
                blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
                skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
            
            with col2:
                insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=79)
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
                age = st.number_input("Age", min_value=0, max_value=120, value=33)
            
            if st.button("Predict"):
                # Prepare the model
                X = cleaned_data.drop('outcome', axis=1)
                y = cleaned_data['outcome']
                
                # Engineer features
                X = engineer_features(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
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
                model.fit(X_train_scaled, y_train)
                
                # Display feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.subheader("Feature Importance")
                st.write(feature_importance)
                
                # Make prediction
                input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                        insulin, bmi, diabetes_pedigree, age]], 
                                        columns=['pregnancies', 'glucose', 'blood_pressure', 
                                                'skin_thickness', 'insulin', 'bmi', 
                                                'diabetes_pedigree', 'age'])
                
                input_engineered = engineer_features(input_data)
                input_scaled = scaler.transform(input_engineered)
                prediction = model.predict(input_scaled)
                probability = model.predict_proba(input_scaled)
                
                # Display prediction
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.error(f"High risk of diabetes (Probability: {probability[0][1]:.2%})")
                else:
                    st.success(f"Low risk of diabetes (Probability: {probability[0][0]:.2%})")
        
        with tab2:
            st.header("Model Performance")
            if st.button("Train and Evaluate Model"):
                # Prepare data
                X = cleaned_data.drop('outcome', axis=1)
                y = cleaned_data['outcome']
                
                # Engineer features
                X = engineer_features(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
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
                model.fit(X_train_scaled, y_train)
                
                # Display feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.subheader("Feature Importance")
                st.write(feature_importance)
                
                # Calculate and display accuracy
                train_accuracy = model.score(X_train_scaled, y_train)
                test_accuracy = model.score(X_test_scaled, y_test)
                st.info(f"Model Accuracy - Training: {train_accuracy:.2%}, Testing: {test_accuracy:.2%}")
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                
                # Display metrics
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report)
                
                # Plot confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
        
        with tab3:
            st.header("Data Visualization")
            
            # Feature correlation
            st.subheader("Feature Correlation")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cleaned_data.corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig)
            
            # Distribution plots
            st.subheader("Feature Distributions")
            feature = st.selectbox("Select Feature", cleaned_data.columns[:-1])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=cleaned_data, x=feature, hue='outcome', multiple="stack")
            plt.title(f'Distribution of {feature} by Diabetes Outcome')
            st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please make sure the data file 'pima-indians-diabetes.csv' is in the correct location.") 