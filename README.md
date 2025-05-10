# Pima Indians Diabetes Prediction Model

This project implements a neural network model to predict diabetes in Pima Indian women based on various health metrics.

## Dataset Description

The dataset contains medical predictor variables and one target variable (outcome) for Pima Indian women. The predictor variables include:

1. Number of pregnancies
2. Plasma glucose concentration
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)²)
7. Diabetes pedigree function
8. Age (years)

The target variable is whether the patient has diabetes (1) or not (0).

## Project Structure

- `diabetes_prediction.py`: Main script containing the model implementation
- `requirements.txt`: List of Python dependencies
- `confusion_matrix.png`: Generated confusion matrix visualization
- `training_history.png`: Generated training history visualization

## Setup and Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset file (`pima-indians-diabetes.csv`) in the project root directory

4. Run the script:
   ```bash
   python diabetes_prediction.py
   ```

## Model Architecture

The neural network consists of:
- Input layer: 8 features
- Hidden layers:
  - Dense layer (32 neurons) with ReLU activation
  - Dropout layer (20%)
  - Dense layer (16 neurons) with ReLU activation
  - Dropout layer (20%)
  - Dense layer (8 neurons) with ReLU activation
- Output layer: 1 neuron with sigmoid activation

## Data Preprocessing

The script performs the following preprocessing steps:
1. Handles missing values (zeros) in relevant columns
2. Scales features using StandardScaler
3. Splits data into 70% training and 30% testing sets
4. Further splits training data for validation during training

## Training

The model is trained with:
- 100 maximum epochs
- Batch size of 32
- Early stopping to prevent overfitting
- Adam optimizer
- Binary cross-entropy loss function

## Output

The script generates:
1. Data analysis summary
2. Model training progress
3. Classification report with accuracy, precision, recall, and F1-score
4. Confusion matrix visualization
5. Training history plot

## Notes

- The model includes dropout layers to prevent overfitting
- Early stopping is implemented to optimize training duration
- The dataset is imbalanced (more negative cases than positive) 