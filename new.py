

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Regression Accuracy Function
def regression_accuracy(y_true, y_pred, tolerance=0.1):
    errors = np.abs((y_true - y_pred) / y_true)
    within_tolerance = errors < tolerance
    accuracy = np.mean(within_tolerance)
    return accuracy * 100

# Load the data
st.title('Gold Price Prediction')
data_path = st.text_input('Enter the path to the CSV file:', )

if data_path:
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Prepare the data
    X = data.drop(columns=['Date', 'GLD'], axis=1)
    y = data['GLD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=2)
    
    # Model training
    random_model = DecisionTreeRegressor()
    random_model.fit(X_train, y_train)
    
    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(random_model, file)
    st.write("Model trained and saved successfully!")

    # Prediction input form
    st.header("Predict Gold Price")
    input_values = {}
    for col in X.columns:
        input_values[col] = st.number_input(f"Input {col}", value=float(data[col].mean()))

    input_df = pd.DataFrame([input_values])
    st.write("Input DataFrame:", input_df)

    if st.button("Predict"):
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        prediction = model.predict(input_df)
        st.write(f"Predicted Gold Price (GLD): {prediction[0]:.2f}")
