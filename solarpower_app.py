import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App Title
st.title("Solar Power Prediction Using Linear Regression")

# Load CSV file dynamically
csv_file = "solarpowergeneration.csv"

if not os.path.exists(csv_file):
    st.error(f"Error: The file `{csv_file}` was not found! Please ensure it is in the same directory.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()  # Clean column names
    return df

df = load_data()

# Display dataset preview
st.write("### Dataset Preview")
st.write(df.head())

# Show column names
st.write("###Columns:")
st.write(df.columns.tolist())

# Define expected column keywords
feature_keywords = ["irradiance", "temperature", "humidity", "wind", "speed"]
target_keywords = ["power", "output", "generation"]

# Automatically select feature columns
features = [col for col in df.columns if any(keyword in col for keyword in feature_keywords)]
target = next((col for col in df.columns if any(keyword in col for keyword in target_keywords)), None)

# If no valid target column found, pick the last column as default
if target is None:
    target = df.columns[-1]

# Prepare Data
X = df[features]
y = df[target]

# Handle missing values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"### Model Performance:\n- **MSE:** {mse:.2f}\n- **R² Score:** {r2:.2f}")

# User Input for Prediction
st.write("##Predicting")

user_inputs = []
for feature in features:
    val = st.number_input(f"Enter {feature.capitalize()}:", value=float(X[feature].mean()))
    user_inputs.append(val)

# Predict Power Output
if st.button("Predict Solar Power Output"):
    user_input_array = np.array([user_inputs])
    prediction = model.predict(user_input_array)
    st.write(f"###  Predicted Power Output: {prediction[0]:.2f} kW")

# Visualization
st.write("##  Actual vs. Predicted Power Output")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, color="blue", label="Predictions")
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label="Perfect Fit")
ax.set_xlabel("Actual Power Output")
ax.set_ylabel("Predicted Power Output")
ax.set_title("Linear Regression Predictions")
ax.legend()
st.pyplot(fig)
