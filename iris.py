# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Train a Random Forest Classifier
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Streamlit App Title
st.title("Iris Flower Species Predictor")
st.write("""
This app predicts the species of an Iris flower based on its features.
""")

# Sidebar for User Input
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), 5.4)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), 1.3)
    petal_width = st.sidebar.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), 0.2)
    data = {
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display User Input
st.subheader("User Input Features")
st.write(input_df)

# Make Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Display Prediction
st.subheader("Prediction")
st.write(prediction[0])

# Display Prediction Probability
st.subheader("Prediction Probability")
st.write(prediction_proba)

# Optional: Display Dataset Overview
if st.checkbox("Show Dataset Overview"):
    st.write(df.head())

# Optional: Display Model Accuracy
if st.checkbox("Show Model Accuracy"):
    accuracy = clf.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2f}")
