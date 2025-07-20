# dploy.py

import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict whether they would survive.")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode categorical variables
sex_encoded = 1 if sex == 'male' else 0

# One-hot encode 'Embarked'
embarked_C = 1 if embarked == 'C' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Prepare the feature array in the same order as training
features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_C, embarked_Q, embarked_S]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("🎉 The passenger **would survive**.")
    else:
        st.error("☠️ The passenger **would not survive**.")
