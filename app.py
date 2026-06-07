import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('kmeans_model.pkl', 'rb'))

st.title("Customer Segmentation using K-Means")

income = st.number_input(
    "Annual Income (k$)",
    min_value=0
)

spending = st.number_input(
    "Spending Score (1-100)",
    min_value=0,
    max_value=100
)

if st.button("Predict Cluster"):

    cluster = model.predict(
        [[income, spending]]
    )[0]

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )