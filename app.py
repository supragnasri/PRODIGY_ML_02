import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('kmeans_model.pkl', 'rb'))

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# Title
st.title("🛍️ Customer Segmentation using K-Means Clustering")

st.write("""
This application segments customers based on their:

- Annual Income (k$)
- Spending Score (1-100)

Enter customer details below to identify the customer segment.
""")

# Inputs
income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=50
)

spending = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

# Predict Button
if st.button("Identify Customer Segment"):

    cluster = model.predict([[income, spending]])[0]

    # NOTE:
    # These names may need adjustment depending on your cluster centers.
    cluster_names = {
        0: "Average Customers",
        1: "Wealthy but Careful Spenders",
        2: "Budget-Conscious Customers",
        3: "Premium Customers",
        4: "High-Spending Customers"
    }

    cluster_info = {
        0: "Customers with average income and average spending habits.",
        1: "Customers with high income but relatively low spending behavior.",
        2: "Customers with lower income and lower spending patterns.",
        3: "High-value customers with both high income and high spending.",
        4: "Customers who spend actively despite comparatively lower income."
    }

    st.success(
        f"Predicted Segment: {cluster_names.get(cluster, f'Cluster {cluster}')}"
    )

    st.info(
        cluster_info.get(cluster, "Customer segment identified successfully.")
    )

    st.subheader("Business Insight")

    if cluster == 3:
        st.write(
            "⭐ These customers are the most valuable segment and are ideal targets for premium offers and loyalty programs."
        )

    elif cluster == 1:
        st.write(
            "💰 These customers have strong purchasing power but spend cautiously. Personalized promotions may increase engagement."
        )

    elif cluster == 4:
        st.write(
            "🛒 These customers demonstrate strong spending behavior and respond well to targeted marketing campaigns."
        )

    elif cluster == 2:
        st.write(
            "📉 These customers tend to spend less and may respond better to discounts and value-based offers."
        )

    else:
        st.write(
            "📊 These customers represent a balanced segment with moderate spending behavior."
        )

# Footer
st.markdown("---")
st.caption("Developed as part of Prodigy InfoTech Machine Learning Internship - Task 02")
