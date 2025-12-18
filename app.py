import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Logistics Delay Predictor", layout="wide")

st.title("Logistics Delivery Delay Risk Predictor")

st.write(
    """
    This tool predicts whether an order is at risk of delivery delay
    using historical order, route, and cost data.
    """
)

# Load trained model
model = pickle.load(open("delay_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# Sidebar inputs
st.sidebar.header("Order Details")

priority = st.sidebar.selectbox("Delivery Priority", ["Express", "Standard", "Economy"])
product = st.sidebar.selectbox("Product Category", ["Electronics", "Fashion", "Industrial", "Food"])
order_value = st.sidebar.number_input("Order Value (INR)", 100.0, 50000.0)
distance = st.sidebar.number_input("Route Distance (KM)", 10.0, 5000.0)
traffic = st.sidebar.number_input("Traffic Delay (Minutes)", 0.0, 120.0)
fuel_cost = st.sidebar.number_input("Fuel Cost (INR)", 50.0, 5000.0)
labor_cost = st.sidebar.number_input("Labor Cost (INR)", 50.0, 5000.0)
delivery_cost = st.sidebar.number_input("Delivery Cost (INR)", 100.0, 10000.0)
rating = st.sidebar.slider("Customer Rating", 1, 5, 3)

# Prepare input
input_df = pd.DataFrame(
    [[priority, product, order_value, distance, traffic, fuel_cost, labor_cost, delivery_cost, rating]],
    columns=[
        "Priority",
        "Product_Category",
        "Order_Value_INR",
        "Distance_KM",
        "Traffic_Delay_Minutes",
        "Fuel_Cost",
        "Labor_Cost",
        "Delivery_Cost_INR",
        "Rating",
    ],
)

# Encode categoricals
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])


if st.button("Predict Delay Risk"):
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]


    st.subheader("Prediction Result")

    st.metric("Delay Risk Probability", f"{prob:.2f}")

    if prob > 0.7:
        st.error("🚨 High Risk of Delay")
        st.markdown("""
        **Recommended Actions:**
        - Reassign order to faster carrier
        - Prioritize warehouse processing
        - Optimize route to avoid traffic
        - Proactively notify customer
        """)
    elif prob > 0.4:
        st.warning("⚠️ Medium Risk of Delay")
        st.markdown("""
        **Recommended Actions:**
        - Monitor shipment closely
        - Alert operations team
        - Prepare alternate route or vehicle
        """)
    else:
        st.success("✅ Low Risk of Delay")
        st.markdown("""
        **Recommended Actions:**
        - Continue with standard delivery process
        """)
