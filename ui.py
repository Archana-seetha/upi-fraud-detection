import streamlit as st
import numpy as np
import pickle

# Load saved files
models = pickle.load(open("models.pkl", "rb"))
best_model_name = pickle.load(open("best_model_name.pkl", "rb"))
results = pickle.load(open("results.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("UPI Fraud Detection System")

state_city_mapping = {
    0: "Delhi",
    1: "Mumbai",
    2: "Bangalore",
    3: "Hyderabad",
    4: "Chennai",
    5: "Kolkata",
    6: "Pune",
    7: "Ahmedabad",
    8: "Jaipur",
    9: "Lucknow",
    10: "Kanpur",
    11: "Nagpur",
    12: "Indore",
    13: "Bhopal",
    14: "Patna",
    15: "Chandigarh",
    16: "Surat",
    17: "Vadodara",
    18: "Coimbatore",
    19: "Visakhapatnam",
    20: "Thiruvananthapuram",
    21: "Kochi",
    22: "Guwahati",
    23: "Ranchi",
    24: "Raipur",
    25: "Dehradun",
    26: "Shimla",
    27: "Jammu",
    28: "Srinagar",
    29: "Amritsar",
    30: "Jalandhar",
    31: "Agra",
    32: "Varanasi",
    33: "Meerut",
    34: "Noida",
    35: "Gurgaon",
    36: "Faridabad",
    37: "Mysore",
    38: "Hubli",
    39: "Madurai",
    40: "Salem",
    41: "Tiruchirappalli",
    42: "Warangal",
    43: "Vijayawada",
    44: "Guntur",
    45: "Nashik",
    46: "Aurangabad",
    47: "Solapur",
    48: "Dhanbad",
    49: "Jamshedpur",
    50: "Others"
}
category_mapping = {
    0: "Food & Dining",
    1: "Shopping",
    2: "Bill Payment",
    3: "Money Transfer",
    4: "Mobile Recharge",
    5: "Travel & Transport",
    6: "Entertainment",
    7: "Healthcare",
    8: "Education",
    9: "Groceries",
    10: "Utilities",
    11: "Subscription Services",
    12: "Insurance",
    13: "Others"
}

# Inputs
trans_hour = st.number_input("Transaction Hour (0-23)")
trans_day = st.number_input("Day")
trans_month = st.number_input("Month")
trans_year = st.number_input("Year")

category_reverse = {v: k for k, v in category_mapping.items()}
category_name = st.selectbox("Transaction Category", list(category_reverse.keys()))
category = category_reverse[category_name]

age = st.number_input("Age")
trans_amount = st.number_input("Transaction Amount")
city_name = st.selectbox("City", list(state_city_mapping.values()))
state = list(state_city_mapping.keys())[list(state_city_mapping.values()).index(city_name)]
zip_code = st.number_input("ZIP Code")

# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ["Best Model"] + list(models.keys())
)

if model_choice == "Best Model":
    selected_model = models[best_model_name]
    selected_name = best_model_name
else:
    selected_model = models[model_choice]
    selected_name = model_choice

# Show metrics
st.subheader("Model Performance")
st.write("Accuracy:", results[selected_name]["accuracy"])
st.write("Precision:", results[selected_name]["precision"])
st.write("Recall:", results[selected_name]["recall"])
st.write("F1 Score:", results[selected_name]["f1"])

# Prediction
  if st.button("Predict"):
    input_data = np.array([[trans_hour, trans_day, trans_month, trans_year,
                            category, age, trans_amount, state, zip_code]])

    input_scaled = scaler.transform(input_data)

    pred = selected_model.predict(input_scaled)[0]
    pred_prob = selected_model.predict_proba(input_scaled)

    # 🔥 Identify fraud class index dynamically
    fraud_class_index = list(selected_model.classes_).index(1) if 1 in selected_model.classes_ else 0

    risk = pred_prob[0][fraud_class_index] * 100

    st.write(f"Fraud Risk: {risk:.2f}%")

    if risk > 75:
        st.error("High Risk")
    elif risk > 40:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")
