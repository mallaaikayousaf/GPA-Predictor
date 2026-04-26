import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import shap

st.set_page_config(page_title="GPA Predictor", layout="wide")
st.title("🎓 GPA Predictor")
st.markdown("Predicts your next semester GPA using Machine Learning")

@st.cache_data
def load_data():
    df = pd.read_csv('grades.csv')
    df['prev_gpa'] = df['gpa'].shift(1)
    df['workload_credits'] = df['workload_hours'] / df['credits']
    return df

df = load_data()
clean_df = df.dropna(subset=['gpa', 'prev_gpa'])

features = ['prev_gpa', 'credits', 'is_math', 'workload_credits']
X = clean_df[features]
y = clean_df['gpa']

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
lr = LinearRegression()
lr.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(rf)

# UI Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Your Data")
    st.dataframe(df)
    
    st.subheader("🎛️ Next Semester Inputs")
    prev_gpa = st.number_input("Previous GPA", 0.0, 4.0, 3.5, step=0.1)
    credits = st.slider("Credits", 6, 21, 16)
    is_math = st.checkbox("Taking a math course?")
    workload = st.slider("Weekly study hours", 5, 40, 15)
    workload_credits = workload / credits

with col2:
    st.subheader("📈 Predictions")
    
    input_data = np.array([[prev_gpa, credits, int(is_math), workload_credits]])
    lr_pred = lr.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]
    
    st.metric("Linear Regression", f"{lr_pred:.2f}")
    st.metric("Random Forest", f"{rf_pred:.2f}", delta=f"{rf_pred - lr_pred:+.2f}")
    
    st.subheader("🎯 Recommended")
    st.metric("Your Predicted GPA", f"{rf_pred:.2f}", delta=None)
    
    st.subheader("🔮 What-If")
    adjustment = st.slider("Add to all course grades", -0.5, 0.5, 0.0, 0.05)
    st.metric("Adjusted GPA", f"{rf_pred + adjustment:.2f}")

# Feature Importance
st.subheader("📊 What Matters Most?")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots()
ax.barh(importance_df['Feature'], importance_df['Importance'])
ax.set_xlabel('Importance')
st.pyplot(fig)

# ========== FIXED SHAP SECTION ==========
st.subheader("🔍 Why This Prediction?")

# Calculate SHAP values
shap_values = explainer.shap_values(input_data)

# Create waterfall plot
fig2, ax2 = plt.subplots(figsize=(10, 5))

# FIX: Use plt.figure() and then convert to st.pyplot
waterfall_plot = shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value if not hasattr(explainer.expected_value, 'item') else explainer.expected_value.item(),
        data=input_data[0],
        feature_names=features
    ),
    show=False
)

# IMPORTANT FIX: Get the current figure
fig2 = plt.gcf()
st.pyplot(fig2)

st.caption("🔴 Red = pushes GPA higher | 🔵 Blue = pushes GPA lower")

st.caption("Built with Streamlit + Random Forest + SHAP")