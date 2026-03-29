import streamlit as st
import pandas as pd
from PIL import Image
import os

# Configure the Streamlit page
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📈 Retail Sales & Demand Forecasting")
st.markdown("This dashboard presents a 30-day forecast of retail sales volume based on 3 years of historical business data.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Executive Summary", "Forecast Output", "Feature Drivers", "Recommendations"])

if page == "Executive Summary":
    st.header("Historical Model Validation")
    st.markdown("Comparing historical actual sales versus what our machine learning model (Random Forest) accurately predicted.")
    if os.path.exists('historical_vs_predicted.png'):
        st.image('historical_vs_predicted.png', use_container_width=True)
    else:
        st.warning("Data not generated yet. Please run `forecast_model.py`.")

elif page == "Forecast Output":
    st.header("30-Day Predictive Forecast")
    st.markdown("Projecting exactly how many units are expected to sell daily over the next 30 days to optimize inventory.")
    
    if os.path.exists('future_forecast.png'):
        st.image('future_forecast.png', use_container_width=True)
        
    st.subheader("Data Forecast (Next 10 Days)")
    if os.path.exists('future_predictions.csv'):
        df = pd.read_csv('future_predictions.csv')
        st.dataframe(df[['Date', 'Predicted_Sales', 'Is_Weekend', 'Is_Holiday', 'Promotion']].head(10))

elif page == "Feature Drivers":
    st.header("What Drives Revenue?")
    st.markdown("A breakdown of exactly which features (promotions, seasonality, holidays) influence customer buying behavior the most.")
    if os.path.exists('feature_importance.png'):
        st.image('feature_importance.png', use_container_width=True)

elif page == "Recommendations":
    st.header("💼 Business Action Plan")
    st.markdown("""
    Based on the forecast output, we recommend:
    1. **📦 Inventory Optimization:** Order stock 2-3 weeks in advance of the projected mid-January peaks shown on the forecast page. 
    2. **🧑‍🤝‍🧑 Staffing Planning:** Schedule 20-30% more staff on predicted high-volume weekends. Reduce staffing dynamically on mid-week troughs.
    3. **💸 Cash Flow Forecasting:** Bank on the anticipated $X volume of sales projected over the next month for budgeting.
    """)
