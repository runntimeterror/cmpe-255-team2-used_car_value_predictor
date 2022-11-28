import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from depreciation_page import show_depreciation_page
# st.set_page_config(layout="wide")
# page = st.sidebar.selectbox("Current Price  Or Depreciation", ("Predict", "Depreciation"))
tab1, tab2 = st.tabs(["Price Prediction", "Explore and Analyze"])

# page = st.tabs(tabs)
# if page == "Predict":
with tab1:
    show_predict_page()
with tab2:
    show_explore_page()