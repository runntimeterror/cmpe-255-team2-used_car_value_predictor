import streamlit as st
import pandas as pd

@st.cache
def load_data():
    df = pd.read_csv("./data/vehiclesFinal.csv") 
    return df