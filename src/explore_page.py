import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from data_cache import load_data
import seaborn as sns
import numpy as np

def show_explore_page():
    st.title("Explore Car prices against other attributes")
    df_final = load_data()    
    # plt.figure(figsize=(5,5))
    # sns.histplot(df_final.price, color="blue",  kde=True,  linewidth=0)
    
    st.write("""#### Car count vs Price """)
    fig1 = plt.figure(figsize=(7, 3))
    sns.histplot(df_final.price, color="blue",  kde=True,  linewidth=0)
    st.pyplot(fig1)


    st.write("""#### Fuel type vs Price """)
    fig2 = plt.figure(figsize=(7, 3))
    sns.barplot(x='fuel',y='price',data=df_final)
    st.pyplot(fig2)


    st.write("""#### Vehicle condition VS Price """)
    fig3 = plt.figure(figsize=(7, 3))
    sns.barplot(x='fuel',y='price',hue='condition',data=df_final)
    st.pyplot(fig3)


    st.write("""#### Vehicle Type vs Price """)
    fig4 = plt.figure(figsize=(7, 3))
    sns.barplot(x='type',y='price',data=df_final)
    st.pyplot(fig4)

    st.write("""#### Manufacturer VS Price """)
    fig5 = plt.figure(figsize=(10, 3))
    sns.barplot(x='manufacturer',y='price',data=df_final,dodge=False)
    plt.xticks(rotation=30)
    st.pyplot(fig5)
