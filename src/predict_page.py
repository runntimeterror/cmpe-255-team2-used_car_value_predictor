import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    # with open('./carprice-prediction/saved_model.pkl', 'rb') as file:
    with open('./carprice-prediction2/saved_model_decision_tree.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()


def show_predict_page():
    st.title("Used car price predictor")

    st.write("""Please provide the required information to predict the current price of your car""")

    year = (
        "2022",
        "2021",
        "2020",
        "2019",
        "2018",
        "2017",
        "2016",
        "2015",
        "2014",
        "2013",
        "2012",
        "2011",
        "2010",
        "2009",
        "2008",
        "2007",
        "2006",
        "2005",
        "2004",
        "2003",
        "2002",
        "2001"
        "2000"
    )

    condition = ("excellent", "fair", "good", "like new","new", "salvage")

    with st.form(key='columns_in_form'):
            c1, c2,  = st.columns(2)
            with c1:
                Year_Selected = c1.selectbox("Year", year)
            
                Odometer = c1.slider("Odometer", 0, 500000)
            
                Condition = c1.selectbox("Condition", condition)
                Condition_Excellent =0 
                Condition_Fair =0
                Condition_Good =0
                Condition_Like_New =0
                Condition_New =0
                Condition_Salvage =0

                if(Condition == 'excellent'):
                    Condition_Excellent=1
                elif(Condition == 'fair'):
                    Condition_Fair=1
                elif(Condition == 'good'):
                    Condition_Good=1
                elif(Condition=='like new'):
                    Condition_Like_New=0
                elif(Condition=='new'):
                    Condition_New=0
                elif(Condition == 'salvage'):
                    Condition_Salvage=1

                
            with c2:
                Fuel_Type = c2.radio("Fuel Type", ("Petrol", "Diesel"))
                Fuel_Type_Petrol = 0
                Fuel_Type_Diesel = 0
                # Transmission = c2.radio("Transmission", ("Mannual", "Automatic"))
                # Transmission_Mannual = 0 
                # Seller_Type = c2.radio("Seller Type", ("Individual", "Dealer"))
                # Seller_Type_Individual = 0

                
                # if(Fuel_Type == 'Petrol'):
                #     Fuel_Type_Petrol=1
                #     Fuel_Type_Diesel=0
                # elif(Fuel_Type=='Diesel'):
                #     Fuel_Type_Petrol=0
                #     Fuel_Type_Diesel=1

                # if(Transmission=='Mannual'):
                #     Transmission_Mannual=1
                # else:
                #     Transmission_Mannual=0

                
                # if(Seller_Type=='Individual'):
                #     Seller_Type_Individual=1
                # else:
                #     Seller_Type_Individual=0


            ok = st.form_submit_button("Calculate Car Value")
            if ok:
                print("i am egre")
                prediction=model.predict([[int(Year_Selected), Odometer,Condition_Excellent,Condition_Fair,Condition_Good,
                Condition_Like_New,
                Condition_New,
                Condition_Salvage]])
                output=round(prediction[0], 2)
                
                st.subheader(f"The estimated Price is ${output}")
          
            ok2 = st.form_submit_button("Price range by year")
            data = {"year", "price"}
            df_range = pd.DataFrame(columns=['year','price'])
            if ok2:
                for x in range(-5,5):
                    pred_year = int(Year_Selected)+x
                    prediction=model.predict([[pred_year, Odometer,Condition_Excellent,Condition_Fair,Condition_Good,
                    Condition_Like_New,
                    Condition_New,
                    Condition_Salvage]])
                    output=round(prediction[0], 2)
                    df_range = df_range.append({'year': pred_year, "price": output}, ignore_index=True)
                print(df_range)
                df_range = df_range.astype('int')
                chart_data = pd.DataFrame(df_range)

                st.line_chart(chart_data,x="year", y="price")