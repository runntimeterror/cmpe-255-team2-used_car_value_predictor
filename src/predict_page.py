import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb


def load_model():
    # with open('./carprice-prediction/saved_model.pkl', 'rb') as file:
    bst = xgb.Booster()  # init model
    bst.load_model('./carprice-prediction2/xbg_model.bin')  # load data
    return bst


model = load_model()

one_hot_columns = [
    # 'year',
    # 'odometer',
    'condition_excellent',
    'condition_fair',
    'condition_good',
    'condition_like new',
    'condition_new',
    'condition_salvage',
    'state_ak',
    'state_al',
    'state_ar',
    'state_az',
    'state_ca',
    'state_co',
    'state_ct',
    'state_dc',
    'state_de',
    'state_fl',
    'state_ga',
    'state_hi',
    'state_ia',
    'state_id',
    'state_il',
    'state_in',
    'state_ks',
    'state_ky',
    'state_la',
    'state_ma',
    'state_md',
    'state_me',
    'state_mi',
    'state_mn',
    'state_mo',
    'state_ms',
    'state_mt',
    'state_nc',
    'state_nd',
    'state_ne',
    'state_nh',
    'state_nj',
    'state_nm',
    'state_nv',
    'state_ny',
    'state_oh',
    'state_ok',
    'state_or',
    'state_pa',
    'state_ri',
    'state_sc',
    'state_sd',
    'state_tn',
    'state_tx',
    'state_ut',
    'state_va',
    'state_vt',
    'state_wa',
    'state_wi',
    'state_wv',
    'state_wy',
    'type_SUV',
    'type_bus',
    'type_convertible',
    'type_coupe',
    'type_hatchback',
    'type_mini-van',
    'type_offroad',
    'type_other',
    'type_pickup',
    'type_sedan',
    'type_truck',
    'type_van',
    'type_wagon',
    'manufacturer_acura',
    'manufacturer_alfa-romeo',
    'manufacturer_aston-martin',
    'manufacturer_audi',
    'manufacturer_bmw',
    'manufacturer_buick',
    'manufacturer_cadillac',
    'manufacturer_chevrolet',
    'manufacturer_chrysler',
    'manufacturer_dodge',
    'manufacturer_fiat',
    'manufacturer_ford',
    'manufacturer_gmc',
    'manufacturer_harley-davidson',
    'manufacturer_honda',
    'manufacturer_hyundai',
    'manufacturer_infiniti',
    'manufacturer_jaguar',
    'manufacturer_jeep',
    'manufacturer_kia',
    'manufacturer_land rover',
    'manufacturer_lexus',
    'manufacturer_lincoln',
    'manufacturer_mazda',
    'manufacturer_mercedes-benz',
    'manufacturer_mercury',
    'manufacturer_mini',
    'manufacturer_mitsubishi',
    'manufacturer_nissan',
    'manufacturer_pontiac',
    'manufacturer_porsche',
    'manufacturer_ram',
    'manufacturer_rover',
    'manufacturer_saturn',
    'manufacturer_subaru',
    'manufacturer_tesla',
    'manufacturer_toyota',
    'manufacturer_volkswagen',
    'manufacturer_volvo',
    'fuel_diesel',
    'fuel_electric',
    'fuel_gas',
    'fuel_hybrid',
    'fuel_other']
one_hot_df = pd.DataFrame(columns=one_hot_columns)
one_hot_df.loc[len(one_hot_df)] = 0

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return (res)

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

    US_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

    states = [s.lower() for s in US_states]
    conditions = ("excellent", "fair", "good", "like new", "new")
    types = ("sedan", "pickup", "truck", "hatchback")
    fuels = ("gas", "hybrid", "electric", "diesel")
    manufacturers = ("gmc", "chevrolet", "ford", "toyota",
                     "volvo", "bmw", "nissan", "lexus", "honda", "tesla")

    with st.form(key='columns_in_form'):
        c1, c2, c3 = st.columns(3)
        with c1:
            Year_Selected = c1.selectbox("Year", year)
            Odometer = c1.slider("Odometer (miles)", 0, 200000)
            State_Selected = c1.selectbox("State", states)

        with c2:
            Fuel_Type = c2.radio("Fuel Type", fuels)
            Body_type = c2.radio("Body Style", types)

        with c3:
            Condition = c3.radio("Condition", conditions)
            Make_Selected = c1.selectbox("Manufacturer", manufacturers)

            dict = {
                'year': int(Year_Selected),
                'odometer': Odometer,
                'manufacturer': Make_Selected,
                'condition': Condition,
                'fuel': Fuel_Type,
                'type': Body_type,
                'state': State_Selected
            }
            df = pd.DataFrame([dict])
            df = encode_and_bind(df, 'condition')
            df = encode_and_bind(df, 'state')
            df = encode_and_bind(df, 'type')
            df = encode_and_bind(df, 'manufacturer')
            df = encode_and_bind(df, 'fuel')
            df = df.merge(one_hot_df, how='outer')
            df.fillna(0, inplace=True)
            df = df[one_hot_df.columns.insert(0, 'odometer').insert(0, 'year')]
            print(df)

        ok = st.form_submit_button("Calculate Car Value")
        if ok:
            prediction = model.predict(xgb.DMatrix(df))
            output = f'{int(prediction[0]):,}'
            print(output)

            st.subheader(f"The estimated Price is ${output}")

        ok2 = st.form_submit_button("Price range by year")
        df_range = pd.DataFrame(columns=['year', 'price'])
        pred_dicts = []
        if ok2:
            for x in range(-5, 5):
                pred_year = int(Year_Selected)+x
                df.at[0, 'year'] = pred_year
                prediction = model.predict(xgb.DMatrix(df))
                output = round(prediction[0], 2)
                pred_dicts.append({'year': str(pred_year), "price": output})
            df_range = pd.concat([df_range, pd.DataFrame.from_records(pred_dicts)])
            print(df_range)
            #df_range = df_range.astype('int')
            chart_data = pd.DataFrame(df_range)

            st.line_chart(chart_data, x="year", y="price")
