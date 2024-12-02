import streamlit as st
import numpy as np


@st.cache_data
def find_range_properties(df,make,model):
    
    if model=="ALL":
        min_price=df[df['make'] == make]['price'].min()
        max_price=df[df['make'] == make]['price'].max()
        min_year=df[df['make'] == make]['year'].min()
        max_year=df[df['make'] == make]['year'].max()
        min_mileage=df[df['make'] == make]['mileage'].min()
        max_mileage=df[df['make'] == make]['mileage'].max()
    else:
        min_price = df[(df['make'] == make) & (df['model'] == model)]['price'].min()
        max_price = df[(df['make'] == make) & (df['model'] == model)]['price'].max()
        min_year = df[(df['make'] == make) & (df['model'] == model)]['year'].min()
        max_year = df[(df['make'] == make) & (df['model'] == model)]['year'].max()
        min_mileage = df[(df['make'] == make) & (df['model'] == model)]['mileage'].min()
        max_mileage = df[(df['make'] == make) & (df['model'] == model)]['mileage'].max()
    return min_price,max_price,min_year,max_year,min_mileage,max_mileage

def display_car_make_selector(df):
    make = st.selectbox("Make", options=df['make'].unique(), index=0)
    model=st.selectbox("Model",options=np.append("ALL",df[df['make'] == make]['model'].unique()),index=0)
    search1=st.button("Search",type="primary")
    return {
        "make":make,
        "model": model
    },search1
def display_filters(df, make, model):
    col1, col2, col3 = st.columns(3)

    search3 = st.button("Apply Other Filter", type='primary')
    if make == "ALL":
        with col1:
            fuel_consumption = st.selectbox("Fuel Consumption", options=np.append("ALL", df['miles_per_gallon'].unique()))
            fuel = st.selectbox("Fuel", options=np.append("ALL", df['fuel'].unique()), index=0)
            is_auction = st.selectbox("Is Auctioned", options=['ALL', 'Yes', 'No'], index=0)
            odometer_issue = st.selectbox("Has odometer issue?", options=['ALL', 'Yes', 'No'], index=0)
            bed_length = st.selectbox("Truck Only", options=np.append("ALL", df['bed_length'].unique()))
        with col2:
            transmission = st.selectbox("Transmission", options=np.append("ALL", df['transmission'].unique()))
            drive_type = st.selectbox("Drive Type", options=np.append("ALL", df['drive_type'].unique()), index=0)
            accident_condition = st.selectbox("Accidents", options=np.append("ALL", df['accidents'].unique()))
            certification = st.selectbox("Certification", options=np.append("ALL", df['certification'].unique()))
            exterior_color = st.selectbox("Exterior Color", options=np.append("ALL", df['exterior_color'].unique()))
        with col3:
            owners = st.selectbox("Owners", options=np.append("ALL", df['owners'].unique()))
            class1 = st.selectbox("Class", options=np.append("ALL", df['class'].unique()))
            open_recalls = st.selectbox("Number of Open Recalls", options=np.append("ALL", df['open_recalls'].unique()))
            cylinder = st.selectbox("Number of Cylinders", options=np.append("ALL", df['cylinders'].unique()), index=0)
            interior_color = st.selectbox("Interior Color", options=np.append("ALL", df['interior_color'].unique()))


    elif model == "ALL":
        with col1:
            fuel_consumption = st.selectbox("Fuel Consumption", options=np.append("ALL", df[df['make'] == make]['miles_per_gallon'].unique()))
            fuel = st.selectbox("Fuel", options=np.append("ALL", df[df['make'] == make]['fuel'].unique()))
            is_auction = st.selectbox("Is Auctioned", options=['Yes', 'No'],index=1)
            odometer_issue = st.selectbox("Has odometer issue?", options=['Yes', 'No'],index=1)
            bed_length = st.selectbox("Truck Only", options=np.append("ALL", df[df['make'] == make]['bed_length'].unique()))
        with col2:
            transmission = st.selectbox("Transmission", options=np.append("ALL", df[df['make'] == make]['transmission'].unique()))
            drive_type = st.selectbox("Drive Type", options=np.append("ALL", df[df['make'] == make]['drive_type'].unique()))
            accident_condition = st.selectbox("Accidents", options=np.append("ALL", df[df['make'] == make]['accidents'].unique()))
            certification = st.selectbox("Certification", options=np.append("ALL", df[df['make'] == make]['certification'].unique()))
            exterior_color = st.selectbox("Exterior Color", options=np.append("ALL", df[df['make'] == make]['exterior_color'].unique()))
        with col3:
            owners = st.selectbox("Owners", options=np.append("ALL", df[df['make'] == make]['owners'].unique()))
            class1 = st.selectbox("Class", options=np.append("ALL", df[df['make'] == make]['class'].unique()))
            open_recalls = st.selectbox("Number of Open Recalls", options=np.append("ALL", df[df['make'] == make]['open_recalls'].unique()))
            cylinder = st.selectbox("Number of Cylinders", options=np.append("ALL", df[df['make'] == make]['cylinders'].unique()))
            interior_color = st.selectbox("Interior Color", options=np.append("ALL", df[df['make'] == make]['interior_color'].unique()))
    else:
        with col1:
            fuel_consumption = st.selectbox("Fuel Consumption", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['miles_per_gallon'].unique()))
            fuel = st.selectbox("Fuel", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['fuel'].unique()))
            is_auction = st.selectbox("Is Auctioned", options=['Yes', 'No'],index=1)
            odometer_issue = st.selectbox("Has odometer issue?", options=['Yes', 'No'],index=1)
            bed_length = st.selectbox("Bed Length(Truck Only)", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['bed_length'].unique()))
        with col2:
            transmission = st.selectbox("Transmission", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['transmission'].unique()))
            drive_type = st.selectbox("Drive Type", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['drive_type'].unique()))
            accident_condition = st.selectbox("Accidents", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['accidents'].unique()))
            certification = st.selectbox("Certification", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['certification'].unique()))
            exterior_color = st.selectbox("Exterior Color", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['exterior_color'].unique()))
        with col3:
            owners = st.selectbox("Owners", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['owners'].unique()))
            class1 = st.selectbox("Class", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['class'].unique()))
            open_recalls = st.selectbox("Number of Open Recalls", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['open_recalls'].unique()))
            cylinder = st.selectbox("Number of Cylinders", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['cylinders'].unique()))
            interior_color = st.selectbox("Interior Color", options=np.append("ALL", df[(df['make'] == make) & (df['model'] == model)]['interior_color'].unique()))

    return { 
        "fuel_consumption": fuel_consumption,
        "fuel": fuel,
        "is_auction": is_auction,
        "odometer_issue": odometer_issue,
        "bed_length": bed_length,
        "transmission": transmission,
        "drive_type": drive_type,
        "accident_condition": accident_condition,
        "certification": certification,
        "exterior_color": exterior_color,
        "owners": owners if owners=="ALL" else int(owners),
        "class1": class1,
        "open_recalls": open_recalls if open_recalls=="ALL" else int(open_recalls),
        "cylinder": cylinder if cylinder=="ALL" else int(cylinder),
        "interior_color": interior_color,
    }, search3
def display_keyinfo_search(min_price=0,max_price=1000000,min_year=2010,max_year=2030,min_mileage=0,max_mileage=1000000):
    col1, col2= st.columns(2)
    set_year=st.radio("Set Range of Year",options=["Yes","No"],index=1,horizontal=True)
    set_price=st.radio("Set Range of Price",options=["Yes","No"],index=1,horizontal=True)
    set_mileage=st.radio("Set Range of Mileage",options=["Yes","No"],index=1,horizontal=True) 
    if set_year=="Yes":
        with col1:
            min_year=st.number_input("Min Year", min_value=min_year, max_value=max_year,step=1,value=min_year)
        with col2:
            max_year=st.number_input("Max Year", min_value=min_year, max_value=max_year,step=1,value=max_year)
    if set_price=="Yes":
        with col1:
            min_price=st.number_input("Min Price",min_value=min_price,max_value=max_price,step=1000,value=min_price)
        with col2:
            max_price=st.number_input("Max Price",min_value=min_price,max_value=max_price,step=1000,value=max_price)
    if set_mileage=="Yes":
        with col1:
            min_mileage=st.number_input("Min Milage",min_value=min_mileage,max_value=max_mileage,step=1000,value=min_mileage)
        with col2:
            max_mileage=st.number_input("Max Milage",min_value=min_mileage,max_value=max_mileage,step=1000,value=max_mileage)

    search3 = st.button("Search With Key Infomation", type="primary")

    return { "min_year": min_year, "max_year": max_year, 
            "min_price": min_price, "max_price": max_price, 
            "min_mileage": min_mileage, "max_mileage": max_mileage, }, search3

