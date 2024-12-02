import streamlit as st
from display_logic import *
import pandas as pd

def display_search_UI(df):
    make = st.selectbox("Make", options=df['make'].unique(), index=0)
    model=st.selectbox("Model",options=np.append("ALL",df[df['make'] == make]['model'].unique()),index=0)
    search1=st.button("Search",type="primary")
    min_price,max_price,min_year,max_year,min_mileage,max_mileage=find_range_properties(df,make,model)

    with st.expander("Define the search range"):
        ranges,search2=display_keyinfo_search(min_price,max_price,min_year,max_year,min_mileage,max_mileage)
    with st.expander("Apply more filters"):
        filters,search3=display_filters(df,make,model)
    
    if search1:
        display_search1(df,make,model)

    if search2:
        display_search2(df,model,make,ranges)
    
    if search3:
        filtered_df=filter_price_mileage_year(df,model,make,ranges)
        filtered_df=filter_other_features(filters,filtered_df)
        
        if(len(filtered_df)==0):
            st.error("No Matches Found")
        else:
            st.write(f"Found {len(filtered_df)} matching records.")
            st.write(filtered_df)


def display_search1(df,make,model):
    st.write(model)
    if model=="ALL":
        results = df[(df['make'] == make)]
        st.write(f"Found {len(results)} records for {make}")
        st.write(results)
    else:
        results = df[(df['make'] == make) & (df['model'] == model)]
        st.write(f"Found {len(results)} records for {make} {model}")
        st.write(results)

def display_search2(df,model,make,ranges):
    filtered_df=filter_price_mileage_year(df,model,make,ranges)

    # Display the results
    st.write(f"Found {len(filtered_df)} matching records.")
    st.write(filtered_df)

    return filtered_df

def filter_price_mileage_year(df,model,make,ranges):
    if model=="ALL": # if choose all, then df should include all the models in that make, equivalent to search without setting range
        filtered_df = df[df['make']==make] 
    else:
        filtered_df=df[df['model']==model]
    # Apply filters based on non-None values
    if ranges["min_year"] is not None:
        filtered_df = filtered_df[filtered_df["year"] >= ranges["min_year"]]
    if ranges["max_year"] is not None:
        filtered_df = filtered_df[filtered_df["year"] <= ranges["max_year"]]
    if ranges["min_price"] is not None:
        filtered_df = filtered_df[filtered_df["price"] >= ranges["min_price"]]
    if ranges["max_price"] is not None:
        filtered_df = filtered_df[filtered_df["price"] <= ranges["max_price"]]
    if ranges["min_mileage"] is not None:
        filtered_df = filtered_df[filtered_df["mileage"] >= ranges["min_mileage"]]
    if ranges["max_mileage"] is not None:
        filtered_df = filtered_df[filtered_df["mileage"] <= ranges["max_mileage"]]
    
    return filtered_df

def filter_other_features(filters,filtered_df):
    # apply filters based on the user inputs
    # filter=="ALL" means we can skip it
    odometer_issue_bool = True if filters["odometer_issue"] == "Yes" else False
    filtered_df = filtered_df[filtered_df["has_odometer_issue"] == odometer_issue_bool]
    is_auction_bool = True if filters["is_auction"] == "Yes" else False
    filtered_df = filtered_df[filtered_df["is_auction"] == is_auction_bool]


    if filters["fuel_consumption"] != "ALL":
        filtered_df = filtered_df[filtered_df["miles_per_gallon"] == filters["fuel_consumption"]]
    if filters["fuel"] != "ALL":
        filtered_df = filtered_df[filtered_df["fuel"] == filters["fuel"]]
    if filters["bed_length"] != "ALL":
        filtered_df = filtered_df[filtered_df["bed_length"] == filters["bed_length"]]
    if filters["transmission"] != "ALL":
        filtered_df = filtered_df[filtered_df["transmission"] == filters["transmission"]]
    if filters["drive_type"] != "ALL":
        filtered_df = filtered_df[filtered_df["drive_type"] == filters["drive_type"]]
    if filters["accident_condition"] != "ALL":
        filtered_df = filtered_df[filtered_df["accidents"] == filters["accident_condition"]]
    if filters["certification"] != "ALL":
        filtered_df = filtered_df[filtered_df["certification"] == filters["certification"]]
    if filters["exterior_color"] != "ALL":
        filtered_df = filtered_df[filtered_df["exterior_color"] == filters["exterior_color"]]
    if filters["owners"] !="ALL":
        filtered_df = filtered_df[filtered_df["owners"] == filters["owners"]]
    if filters["class1"] != "ALL":
        filtered_df = filtered_df[filtered_df["class"] == filters["class1"]]
    if filters["open_recalls"] !="ALL":
        filtered_df = filtered_df[filtered_df["open_recalls"] == filters["open_recalls"]]
    if filters["cylinder"] !="ALL":
        filtered_df = filtered_df[filtered_df["cylinders"] == filters["cylinder"]]
    if filters["interior_color"] != "ALL":
        filtered_df = filtered_df[filtered_df["interior_color"] == filters["interior_color"]]
    
    return filtered_df
