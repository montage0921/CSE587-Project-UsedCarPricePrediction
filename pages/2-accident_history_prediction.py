import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date,time
import numpy as np
import time
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
from xgboost import XGBRegressor
import pymysql
import yaml
from yaml.loader import SafeLoader
# -------------- Constant ----------------------

# build connection to sql database
with open('config.yaml') as file:
    config=yaml.load(file,Loader=SafeLoader)

@st.cache_resource
def connect_to_database():
    conn = pymysql.connect(
        host=config['credentials']['database']['host'],
        port=3306,
        user=config['credentials']['database']['user'],
        password=config['credentials']['database']['password'],
        database=config['credentials']['database']['database']
    )
    return conn

# @st.cache_resource(show_spinner="Connecting to database...")
def build_connection_with_database():
    conn=connect_to_database()
    cursor=conn.cursor()
    st.write("Connect to database successfully!")
    return (conn,cursor)


# read data from sql
@st.cache_data
def get_data(_conn):
    extract_all_query="""SELECT * FROM used_cars"""
    df=pd.read_sql(extract_all_query,_conn)
    df = df.drop(columns=["id"])
    return df

# data cleaning
@st.cache_data
def clean_data(df,list=None):
    df_cleaned = df[df['price'] != 0]
    df_cleaned['model'] = df_cleaned['model'].str.strip()
    return df_cleaned

@st.cache_data
def calculate_average(mpg):
    try:
        city_mpg, highway_mpg = mpg.split(' city/')[0], mpg.split('/')[1].split(' hwy')[0]
        return (int(city_mpg) + int(highway_mpg)) / 2
    except (IndexError, ValueError):
        return None  

@st.cache_data(show_spinner="Clean input make's data")
def clean_brand_data(df,brand):
    df_dmg=df.dropna(subset=['mileage','year','make','model','miles_per_gallon','cylinders','price','accidents']) 
    df_dmg = df_dmg[df_dmg['price']!=0] 
    brand_data=df_dmg.loc[df_dmg['make'].isin([brand])]
    brand_data=pd.concat([brand_data], axis=0)
    brand_data=brand_data[['mileage','year','model','miles_per_gallon','cylinders','price','accidents']]
    brand_data['accidents'] = np.where(brand_data['accidents'] == 'No Issue', 0, 1)
    brand_data['model_id'], _ =pd.factorize(brand_data['model'])
    brand_data['Average MPG'] = brand_data['miles_per_gallon'].apply(calculate_average)
    return brand_data

@st.cache_data(show_spinner="Training neural_network accident predictor")
def train_accident_neural_network(brand_data):
    brand_data=brand_data.dropna()
    X = brand_data.drop(['accidents','model','miles_per_gallon'], axis=1)
    X = X
    y = brand_data['accidents']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test[X_test.index]

    brand_mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, activation='logistic', solver='adam', random_state=42)
    brand_mlp_model.fit(X_train, y_train)
    mlp_predictions = brand_mlp_model.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_predictions)

    X_price = X.drop(['price'], axis=1).dropna()
    X_train_price, X_test_price, y_train, y_test = train_test_split(X_price, y, test_size=0.1)
   
    brand_mlp_model_price = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, activation='logistic', solver='adam', random_state=42)
    brand_mlp_model_price.fit(X_train_price, y_train)
    mlp_predictions = brand_mlp_model_price.predict(X_test_price)
    mlp_accuracy_price = accuracy_score(y_test, mlp_predictions)

    X_mileage = X.drop(['mileage'], axis=1).dropna()
    X_train_mileage, X_test_mileage, y_train, y_test = train_test_split(X_mileage, y, test_size=0.1)

    brand_mlp_model_mileage = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, activation='logistic', solver='adam', random_state=42)
    brand_mlp_model_mileage.fit(X_train_mileage, y_train)
    mlp_predictions = brand_mlp_model_mileage.predict(X_test_mileage)
    mlp_accuracy_mileage = accuracy_score(y_test, mlp_predictions)

    X_year = X.drop(['year'], axis=1).dropna()
    X_train_year, X_test_year, y_train, y_test = train_test_split(X_year, y, test_size=0.1)

    brand_mlp_model_year = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, activation='logistic', solver='adam', random_state=42)
    brand_mlp_model_year.fit(X_train_year, y_train)
    mlp_predictions = brand_mlp_model_year.predict(X_test_year)
    mlp_accuracy_year = accuracy_score(y_test, mlp_predictions)

    X_cylinders = X.drop(['cylinders'], axis=1).dropna()
    X_train_cylinders, X_test_cylinders, y_train, y_test = train_test_split(X_cylinders, y, test_size=0.1)

    brand_mlp_model_cylinders = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, activation='logistic', solver='adam', random_state=42)
    brand_mlp_model_cylinders.fit(X_train_cylinders, y_train)
    mlp_predictions = brand_mlp_model_cylinders.predict(X_test_cylinders)
    mlp_accuracy_cylinders = accuracy_score(y_test, mlp_predictions)

    X_mpg = X.drop(['Average MPG'], axis=1).dropna()
    X_train_MPG, X_test_MPG, y_train, y_test = train_test_split(X_mpg, y, test_size=0.1)

    brand_mlp_model_MPG = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, activation='logistic', solver='adam', random_state=42)
    brand_mlp_model_MPG.fit(X_train_MPG, y_train)
    mlp_predictions = brand_mlp_model_MPG.predict(X_test_MPG)
    mlp_accuracy_MPG = accuracy_score(y_test, mlp_predictions)

    X_id = X.drop(['model_id'], axis=1).dropna()
    X_train_model_id, X_test_model_id, y_train, y_test = train_test_split(X_id, y, test_size=0.1)

    brand_mlp_model_model_id = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, activation='logistic', solver='adam', random_state=42)
    brand_mlp_model_model_id.fit(X_train_model_id, y_train)
    mlp_predictions = brand_mlp_model_model_id.predict(X_test_model_id)
    mlp_accuracy_model_id = accuracy_score(y_test, mlp_predictions)
    
    return brand_mlp_model,[mlp_accuracy,mlp_accuracy_price,mlp_accuracy_mileage,mlp_accuracy_year,mlp_accuracy_cylinders,mlp_accuracy_MPG,mlp_accuracy_model_id]

@st.cache_data
def predict_accident(_predictor,make_data,mileage,year,cylinders,price,mpg,model):
    mpg=calculate_average(mpg)
    model=make_data[make_data['model']==model]['model_id'].iloc[0]
    # user_data = pd.DataFrame([mileage,year,cylinders,price,mpg,model])
    predicted_price = _predictor.predict([[mileage,year,cylinders,price,mpg,model]])
    return predicted_price[0]

# one-hot encoded
@st.cache_data
def encoded(df):
    df_encoded = pd.get_dummies(df)
    if 'fuel_encoded' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns = ['fuel_encoded'])
    return df_encoded

def stream_data_md(text, delay=0.1):
    placeholder = st.empty()  # Create an empty placeholder
    streamed_text = ""  # Initialize the text accumulator
    for word in text.split(" "):  # Split the text into words
        streamed_text += word + " "  # Add the word and a space
        placeholder.markdown(streamed_text)  # Update the placeholder
        time.sleep(delay)  # Add delay for streaming effect

def background_fig():

    # Set page configuration to wide layout
    st.set_page_config(layout="wide")

    # HTML and CSS for the background image with overlay text and button
    st.markdown(
        """
        <style>
            .hero-container {
                position: relative;
                width: 100%;
                height: 500px; /* Adjust the height to your preference */
                background-image: url('https://mystrongad.com/VLB_BerglundVolvoLynchburg/Interactive/Used/Used-Car-page.png'); /* Replace with your image path */
                background-size: cover;
                background-position: center;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .hero-content {
                text-align: center;
                color: white;
            }
            .hero-title {
                font-size: 36px;
                font-weight: bold;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            .hero-button {
                display: inline-block;
                background-color: #FFCC00;
                color: blue;
                padding: 15px 30px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                text-decoration: none;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
            }
            .hero-button:hover {
                background-color: #E6B800;
            }
        </style>
        <div class="hero-container">
            <div class="hero-content">
                <div class="hero-title">A huge database of over 20,000 cars, easily predicting car prices</div>
                <a href="#" class="hero-button">PREDICT CAR ACCIDENT</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

if __name__ == "__main__":
    background_fig()
    st.title("🚗Accident History Predictior")
    conn,cursor=build_connection_with_database()

    df=get_data(conn) # get dataset csv
    # # -----------Expander for CSV-------------
    # with st.expander("Show the full table"):
    #     st.write(df)
    
    # --------- Data Processing --------------
    cleaned_df=clean_data(df) # cleaning
    df_encoded=encoded(cleaned_df) # encoding
    # --------- Accident Prediction--------------
    make=st.selectbox("Make",options=cleaned_df['make'].unique(),index=0)
    model=st.selectbox("Model",options=cleaned_df[cleaned_df['make'] == make]['model'].unique(),index=0)
    year=st.number_input("Year",min_value=2000,max_value=2024,step=1,value=2021)
    mileage=st.number_input("Mileage", 0, 200000, step=1000,value=30000)
    resale_price=st.number_input("Resale Price",min_value=1000,max_value=1000000,step=1000,value=10000)
    cylinders=st.selectbox("cylinder Number",options=cleaned_df[cleaned_df['model'] == model]['cylinders'].unique(),index=0)
    miles_per_gallon=st.selectbox("miles_per_gallon",options=cleaned_df[cleaned_df['model'] == model]['miles_per_gallon'].unique(),index=0)
    make_data= clean_brand_data(cleaned_df,make)
    accident_predictor,accuracies=train_accident_neural_network(make_data)
    accident_prediction=predict_accident(accident_predictor,make_data,mileage,year,cylinders,resale_price,miles_per_gallon,model)
    with st.form("prediction_form"):
        submit = st.form_submit_button("Predict", type="primary")
    if submit:
        tab1=st.tabs(["If the car had an accident?"])
        with tab1[0]: 
            if accident_prediction == 0:
                st.write("No")
            else:
                st.write("Yes")
    # -------- Plot the feature importance analysis
    stream_data_md("## 🔍 What Factors Matter?")
    # Get the feature importance
    accuracies=[accuracies[i+1]-accuracies[0] for i in range(len(accuracies)-1)]
    model_variants = ["Without Price", "Without Mileage", "Without Year", "Without Cylinders", "Without MPG", "Without Model ID"]
    df = pd.DataFrame({
        'Model Variants': model_variants,
        'Accuracy (%)': accuracies
    })

    # Create the bar chart
    fig = px.bar(df, x='Model Variants', y='Accuracy (%)', title="Accident Prediction Accuracy with Different Features Omitted")

    # Display the chart in Streamlit
    st.title('Accident Prediction Model Analysis')
    st.write("This visualization shows the accuracy of the accident prediction model with various features omitted.")
    st.plotly_chart(fig)
    

