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
from xgboost import XGBRegressor
import seaborn as sns
import pymysql
import yaml
from yaml.loader import SafeLoader
# -------------- Constant ----------------------
kBest=10

with open('../config.yaml') as file:
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
    st.toast("Connect to database successfully!")
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
    df_cleaned_1 = df.dropna(subset = ['exterior_color', 'interior_color', 'fuel','drive_type'])
    df_cleaned = df_cleaned_1[df_cleaned_1['price'] != 0]
    df_unique = df_cleaned.drop_duplicates(subset = 'VIN', keep = False)    
    df_unique = df_unique.drop(columns = ['Unnamed: 0'])

    if list:
        df_unique=df_unique[list]
    return df_unique

# one-hot encoded
@st.cache_data
def encoded(df):
    df_encoded = pd.get_dummies(df)
    if 'fuel_encoded' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns = ['fuel_encoded'])
    return df_encoded

# train the best-feature selector
@st.cache_resource(show_spinner="training KBest-Selector...")
def trainKBestSelector(df,kBest):
    # split the data
    train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)
    x_train = train_data.drop(columns = ['price'])
    y_train = train_data['price']

    # handling missing data with SimpleImputer
    imputer = SimpleImputer(strategy = "mean")
    x_train_imputed = imputer.fit_transform(x_train)

    # create a selector to get the most kBest importatnt features
    selector = SelectKBest(score_func = f_regression, k = min(kBest, x_train_imputed.shape[1]))

    #train the selector
    selector.fit_transform(x_train_imputed, y_train)

    return selector

# get the best feature
@st.cache_data(show_spinner=f"selecting the best {kBest} features ")
def getKBestFeatures(_selector,df):
    selected_features = _selector.get_support(indices = True)
    df_without_price=df.drop(columns=['price'])
    selected_feature_names = df_without_price.columns[selected_features]

    # Extract the base feature names
    base_feature_names = list(set([name.split('_')[0] for name in selected_feature_names]))
    
    return base_feature_names

@st.cache_resource(show_spinner="Training random forest regressor")
def trainRandomForestRegressor(df_encoded):
    train_data, test_data = train_test_split(df_encoded, test_size = 0.5, random_state = 30)
    x_train = train_data.drop(columns=['price'])
    x_test = test_data.drop(columns=['price'])
    y_train = train_data['price']
    y_test = test_data['price']

    # Learn n_estimators
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250]
    }

    # Initialize the model with a fixed random_state
    model = RandomForestRegressor(random_state = 42)

    # Perform Grid Search
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
    grid_search.fit(x_train, y_train)

    # Get the best number of estimators
    best_n_estimators = grid_search.best_params_['n_estimators']

    # Initialize and fit the model
    # best_n_estimators = 200
    model = RandomForestRegressor(n_estimators = best_n_estimators, random_state = 42)
    model.fit(x_train, y_train)

    return model

@st.cache_data
def predictByRandomForest(_model,user_input,df_important_encoded):

    user_data = pd.DataFrame([user_input])

    user_data_encoded=pd.get_dummies(user_data)
    user_data_encoded = user_data_encoded.reindex(columns=df_important_encoded.drop(columns=['price']).columns, fill_value=0)
    predicted_price = _model.predict(user_data_encoded)
    return predicted_price[0]


if __name__ == "__main__":
    conn,cursor=build_connection_with_database()

    df=get_data(conn) # get dataset csv

    st.title("Brand Analysis")
    st.write("Analyze the influence of different brands on car prices using XGBoost.")

    data = pd.read_csv("../cleaned_for_sql.csv")

    select_columns = ['make', 'model', 'price', 'year', 'mileage', 'owners']
    data = data.dropna(subset=select_columns)
    df_selected = data[select_columns]
    df_q5 = df_selected[(df_selected['price'] != 0) & (df_selected['year'] != 2025)]
    df = df_q5

    brand_avg_prices = df.groupby(['year', 'make'])['price'].mean().unstack()
    brand_avg_prices.fillna(method='ffill', inplace=True)
    brand_avg_prices.fillna(method='bfill', inplace=True)

    brands = brand_avg_prices.columns
    top_features_dict = {}  

    for target_brand in brands:
        X = brand_avg_prices.drop(columns=[target_brand], errors='ignore')
        y = brand_avg_prices[target_brand]

        if y.isnull().all():
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(3)
        top_features_dict[target_brand] = top_features

    top_features_df = pd.DataFrame.from_dict(top_features_dict, orient='index').fillna(0)

    st.subheader("Top Influential Brands for Each Target Brand")
    st.write(top_features_df)


    st.subheader("Explore Individual Brand Influence")
    selected_brand = st.selectbox("Choose a target brand:", options=top_features_df.index)
    if selected_brand in top_features_df.index:
        top_3_influences = top_features_df.loc[selected_brand].nlargest(3)
        
        st.write(f"### Top Influential Brands for {selected_brand}")
        st.bar_chart(top_3_influences)


