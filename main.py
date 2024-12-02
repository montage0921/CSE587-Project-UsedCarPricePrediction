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

# -------------- Constant ----------------------


# read data from csv
@st.cache_data
def get_data():

    df=pd.read_csv("carinfo_after_pre_clean.csv")
    return df

# data cleaning
@st.cache_data
def clean_data(df,list=None):
    df_cleaned_1 = df.dropna(subset = ['exterior_color', 'interior_color', 'fuel','drive_type'])
    df_cleaned = df_cleaned_1[df_cleaned_1['price'] != 0]
    df_unique = df_cleaned.drop_duplicates(subset = 'VIN', keep = False)
    # df_unique = df_unique.drop(columns = ['Unnamed: 0'])

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
def trainKBestSelector(df):
    # split the data
    train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)
    x_train = train_data.drop(columns = ['price'])
    y_train = train_data['price']

    # handling missing data with SimpleImputer
    imputer = SimpleImputer(strategy = "mean")
    x_train_imputed = imputer.fit_transform(x_train)

    # create a selector to get the most kBest importatnt features
    selector = SelectKBest(score_func = f_regression, k = min(20, x_train_imputed.shape[1]))

    #train the selector
    selector.fit_transform(x_train_imputed, y_train)

    return selector

# get the best feature
@st.cache_data(show_spinner=f"selecting the best features ")
def getKBestFeatures(_selector,df):
    selected_features = _selector.get_support(indices = True)
    df_without_price=df.drop(columns=['price'])
    selected_feature_names = df_without_price.columns[selected_features]

    # Extract the base feature names
    base_feature_names = list(set([
    '_'.join(name.split('_')[:-1]) if '_' in name else name
    for name in selected_feature_names
]))

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
    st.title("🚗Used Car Price Predictor")
    df=get_data() # get dataset csv

    # -----------Expander for CSV-------------
    with st.expander("Show the full table"):
        st.write(df)
    
    # --------- Data Processing --------------
    cleaned_df=clean_data(df) # cleaning
    df_encoded=encoded(cleaned_df) # encoding

    # ---------Train KBest Selector ---------
    selector=trainKBestSelector(df_encoded)
    important_features=getKBestFeatures(selector,df_encoded) # e.g. ["mileage", "year","color"]
    important_features.append("price")
    st.write(important_features)
    # --------Train Random Forest Regressor ------- 
    df_important=clean_data(df,important_features)
    df_important_encoded=encoded(df_important)

    randomForestRegressor=trainRandomForestRegressor(df_important_encoded)
    

    # predicted_price=predictByRandomForest(randomForestRegressor,user_input,df_important_encoded)
    # ----------Build the UI Dictionary ------------------------------
    
    make=st.selectbox("Make",options=cleaned_df['make'].unique(),index=0)
    original_price=st.number_input("Original Price",min_value=5000,max_value=1000000,step=1000,value=10000)
    model=st.selectbox(
            "Model",options=cleaned_df[cleaned_df['make'] == make]['model'].unique(),
            index=0)
    feature_input_map={
        'mileage': lambda: st.number_input("Mileage", 0, 200000, step=1000,value=30000),
        'year':lambda:st.number_input("Year", 2010, date.today().year+1, step=1,value=2020),
        'make': lambda:make,
        'model':lambda:st.selectbox(
            "Model",options=cleaned_df[cleaned_df['make'] == make]['model'].unique(),
            index=0),
        "drive_type":lambda:st.selectbox("Drive Type",options=cleaned_df['drive_type'].unique(),index=0),
        "owners":lambda:st.number_input("Number of Owner",1,5,1),
        "class":lambda:st.selectbox("Class",
                                    options=cleaned_df[cleaned_df['make'] == make]['class'].unique(),
                                    index=0),
        "cylinders":lambda:st.selectbox("Number of Cylinder",
                                       options=cleaned_df[cleaned_df['make'] == make]['cylinders'].unique(),
                                       index=0),
        "interior_color":lambda:st.selectbox("Interior Color",options=cleaned_df[cleaned_df['make'] == make]['interior_color'].unique(),index=0),
        "miles_per_gallon":lambda:st.selectbox("Miles Per Gallon",options=cleaned_df[cleaned_df['make'] == make]['miles_per_gallon'].unique(),index=0),
        "fuel":lambda:st.selectbox("Fuel",options=cleaned_df[cleaned_df['make'] == make]['fuel'].unique(),index=0),
        "bed_length":lambda:st.selectbox("Bed Length (Truck Only)",options=cleaned_df[cleaned_df['make'] == make]['bed_length'].unique(),index=0),                   
    }

    
    
    with st.form(key="user_car_info"):
        st.subheader("Enter Your Car's Information!")
        user_input={"make":make,"model":model}
        col1,col2,col3=st.columns(3)
        columns=[col1,col2,col3]
        counter=0
        for feature in important_features:
            if feature !='make' and feature!='price' and feature!='model':
                with columns[counter%3]:
                    user_input[feature]=feature_input_map[feature]()
                counter+=1
        submit=st.form_submit_button("Predict",type="primary")
        

    
    if submit:
        tab1,tab2=st.tabs(["Predicted Report","Graph?"])
        st.write(user_input)
        with tab1:
            predict_price=predictByRandomForest(randomForestRegressor,user_input,df_important_encoded)
            original_price=int(original_price)
            st.metric(label="Predicted Price",value=round(predict_price,2),delta=-round(original_price-predict_price,2),delta_color="normal")
        
    