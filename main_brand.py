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
# -------------- Constant ----------------------
kBest=10

# read data from csv
@st.cache_data
def get_data():
    df=pd.read_csv("carinfo_after_pre_clean.csv")
    return df

# data cleaning
@st.cache_data
def clean_data(df,list=None):
    df_cleaned_1 = df.dropna(subset = ['exterior_color', 'interior_color', 'fuel','Drive type'])
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
    st.title("🚗Used Car Price Predictor")
    df=get_data() # get dataset csv

    # -----------Expander for CSV-------------
    with st.expander("Show the full table"):
        st.write(df)
    
    # --------- Data Processing --------------
    cleaned_df=clean_data(df) # cleaning
    df_encoded=encoded(cleaned_df) # encoding

    # ---------Train KBest Selector ---------
    selector=trainKBestSelector(df_encoded,kBest)
    important_features=getKBestFeatures(selector,df_encoded) # e.g. ["mileage", "year","color"]
    important_features.append("price")
    # --------Train Random Forest Regressor ------- 
    df_important=clean_data(df,important_features)
    df_important_encoded=encoded(df_important)

    randomForestRegressor=trainRandomForestRegressor(df_important_encoded)
    

    # predicted_price=predictByRandomForest(randomForestRegressor,user_input,df_important_encoded)
    # ----------Build the UI Dictionary ------------------------------
    
    make=st.selectbox("Make",options=cleaned_df['make'].unique(),index=0)
    original_price=st.number_input("Original Price",min_value=5000,max_value=1000000,step=1000,value=10000)
    feature_input_map={
        'mileage': lambda: st.slider("Mileage", 0, 200000, step=1000,value=30000),
        'year':lambda:st.slider("Year", 2010, date.today().year+1, step=1,value=2020),
        'make': lambda:make,
        'model':lambda:st.selectbox(
            "Model",options=cleaned_df[cleaned_df['make'] == make]['model'].unique(),
            index=0),
        "Drive type":lambda:st.selectbox("Drive Type",options=cleaned_df['Drive type'].unique(),index=0),
        "owner":lambda:st.slider("Number of Owner",1,5,1),
        "class":lambda:st.selectbox("Class",
                                    options=cleaned_df[cleaned_df['make'] == make]['class'].unique(),
                                    index=0),
        "cylinders":lambda:st.selectbox("Number of Cylinder",
                                       options=cleaned_df[cleaned_df['make'] == make]['cylinders'].unique(),
                                       index=0)                    
    }

    
    with st.form(key="user_car_info"):
        st.subheader("Enter Your Car's Information!")
        user_input={"make":make}
        col1,col2,col3=st.columns(3)
        columns=[col1,col2,col3]
        counter=0
        for feature in important_features:
            if feature !='make' and feature!='price':
                with columns[counter%3]:
                    user_input[feature]=feature_input_map[feature]()

            counter+=1
        submit=st.form_submit_button("Predict",type="primary")

    
    if submit:
        tab1,tab2=st.tabs(["Predicted Report","Brand Analysis"])
        with tab1:
            predict_price=predictByRandomForest(randomForestRegressor,user_input,df_important_encoded)
            original_price=int(original_price)
            st.metric(label="Predicted Price",value=round(predict_price,2),delta=-round(original_price-predict_price,2),delta_color="normal")
       


st.header("Brand Analysis")
st.write("Analyze the influence of different brands on car prices using XGBoost.")

data = pd.read_csv("carinfo_after_pre_clean.csv")
select_columns = ['make', 'model', 'price', 'year', 'mileage', 'owner']
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


