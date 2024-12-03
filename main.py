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
import seaborn as sns
import plotly.express as px
from pathlib import Path
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

def findImportantFeatures(_model, df_encoded):
    # Extract feature importances from the model
    feature_importances = _model.feature_importances_

    # Get feature names, excluding 'price'
    x_train = df_encoded.drop(columns=['price'])
    important_features = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    return important_features.head(10)

# typewriter effect
def stream_data_common(data, delay: float=0.02):
	for word in data:
		yield word + " "
		time.sleep(delay)

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
                <a href="#" class="hero-button">PREDICT PRICE</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

# Function to calculate price range
def price_range_search(data, car_attributes):
    # Filter the dataset based on car attributes
    filtered_cars = data[
        (data["make"] == car_attributes["make"]) &
        (data["model"] == car_attributes["model"]) &
        (data["year"] == car_attributes["year"])
    ]

    # Calculate the price range
    if not filtered_cars.empty:
        min_price = filtered_cars["price"].min()
        max_price = filtered_cars["price"].max()
        return min_price, max_price
    else:
        return None, None  # Explicitly return None for both min_price and max_price if no match is found

# Function to compare predicted price with the price range
def price_compare(min_price, max_price, predicted_price, car_attributes):
    if min_price is not None and max_price is not None:
        st.write(f"**Price Range for {car_attributes['make']} {car_attributes['model']} ({car_attributes['year']}):**")
        st.write(f"Minimum Price: ${min_price}")
        st.write(f"Maximum Price: ${max_price}")
        st.write(f"Predicted Price: ${predicted_price}")

        # Plot the price range
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([min_price, max_price], [0, 0], color="skyblue", linewidth=10, label="Price Range")
        ax.scatter(predicted_price, 0, color="red", s=100, label="Predicted Price", zorder=5)
        ax.text(predicted_price, 0.1, f"${predicted_price}", color="red", fontsize=12, ha="center")

        # Customize plot
        ax.set_title("Price Range with Predicted Price Highlighted", fontsize=14)
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xlabel("Price (USD)", fontsize=12)
        ax.legend()
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Display the plot
        st.pyplot(fig)
    else:
        st.toast("No matching cars found to determine a price range.")


if __name__ == "__main__":
    background_fig()
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
    # st.write(important_features)
    # --------Train Random Forest Regressor ------- 
    df_important=clean_data(df,important_features)
    df_important_encoded=encoded(df_important)

    randomForestRegressor=trainRandomForestRegressor(df_important_encoded)

    # -------- Show the importance data as a table
    stream_data_md("## 🔍 What Factors Matter Most?")
    # Get the feature importance
    importance_df = findImportantFeatures(randomForestRegressor, df_important_encoded)
    # Create a single container for both the chart and the feature output
    fig = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation='h',  # Horizontal bars
        title="Top 10 Most Important Features",
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
        text="Importance"
    )
    # Customize the hover behavior
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside', hoverinfo='y+x')
    # Layout adjustments
    fig.update_layout(
        height=600,
        width=800,
        title_font=dict(size=18, color='darkblue'),
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


    # predicted_price=predictByRandomForest(randomForestRegressor,user_input,df_important_encoded)
    # ----------Build the UI Dictionary ------------------------------
    
    make=st.selectbox("Make",options=cleaned_df['make'].unique(),index=0)
    original_price=st.number_input("Original Price",min_value=5000,max_value=1000000,step=1000,value=10000)
    model=st.selectbox(
            "Model",options=cleaned_df[cleaned_df['make'] == make]['model'].unique(),
            index=0)
    # year=st.number_input("Year",min_value=2000,max_value=2024,step=1,value=2021)
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
        
    # Define the specific car's attributes
    car_attributes = {
        "make":make,
        "model": model,
        "year": feature_input_map["year"]
    }
    
    if submit:
        tab1,tab2=st.tabs(["Predicted Report","Nice price?"])
        st.write(user_input)
        with tab1:
            predict_price=predictByRandomForest(randomForestRegressor,user_input,df_important_encoded)
            original_price=int(original_price)
            st.metric(label="Predicted Price",value=round(predict_price,2),delta=-round(original_price-predict_price,2),delta_color="normal")
        with tab2:
            min_price, max_price = price_range_search(cleaned_df, car_attributes)
            if min_price is None or max_price is None:
                st.toast("No matching cars found to determine a price range.")
                with st.spinner("Thinking..."):
                    time.sleep(2) # sleep 2s
                    st.write("No matching cars found to determine a price range")
            else:
                price_compare(min_price, max_price, predict_price, car_attributes)
    