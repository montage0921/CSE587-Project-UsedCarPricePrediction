import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from search_result_logic import *
from edit_logic import *
from delete_logic import *
from add_logic import *
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import main as ma

train_para_flag = False
train_all_para_flag = False
feature_number = 9
test_flag = True
update_ymal_flag = False

def findKeyFeaturesByRandomForest(_selector, df_encoded, df):
    # Get feature importances and their indices
    feature_importances = _selector.feature_importances_
    sorted_indices = np.argsort(feature_importances)  # Indices sorted in ascending order

    # Get the top N feature indices
    top_indices = sorted_indices[-100:]
    top_indices = top_indices[::-1]  # Reverse to descending for top features

    # Get encoded feature names
    df_encoded_without_price = df_encoded.drop(columns=['price'])
    selected_feature_names = df_encoded_without_price.columns[top_indices].tolist()
    selected_feature_importances = feature_importances[top_indices]

    # Map encoded features back to original column names
    base_feature_names = []
    for selected_name in selected_feature_names:
        for name in df.columns:
            if name in selected_name:
                base_feature_names.append(name)

    # Remove duplicates while maintaining order
    base_feature_names = list(dict.fromkeys(base_feature_names))

    # Create a sorted DataFrame of features and their importance
    feature_importance_df = pd.DataFrame({
        'Feature': base_feature_names,
        'Importance': selected_feature_importances[:len(base_feature_names)]
    }).sort_values(by='Importance', ascending=True)

    return feature_importance_df

@st.cache_resource(show_spinner="Training RandomForest Regressor by all features")
def trainRandomForestRegressor_by_all(df_encoded):
    # Feed all features
    train_data, test_data = train_test_split(df_encoded, test_size = 0.2, random_state = 42)
    x_train = train_data.drop(columns=['price'])
    x_test = test_data.drop(columns=['price'])
    y_train = train_data['price']
    y_test = test_data['price']

    if train_all_para_flag:
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
        best_model = RandomForestRegressor(n_estimators = best_n_estimators, random_state = 42) # random_state = 50
        best_model.fit(x_train, y_train)
    else:
        best_n_estimators = 200
        best_model = RandomForestRegressor(n_estimators = best_n_estimators, random_state = 42) # random_state = 50
        best_model.fit(x_train, y_train)

    # Make predictions
    y_pred = best_model.predict(x_test)

    # Calculate performance for regression
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tolerance = 0.1  # Define tolerance as 10% of the true value
    accuracy = np.mean(np.abs(y_test - y_pred) <= (tolerance * y_test)) * 100
    evaluate = [rmse, r2, accuracy]

    return best_model, evaluate


@st.cache_resource(show_spinner="Training CatBoost Regressor by all features")
def trainCatBoostRegressor_by_all(df):
    # Handle missing values
    df = df.fillna({
        col: 'unknown' if df[col].dtype == 'object' else df[col].mean()
        for col in df.columns
    })

    # Separate target and features
    y = df['price']
    x = df.drop(columns=["price"])

    # Identify categorical features dynamically
    categorical_features = x.select_dtypes(include=['object']).columns.tolist()
    cat_feature_indices = [x.columns.get_loc(col) for col in categorical_features]

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if train_all_para_flag:
    # Define parameter grid
        param_grid = {
            'iterations': [200, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8, 10]
        }

        # Initialize the CatBoost Regressor
        model = CatBoostRegressor(cat_features=cat_feature_indices, verbose=0)

        # Perform grid search using CatBoost's grid search utility
        grid_search_result = model.grid_search(param_grid, X=x_train, y=y_train, cv=3, verbose=100)

        # Refit model with best parameters
        best_params = grid_search_result['params']
    else:
        best_params = {
            'iterations': 800,
            'learning_rate': 0.1,
            'depth': 6
        }

    best_model = CatBoostRegressor(**best_params, cat_features=cat_feature_indices, verbose=0)
    best_model.fit(x_train, y_train)

    best_model = CatBoostRegressor(**best_params, cat_features=cat_feature_indices, verbose=0)
    best_model.fit(x_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(x_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tolerance = 0.1  # Define tolerance as 10% of the true value
    accuracy = np.mean(np.abs(y_test - y_pred) <= (tolerance * y_test)) * 100

    evaluate = [rmse, r2, accuracy]

    return best_model, evaluate


# Dynamically train CatBoostRegressor model
def CatBoostRegressor_model(base_feature_names, df):
    # Prepare the dataset
    df = df[base_feature_names + ['price']]

    # Handle missing values
    df = df.fillna({
        col: 'unknown' if df[col].dtype == 'object' else df[col].mean()
        for col in base_feature_names
    })

    # Separate target and features
    y = df['price']
    x = df[base_feature_names]

    # Identify categorical features dynamically
    categorical_features = x.select_dtypes(include=['object']).columns.tolist()
    cat_feature_indices = [x.columns.get_loc(col) for col in categorical_features]

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    if train_para_flag:
    # Define parameter grid
        param_grid = {
            'iterations': [200, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8, 10]
        }

        # Initialize the CatBoost Regressor
        model = CatBoostRegressor(cat_features=cat_feature_indices, verbose=0)

        # Perform grid search using CatBoost's grid search utility
        grid_search_result = model.grid_search(param_grid, X=x_train, y=y_train, cv=3, verbose=100)

        # Output the best parameters
        print("Best Parameters:", grid_search_result['params'])

        # Refit model with best parameters
        best_params = grid_search_result['params']
    else:
        best_params = {
            'iterations': 800,
            'learning_rate': 0.1,
            'depth': 6
        }
    best_model = CatBoostRegressor(**best_params, cat_features=cat_feature_indices, verbose=0)
    best_model.fit(x_train, y_train)

    # Predict on test data
    y_pred = best_model.predict(x_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tolerance = 0.1  # Define tolerance as 10% of the true value
    accuracy = np.mean(np.abs(y_test - y_pred) <= (tolerance * y_test)) * 100

    evaluate = [rmse, r2, accuracy]

    return best_model, evaluate

def page_background_fig():

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
                background-image: url('https://bitrefine.group/images/1920x870/optimal_price_1920x870.jpg'); /* Replace with your image path */
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
            .hero-button:hover {
                background-color: #E6B800;
            }
        </style>
        <div class="hero-container">
            <div class="hero-content">
                <div class="hero-title">Regressor Model, easily exploring how many parameters can be input to predict more accurate prices</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)


if __name__ == "__main__":
    # --------- Add background --------------
    page_background_fig()
    st.markdown("### **Accuracy about what you input???**")

    # --------- Build Connection --------------
    conn,cursor=ma.build_connection_with_database()

    # --------- Data Processing --------------
    df=ma.get_data(conn) # get dataset csv
    cleaned_df=ma.clean_data(df) # cleaning
    df_encoded=ma.encoded(cleaned_df) # encoding

    # -----------Expander for CSV-------------
    with st.expander("Show the full table"):
        st.write(df)

    best_model_RF, evaluate_RF = trainRandomForestRegressor_by_all(df_encoded)
    key_base_feature_info = findKeyFeaturesByRandomForest(best_model_RF, df_encoded, cleaned_df)
    key_base_feature_names = key_base_feature_info["Feature"]
    best_model_CB_all, evaluate_CB_all = trainCatBoostRegressor_by_all(cleaned_df)

    # -------- Insights performance of random forest and catboost
    st.subheader("Insights")
    st.markdown("> The graphs above compare the performance metrics (RMSE, R² Score, and Accuracy)")
    # Metric labels
    metrics = ["RMSE", "R2 Score", "Accuracy"]
    columns = st.columns(3)
    # Graph 1: RMSE
    with columns[0]:
        st.subheader("RMSE")
        fig1, ax1 = plt.subplots()
        ax1.bar(["CatBoost", "Random Forest"], [evaluate_CB_all[0], evaluate_RF[0]], color=['blue', 'orange'])
        ax1.set_ylabel("RMSE")
        st.pyplot(fig1)

    # Graph 2: R2 Score
    with columns[1]:
        st.subheader("R2 Score")
        fig2, ax2 = plt.subplots()
        ax2.bar(["CatBoost", "Random Forest"], [evaluate_CB_all[1], evaluate_RF[1]], color=['blue', 'orange'])
        ax2.set_ylabel("R2 Score")
        st.pyplot(fig2)

    # Graph 3: Accuracy
    with columns[2]:
        st.subheader("Accuracy")
        fig3, ax3 = plt.subplots()
        ax3.bar(["CatBoost", "Random Forest"], [evaluate_CB_all[2], evaluate_RF[2]], color=['blue', 'orange'])
        ax3.set_ylabel("Accuracy (%)")
        st.pyplot(fig3)

    # -------- User input
    number_select=st.number_input(label="### *Select the number of features*",min_value=3,max_value=20,step=1,value=9)
    number = feature_number if not number_select else number_select
    key_base_feature_list = key_base_feature_names[-number:].tolist()

    # Add image 
    image_url = "https://cdn.dribbble.com/users/41854/screenshots/2614190/media/94cbc0074b44f6b76a8c1fc0cdabbe12.gif"
    # Center the image using columns
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratios for centering
    with col2:
        st.image(image_url, caption="Thinking......", use_container_width=True)
    
    # -------- Dynamically compute accuracy of the selected number of features
    best_model_CB_dy, evaluate_CB_dy = CatBoostRegressor_model(key_base_feature_list, cleaned_df)

    # -------- Draw the result of different number of features
    st.markdown("#### List of Your Selected Features")
    st.markdown(
        f"""
        <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
            <ul>
                {"".join([f"<li>{feature}</li>" for feature in key_base_feature_names[-number:]])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    # Display bar of accuracy for different number of features
    x_label = ["All features", f"{number} features"]
    y_label = [evaluate_CB_all[2], evaluate_CB_dy[2]]  # Replace with actual evaluation data

    # Create a DataFrame for the bar chart
    data = pd.DataFrame({
        "The number of features": x_label,
        "Accuracy (Define tolerance as 10% of the true value)": y_label
    })

    # Streamlit bar chart
    st.bar_chart(data.set_index("The number of features"))

    # Display the accuracy values
    with st.container():
        st.markdown("### Accuracy Values")
        for feature, accuracy in zip(x_label, y_label):
            st.write(f"- **{feature}**: {accuracy:.2f}%")
