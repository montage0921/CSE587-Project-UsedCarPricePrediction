import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date,time
import numpy as np
from display_logic import *
from search_logic import *

# --------------- Authentication --------------------
with open('config.yaml') as file:
    config=yaml.load(file,Loader=SafeLoader)

# create an authenticator
authenticator=stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)


# Render Login Widget
try:
    authenticator.login(location="sidebar",max_concurrent_users=4,max_login_attempts=3,fields={
        'Form name': 'Admin Login',
        'Captcha':'Captcha' 
    })
except Exception as e:
    st.error(e)

# ---------------------- Connect to Database ---------------------------

def connect_to_database():
    conn = pymysql.connect(
        host=config['credentials']['database']['host'],
        port=3306,
        user=config['credentials']['database']['user'],
        password=config['credentials']['database']['password'],
        database=config['credentials']['database']['database']
    )
    return conn

@st.cache_resource(show_spinner="Connecting to database...")
def build_connection_with_database():
    conn=connect_to_database()
    cursor=conn.cursor()
    st.write("Connect to database successfully!")
    return (conn,cursor)


# ------------------  General Data -----------------------------
@st.cache_data
def extract_all_data(_conn):
    query="SELECT * from used_cars"
    df=pd.read_sql(query,conn)
    return df

@st.cache_data
def get_general_info(_cursor):
    query_total_data="SELECT COUNT(*) AS total_data FROM used_cars";
    query_total_brands="SELECT COUNT(DISTINCT make) AS unique_brands FROM used_cars;"


    cursor.execute(query_total_data)
    total_data=cursor.fetchone()[0]

    cursor.execute(query_total_brands)
    total_brands=cursor.fetchone()[0]

    general_info=f"Our dataset currently contains information on {total_data} cars from {total_brands} different brands"
    return general_info


@st.cache_data
def display_brand_bar_graph(dataframe):

    brand_counts = dataframe['make'].value_counts()

    fig, ax = plt.subplots(figsize=(16, 8))  # Wider figure

    brand_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)

    ax.set_title('Number of Cars by Brand', fontsize=20)
    ax.set_xlabel('Car Brand', fontsize=14)
    ax.set_ylabel('Number of Cars', fontsize=14)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') 

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    return fig

# ------------------------- UI -----------------------------------------------
if st.session_state["authentication_status"]:
    st.title("⚙️Welcome to Admin Page")
    authenticator.logout('Logout from admin','sidebar',key='unique_key')
    # --------- Display Admin Page UI ---------------\
    conn,cursor=build_connection_with_database()
    df=extract_all_data(conn)

    # general info
    general_info=get_general_info(cursor)
    with st.expander(general_info):
        st.write(display_brand_bar_graph(df))
        st.write(df)

    tab1,tab2,tab3,tab4=st.tabs(["Find","Edit","Delete","Add"])

    with tab1:
        display_search_UI(df)
        




elif st.session_state["authentication_status"]==False:
    st.sidebar.warning("Please enter correct username/password")
elif st.session_state["authentication_status"]==None:
    st.sidebar.warning("Please enter admin's username/password")
