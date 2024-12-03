import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date,time
import numpy as np
from search_widgets_render import *
from search_result_logic import *
from edit_logic import *
from delete_logic import *
from add_logic import *

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

# @st.cache_resource(show_spinner="Connecting to database...")
def build_connection_with_database():
    conn=connect_to_database()
    cursor=conn.cursor()
    st.write("Connect to database successfully!")
    return (conn,cursor)


# ------------------  General Data -----------------------------

def extract_all_data(_conn):
    query="SELECT * from used_cars"
    df=pd.read_sql(query,conn)
    return df


def get_general_info(_cursor):
    query_total_data="SELECT COUNT(*) AS total_data FROM used_cars";
    query_total_brands="SELECT COUNT(DISTINCT make) AS unique_brands FROM used_cars;"


    cursor.execute(query_total_data)
    total_data=cursor.fetchone()[0]
    print(total_data)

    cursor.execute(query_total_brands)
    total_brands=cursor.fetchone()[0]

    general_info=f"Our dataset currently contains information on {total_data} cars from {total_brands} different brands"
    return general_info


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
    authenticator.logout('Logout from admin', 'sidebar', key='unique_key')

    # --------- Display Admin Page UI ---------------\
    conn, cursor = build_connection_with_database()

    if "df" not in st.session_state:
        st.session_state["df"] = extract_all_data(conn)
    if "general_info" not in st.session_state:
        st.session_state["general_info"] = get_general_info(cursor)

    # General Info and Graph Container
    general_info_container = st.container()

    def refresh_general_info():
        """Refreshes general info and graph dynamically."""
        st.session_state["df"] = extract_all_data(conn)
        st.session_state["general_info"] = get_general_info(cursor)
        df = st.session_state["df"]
        general_info = st.session_state["general_info"]
        general_info_container.empty()  # Clear the container
        with general_info_container:  # Re-render inside the same container
            with st.expander(general_info, expanded=False):
                st.write(display_brand_bar_graph(df))
                st.write(df)


    tab1, tab2, tab3, tab4 = st.tabs(["Find", "Edit", "Delete", "Add"])

    with tab1:
        st.subheader("🔍Search A Record")
        select_all = st.radio("Search from all cars?", options=["Yes", "No"], horizontal=True)
        if select_all == "No":
            df = st.session_state["df"]
            display_search_UI(df)
        else:
            df = st.session_state["df"]
            display_search_UI_for_all(df)

    with tab2:
        st.subheader("✏️Edit A Record")
        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = False

        edit_btn, id = edit_widget()
        if edit_btn:
            st.session_state.edit_mode = True
            st.session_state.edit_id = id

        if st.session_state.edit_mode:
            res = get_by_id(st.session_state.edit_id, cursor)
            if res:
                render_update_widget(conn, cursor, res)
            else:
                st.warning(f"No record found with ID {st.session_state.edit_id}")

    with tab3:
        st.subheader("Delete A Record")
        del_btn, id = delete_widget()
        if del_btn:
            try:
               
                res = get_by_id(id, cursor)
                if not res:
                    st.error(f"Record with ID {id} not found.")
                else:
                   
                    delete_byId(id, conn, cursor)
                    st.success(f"Record with ID {id} deleted successfully!")

                    refresh_general_info()
            except Exception as e:
                st.error(f"Failed to delete record: {e}")
    
    with tab4:
        st.subheader("⬆️Add A New Record")
        render_add_widget(conn, cursor)
         
        if st.session_state.get("record_added", False):  
            refresh_general_info()  
            st.session_state["record_added"] = False  


elif st.session_state["authentication_status"] == False:
    st.sidebar.warning("Please enter correct username/password")
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning("Please enter admin's username/password")



