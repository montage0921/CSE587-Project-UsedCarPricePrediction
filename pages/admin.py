import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pymysql

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

st.write(st.session_state)
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

st.cache_resource(show_spinner="Connecting to database...")
def build_connection_with_database():
    conn=connect_to_database()
    cursor=conn.cursor
    st.write("Connect to database successfully!")
    return {conn,cursor}


if st.session_state["authentication_status"]:
    st.write("Welcome to Admin Page")
    authenticator.logout('Logout from admin','sidebar',key='unique_key')
    # --------- Display Admin Page UI ---------------\
    conn,cursor=build_connection_with_database()
    st.write(cursor)

elif st.session_state["authentication_status"]==False:
    st.sidebar.warning("Please enter correct username/password")
elif st.session_state["authentication_status"]==None:
    st.sidebar.warning("Please enter admin's username/password")
