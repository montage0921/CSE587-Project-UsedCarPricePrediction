import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth

# load yaml as file
with open('config.yaml') as file:
    config=yaml.load(file,Loader=SafeLoader)

# Create authenticator object
authenticator=stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render Login Widget
try:
    authenticator.login(location='main',max_concurrent_users=10,max_login_attempts=3)
except Exception as e:
    st.error(e)

# Create Guest Login Button
try:
    authenticator.experimental_guest_login('Login with Google',
                                           provider='google',
                                           oauth2=config['oauth2'])
except Exception as e:
    st.error(e)

# Authenticating Users and User Logout
if st.session_state['authentication_status']:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')

# Reset Password Widget
if st.session_state['authentication_status']:
    try:
        if authenticator.reset_password(st.session_state['username']):
            st.success('Password modified successfully')
    except Exception as e:
        st.error(e)

# New User Registration Widget
try:
    email_of_registered_user, \
    username_of_registered_user, \
    name_of_registered_user = authenticator.register_user(pre_authorized=config['pre-authorized']['emails'])
    if email_of_registered_user:
        st.success('User registered successfully')
except Exception as e:
    st.error(e)

# Forgot Password Widget
try:
    username_of_forgotten_password, \
    email_of_forgotten_password, \
    new_random_password = authenticator.forgot_password()
    if username_of_forgotten_password:
        st.success('New password to be sent securely')
        # The developer should securely transfer the new password to the user.
    elif username_of_forgotten_password == False:
        st.error('Username not found')
except Exception as e:
    st.error(e)

# Forgot Password Widget
try:
    username_of_forgotten_username, \
    email_of_forgotten_username = authenticator.forgot_username()
    if username_of_forgotten_username:
        st.success('Username to be sent securely')
        # The developer should securely transfer the username to the user.
    elif username_of_forgotten_username == False:
        st.error('Email not found')
except Exception as e:
    st.error(e)

# Update Config File
#  the config file should be re-saved whenever the contents are modified
    yaml.dump(config, file, default_flow_style=False)

hashed_password = stauth.Hasher.hash_passwords(config["credentials"][0])
st.write(hashed_password)