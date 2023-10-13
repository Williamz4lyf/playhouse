import streamlit as st
import playhouse_app
import playhouse_app_2

st.title("Combined Streamlit Dashboard")

# Create a sidebar for navigation
app_selection = st.sidebar.selectbox("Select App", ["playhouse_app", "playhouse_app_2"])

if app_selection == "playhouse_app":
    exec(open("playhouse_app.py").read())
elif app_selection == "playhouse_app_2":
    exec(open("playhouse_app_2.py").read())

