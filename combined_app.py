import streamlit as st
import playhouse_app
import playhouse_app_2

st.title("Combined Streamlit Dashboard")

# Create a sidebar for navigation
app_selection = st.sidebar.selectbox("Select App", ["Data Exploration", "Data Modelling"])

if app_selection == "Data Exploration":
    exec(open("playhouse_app.py").read())
elif app_selection == "Data Modelling":
    exec(open("playhouse_app_2.py").read())

