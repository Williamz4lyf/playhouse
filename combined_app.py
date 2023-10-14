import streamlit as st
import playhouse_app
import playhouse_app_2

# Set the page layout to wide
st.set_page_config(
    page_title="Playhouse Social Media Analytics",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)

st.title("Playhouse Social Media Dashboard")

# Create a sidebar for navigation
app_selection = st.sidebar.selectbox("Select App", ["Data Exploration", "Data Modelling"])

if app_selection == "Data Exploration":
    exec(open("playhouse_app.py").read())
elif app_selection == "Data Modelling":
    exec(open("playhouse_app_2.py").read())

