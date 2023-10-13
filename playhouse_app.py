import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# Set the page layout to wide
st.set_page_config(
    page_title="Playhouse Social Media Analytics",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)

st.title("💬 Playhouse Social Media Analytics")
st.markdown(" This is a Streamlit app to explore social media analytics. ")

# Sample DataFrame
url = 'https://raw.githubusercontent.com/Williamz4lyf/playhouse/71e82ae54addd56c12e1edb9cf2374ad8d9805cf/playhouse.csv'
data = pd.read_csv(url)
data = data.rename(columns={'Unnamed: 0': 'date'})
data['date'] = pd.to_datetime(data.date)
gen_metrics = [
    'impressions', 'reach', 'engagements',
    'engagement_rate_per_impression', 'engagement_rate_per_reach',
    'reactions', 'likes', 'comments', 'shares', 'post_link_clicks',
    'post_clicks_all', 'video_views',
]
cat_cols = ['network', 'content_type', 'sent_by']
data = data[['date'] + gen_metrics + cat_cols]
data = data.set_index('date')

# Metrics Header - Calculate KPIs based on filters
st.write("Select a specific month and year:")
selected_year = st.slider("Year", data.index.year.min(), data.index.year.max())
selected_month = st.slider("Month", 1, 12)

# Filter data based on the selected year and month
filtered_data = data[(data.index.year == selected_year) & (data.index.month == selected_month)]
kpi_features = ['impressions', 'engagements', 'reach', 'reactions', 'shares']
kpis = filtered_data[kpi_features].mean()
previous_month_data = data[kpi_features].resample('M').mean().shift(1)
delta_kpis = kpis - previous_month_data.mean()

kpi_names = ["Avg Impressions", "Avg Engagements", "Avg Reach", "Avg Reactions", "Avg Shares"]
for i, (col, (kpi_name, kpi_value, delta_kpi)) in enumerate(zip(st.columns(5), zip(kpi_names, kpis, delta_kpis))):
    col.metric(label=kpi_name, value=f'{kpi_value:,.2f}', delta=f'{delta_kpi:.1f}', delta_color='normal')

# Add some vertical spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Organize the dashboard into rows and columns
col1, col2 = st.columns(2)
# Add space between columns using HTML
col1.markdown('<div style="margin: 10px;"></div>', unsafe_allow_html=True)
# First Row - Two Columns
with col1:
    # Display the DataFrame
    st.subheader("Sample Dataset")
    st.write("Sample Data:", data.sample(10))

with col2:
    # Scatter Plot with Regression Line
    st.subheader("Metrics Compared with Regression Line")

    # Select two features for the scatter plot
    selected_x_feature = st.selectbox("Select X-axis Feature", data.select_dtypes(include='number').columns)
    selected_y_feature = st.selectbox("Select Y-axis Feature", data.select_dtypes(include='number').columns)

    # Create a scatter plot with regression line
    scatter_plot = alt.Chart(data.reset_index()).mark_circle().encode(
        x=selected_x_feature,
        y=selected_y_feature,
        color='network'
    )

    regression_line = scatter_plot.transform_regression(
        on=selected_x_feature,
        regression=selected_y_feature,
    ).mark_line().transform_fold(
        ["reg-line"],
        as_=["Regression", "y"]
    ).encode(alt.Color("Regression:N"))

    combined_chart = (scatter_plot + regression_line).properties(
        width=700,
        height=400
    )

    st.altair_chart(combined_chart, use_container_width=True)

# Add some vertical spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Second Row - Line Chart
# Select multiple numeric features to plot
st.subheader("Trends in Social Media Metrics")
selected_numeric_features = st.multiselect("Select Numeric Features", data.select_dtypes(include='number').columns)

if selected_numeric_features:
    filtered_data = data[selected_numeric_features]
    chart = alt.Chart(filtered_data.reset_index()).mark_line().encode(
        x=alt.X('yearmonth(date):T', title='Date'),
        y=alt.Y('mean(value):Q', title='Value'),
        color=alt.Color('variable:N', title='Metric')
    ).properties(
        width=700,  # Adjust the width as needed
        height=400
    ).transform_fold(
        selected_numeric_features, as_=['variable', 'value']
    )
    st.altair_chart(chart, use_container_width=True)

else:
    st.write("Select one or more numeric features to plot on the line chart.")

# Add some vertical spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Last Row - Two Columns
col3, col4 = st.columns(2)
# Add space between columns using HTML
col1.markdown('<div style="margin: 10px;"></div>', unsafe_allow_html=True)

with col3:
    # Bar Plot
    st.subheader("Categories Distributed by Metrics")
    # Select a categorical and a numeric feature for the bar plot
    selected_category_feature = st.selectbox("Select a Categorical Feature",
                                             data.select_dtypes(include='object').columns)
    selected_numeric_feature = st.selectbox("Select a Numeric Feature", data.select_dtypes(include='number').columns)
    bar_plot = alt.Chart(data.reset_index()).mark_bar().encode(
        x=selected_category_feature,
        y=selected_numeric_feature,
        color='network'
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(bar_plot, use_container_width=True)

with col4:
    # Pie Chart
    st.subheader("Category Distributions")
    selected_category_feature_pie = st.selectbox("Select Categorical Feature for Pie Chart", data.columns[-3:])
    pie_chart_data = data[selected_category_feature_pie].value_counts()
    fig = px.pie(names=pie_chart_data.index, values=pie_chart_data.values)
    st.plotly_chart(fig, use_container_width=True)

# Add some vertical spacing
st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("Prepared by Nanke Williams")
