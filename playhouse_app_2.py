import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

st.title("💬 Playhouse Social Media Analytics: Regression Modelling")

# Load the data
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
data = data[['date'] + gen_metrics + cat_cols].set_index('date')
data = data.assign(
    year=data.index.year, month=data.index.month,
    day=data.index.day, hour=data.index.hour,
    minute=data.index.minute
)
st.markdown("<br><br>", unsafe_allow_html=True)
# Sample Dataset
st.subheader("Sample Dataset")
st.write(data.sample(5))
st.markdown("<br><br>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header('Regression Model Configuration')
target_feature = st.sidebar.selectbox('Select the target feature:', data.select_dtypes(include='number').columns)
categorical_features = st.sidebar.multiselect('Select categorical features:', data.select_dtypes(include='object').columns, default=['network', 'sent_by'])
numeric_features = st.sidebar.multiselect('Select numeric features:', data.select_dtypes(include='number').columns, default=['year', 'engagements', 'reach', 'reactions'])

# Filter the data to include only selected categorical and numeric features
selected_features = categorical_features + numeric_features
filtered_data = data[selected_features + [target_feature]].reset_index().drop(columns='date')

# Train-test split based on year
st.sidebar.header('Train-Test Split')
filtered_data = pd.get_dummies(filtered_data, columns=categorical_features, drop_first=True)
train_years = st.sidebar.slider('Select the training years:', min_value=filtered_data.year.min(), max_value=2021, value=(filtered_data.year.min(), 2021))
test_years = st.sidebar.slider('Select the testing years:', min_value=2015, max_value=filtered_data.year.max(), value=(2022, filtered_data.year.max()))

train_data = filtered_data[(filtered_data['year'] >= train_years[0]) & (filtered_data['year'] <= train_years[1])]
test_data = filtered_data[filtered_data['year'] >= test_years[0]]

# Train a Gradient Boosting Regressor model
X_train = train_data.drop(columns=[target_feature])
y_train = train_data[target_feature]

X_test = test_data.drop(columns=[target_feature])
y_test = test_data[target_feature]

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions for both models
y_pred_gb = gb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# Calculate R2 score and RMSE for both models
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# Display model predictions, R2 score, and RMSE for both models
st.subheader('Model Predictions')
# Create a layout with two columns
col1, col2 = st.columns(2)
# Display RMSE and R2 for the Gradient Boosting model in the first column
with col1:
    st.subheader('Gradient Boosting')
    st.metric("R2 Score", f'{r2_gb:,.3f}')
    st.metric("RMSE", f'{rmse_gb:,.3f}')

# Display RMSE and R2 for the Linear Regression model in the second column
with col2:
    st.subheader('Linear Regression')
    st.metric("R2 Score", f'{r2_lr:,.3f}')
    st.metric("RMSE", f'{rmse_lr:,.3f}')

st.markdown("<br><br>", unsafe_allow_html=True)

# Generate a regression model summary using the entire dataset with feature importances
# Fit an OLS regression model and display summary
X = data.drop(columns=[target_feature])
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
y = data[target_feature]
# Filter features with importance greater than 0.05
selected_features = X_train.columns[gb_model.feature_importances_ > 0]
X = X[selected_features]
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()

st.subheader('Regression Model Summary - with Important Features')
st.text(ols_model.summary())

st.markdown("<br><br>", unsafe_allow_html=True)

# Plot regression diagnostics
st.subheader('Regression Diagnostics Plots')
fig, axes = plt.subplots(2, 2, figsize=(22, 15))

# Plot 1: Residuals vs. Fitted
residuals = ols_model.resid
sns.regplot(x=ols_model.fittedvalues, y=residuals, lowess=True, ax=axes.ravel()[0])
axes.ravel()[0].set_xlabel('Fitted Values')
axes.ravel()[0].set_ylabel('Residuals')
axes.ravel()[0].set_title('Residuals vs. Fitted')

# Plot 2: Residuals vs. Residuals
sns.residplot(x=ols_model.fittedvalues, y=y, lowess=True, ax=axes.ravel()[1])
axes.ravel()[1].set_xlabel('Fitted Values')
axes.ravel()[1].set_ylabel('Residuals')
axes.ravel()[1].set_title('Residuals vs. Fitted Values')

# Plot 3: Quantile-Quantile (QQ) Plot
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True, ax=axes.ravel()[2])
axes.ravel()[2].set_title('QQ Plot')

# Plot 4: Component-Component (CC) Plot
feature_importances = gb_model.feature_importances_
most_important_index = feature_importances.argmax()
sm.graphics.plot_ccpr(ols_model, X_train.columns[most_important_index], ax=axes.ravel()[3])
axes.ravel()[3].set_xlabel('Fitted Values')
axes.ravel()[3].set_ylabel('Partial Residuals')
axes.ravel()[3].set_title('Component-Component (CC) Plot')

st.pyplot(fig)

st.write('Prepared by Nanke Williams')