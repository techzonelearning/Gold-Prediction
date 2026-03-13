import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/datasets/gold-prices/refs/heads/main/data/monthly.csv"
)

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df = df.drop("Date", axis=1)

X = df[["Year", "Month"]]
y = df["Price"]

# data train
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


st.title("Gold Price Predication - Machine Learning")
st.subheader("Preview")
st.write(df.head(10))


st.sidebar.subheader("Algorithms")
algo = st.sidebar.selectbox(
    "select algorithms", ["Linear Regression", "Decision Tree", "Random Forest"]
)

if algo == "Linear Regression":
    model = LinearRegression()
elif algo == "Decision Tree":
    model = DecisionTreeRegressor()
else:
    model = RandomForestRegressor(n_estimators=100)

model.fit(x_train, y_train)
pred = model.predict(x_test)


mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error", f"{mae:.2f}")
col2.metric("Mean Squared Error", f"{mse:.2f}")
col1.metric("R2 Score", f"{r2:.2f}")
col2.metric("Root Mean Squared Error", f"{rmse:.2f}")


fig = plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(pred, label="Prediction")
plt.legend()
st.pyplot(fig)

st.sidebar.header("Future Gold Prediction")
year = st.sidebar.number_input("Enter your Year", 2000, 2100, 2026)
month = st.sidebar.slider("Select your month", 1, 12, 1)

if st.sidebar.button("Predict"):
    st.header("Gold Prediction")
    pred = model.predict([[year, month]])
    st.success(f"Predict Price {pred[0]:.2f}")
