from config import (PATH, 
                    featureColumns,
                    saveModelPath, 
                    REGRESSIONMODELINFO)
from utils import getData
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie
import pickle
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Regression Project App", page_icon=":bar_chart:", layout="wide"
)

# Title
st.title("Regression Project App")
st.header("Regression Model Info")
st.write(REGRESSIONMODELINFO)

# Sidebar
st.sidebar.header("User Input")
input_features = []

# Example feature inputs (you can replace these with actual input fields)
for i in range(4):
    input_feature = st.sidebar.number_input(f"Feature {i+1}", value=0.0)
    input_features.append(input_feature)

# Load your regression model (replace this with your model loading code)
with open(saveModelPath + "LinearRegrssion.pkl", "rb") as model:
    linearModel = pickle.load(model)

# Predict target based on user inputs
if st.sidebar.button("Predict"):
    input_features_array = np.array(input_features).reshape(1, -1)
    prediction = linearModel.predict(input_features_array)[0][0]
    st.sidebar.success(f"Predicted Target: {prediction:.2f}")

# Sample data for visualization (replace this with your data)
df = getData(PATH)

# Main content
st.header("Data Visualization")
st.write("Sample data for visualization:")
st.dataframe(df)
color = st.sidebar.color_picker("Choose color", "#1f77b4")
size = st.sidebar.slider("Marker size", 10, 100, 30)
features = [
    'select here',
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership",
]
selected_option = st.selectbox("Feature", features)
# user_text = st.text_input("Enter feature name")
# print(selected_option)
if selected_option == "select here":
    pass
else:
    x = np.array(df[selected_option])
    y = np.array(df["Yearly Amount Spent"])
    st.subheader("Scatter Plot")
    plt.figure(figsize=(3, 3))
    plt.scatter(x, y, color=color, s=size)
    plt.xlabel(selected_option)
    plt.ylabel("Target")
    plt.title("Scatter Plot")
    st.pyplot(plt)


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit by Your Name")
