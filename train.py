"""This  file Trains and saves the model"""
from config import (PATH,
                     featureColumns,
                       labelColumn, 
                       saveModelPath)
from utils import getData
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = getData(PATH)
x = data[featureColumns]
y = data[labelColumn]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=100
)
model = LinearRegression()
print("Training Started")
model.fit(x_train, y_train)

print("Model Saving")
with open(saveModelPath + "LinearRegrssion.pkl", "wb") as files:
    pickle.dump(model, files)
