import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import mlflow.sklearn

arr1 = [0.2, 0.3, 0.4, 0.8, 0.9]
arr2 = [0.4, 0.6, 0.8, 1.6, 1.8]
X = np.array(arr1).reshape(-1, 1)  
y = np.array(arr2)
print("X = ", X)
print("y = ", y)

lr = LinearRegression()
model = lr.fit(X, y)
y_pred = model.predict([[1]]) 
print("Prediction for X=1:", y_pred)

plt.plot(X,y)
plt.show()