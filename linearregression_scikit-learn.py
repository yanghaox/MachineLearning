import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('headbrain.csv')

X = data['Head Size(cm^3)'].values
Y = data["Brain Weight(grams)"].values
m = len(X)
X = X.reshape((m,1))

reg = LinearRegression()
reg =reg.fit(X,Y)
predict_y = reg.predict(X)

mse = mean_squared_error(Y, predict_y)
rmse = np.sqrt(mse)
r = reg.score(X,Y)

print(np.sqrt(mse))
print(r)