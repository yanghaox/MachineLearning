import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


data = pd.read_csv('headbrain.csv')
#show the total data amount, weight, height
# print(data.shape)
#data.head()

X = data['Head Size(cm^3)'].values
Y = data["Brain Weight(grams)"].values

#print(X, Y)
#data.head()
#
mean_x = np.mean(X)
mean_y = np.mean(Y)
#print(X, '\n')
#display the mean of head size
#  print(mean_x)

m = len(X)
# total number of values
# print(m)

denom = 0
numer = 0
#y = bo + b1*x this is linear model,
# use for loop
for i in range(m):
	numer += ((X[i] - mean_x) * (Y[i] - mean_y))
	denom += (X[i] - mean_x) * (X[i] - mean_x)

Beta1 = numer/denom
Beta0 = mean_y - (Beta1*mean_x)

#print(Beta0,Beta1)
max_x = np.max(X) + 100
min_x = np.min(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = Beta0 + Beta1*x
'''
plt.plot(x,y, color = '#58b970', label = 'regression line')
plt.scatter(X,Y,label='scatter plot')
plt.xlabel("head size")
plt.ylabel("brain wight")
plt.legend()
plt.show()
'''
#root mean squared error
# for evaluating models
#squared root of sum of all errors/ number of values

rmse = 0

for i in range(m):
	predict_y = Beta0 + Beta1* X[i]
	rmse += (Y[i] - predict_y)**2
rmse = np.sqrt(rmse/m)
print(rmse)

#R**2 become negative if the model is completely wrong
sst = 0
ssr = 0
for i in range(m):
	predict_y = Beta0 + Beta1 * X[i]
	sst += (Y[i] - mean_y)**2
	ssr += (Y[i] - predict_y) **2
R = 1 - ssr/sst
print(R)
