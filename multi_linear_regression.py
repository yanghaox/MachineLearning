# multiple linear regression
# y = b0 + b1 * x1 + b2 * x2.....

#Step 1
#Initialize values β0, β1,…, βn with some value. In this case we will initialize with 0.

#Step 2
#Iteratively update, βj:=βj−α∂∂βjJ(β)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('student.csv')
#print(data)
data.head()

math = data['Math'].values
reading = data['Reading'].values
writing = data['Writing'].values

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math,reading,writing)
#plt.show()

'''now we are generating x, y , beta
'''

m = len(math)
x0 = np.ones(m)
#give the new array, similar with a[m], and also default number with 1;
# print(x0)

#create array, and self translate to matrix
x = np.array([x0, math,reading]).T
#initial coefficients
beta = np.array([0 , 0, 0])
y = np.array(writing)
alpha = 0.0001

'''define cost function, the cost function is for checking how far from 'big mistake' to mimum 
    and then using Batch Gradient Descent to reduce cost
'''
def cost_function(x,y, beta):
    m = len(y)
    J = np.sum((x.dot(beta) - y) ** 2) /(2 * m)
    return J
cost = cost_function(x,y,beta)
#print(cost) #2470.11

def gradient_descent(x,y,beta,alpha,iterations):
    cost_hitory = [0] * iterations
    m = len(y)

    for iteration in range(iterations):
        hypothesis = x.dot(beta)
        loss = hypothesis - y # defference between from hypothesis and actual y
        gradient = x.T.dot(loss) / m #gradient calculation
        beta = beta - alpha * gradient #changing values of beta using gradient
        cost = cost_function(x, y, beta)
        cost_hitory[iteration]  = cost
    return beta, cost_hitory
newBeta, cost_hitory = gradient_descent(x,y, beta, alpha, 1000000)
print(newBeta)
print(cost_hitory[-1])

