import pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  seaborn as sns
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing


data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
print(data.head())