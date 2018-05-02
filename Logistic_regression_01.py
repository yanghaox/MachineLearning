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
#print(data.shape)
#print(list(data.columns))
#print(data.head())

data['education'].unique()
#print(data['education'].unique())
data['education']=np.where(data['education'] == 'basic.9y', 'Basic',data['education'] )
data['education']=np.where(data['education'] == 'basic.4y', 'Basic',data['education'] )
data['education']=np.where(data['education'] == 'basic.6y', 'Basic',data['education'] )
#print(data['education'].unique())

data['y'].value_counts()
#print(data['y'].value_counts())
#print(data['y'].unique())

sns.countplot(x = 'y', data=data, palette='hls')
#plt.show()
plt.savefig('count_plot')

data.groupby('y').mean()
print(data.groupby('y').mean())