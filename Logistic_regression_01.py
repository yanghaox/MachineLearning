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

#sns.countplot(x = 'y', data=data, palette='hls')
#plt.show()
#plt.savefig('count_plot')

data.groupby('y').mean()
data.groupby('education').mean()
data.groupby('marital').mean()
data.groupby('job').mean()
#print(data.groupby('y').mean())
#print(data.groupby('education').mean())
#print(data.groupby('marital').mean())
#print(data.groupby('job').mean())

'''
#analysis the data set


pd.crosstab(data.job, data.y,).plot(kind='bar')
plt.title('Purchase Frequency for job title')
plt.xlabel('Job')
plt.ylabel('frequecy of purchase')
plt.savefig('purchase_fre_job')

pd.crosstab(data.marital, data.y).plot(kind='bar')
plt.title('Stacked bar chart of marital status vs pruchase')
plt.xlabel('marital status')
plt.ylabel('proportion of customers')
plt.savefig('marital_vs_pur_stack')

table = pd.crosstab(data.education, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked bar chart of educaton vs purchase')
plt.xlabel('education')
plt.ylabel('proportion of customers')
plt.savefig('ed_vs_pur_stack')

pd.crosstab(data.month, data.y).plot(kind='bar', stacked=False)
plt.title('purchase frequency for month')
plt.xlabel('month')
plt.ylabel('frequency of purchase')
plt.savefig('pur_fre_month_bar')


data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_pout_bar')
plt.show()
'''
#create dummy variables

cat_vars = 

