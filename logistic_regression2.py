'''
the dataset from the tatannic

'''



import numpy as np
import pandas as pd
import  seaborn as sb
import  matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

titanic = pd.read_csv('train.csv')
a = titanic.head()

# print(a)
#sb.countplot(x='Survived', data=titanic, palette='hls')

'''
isnull() is easy to check for missing values
'''
#miss_value = titanic.isnull().sum()
#print(miss_value)
#titanic.info()
'''
so there are only 891 rows in the titanic data frame. Cabin is almost all missing values, so we can drop that variable completely, but what about age? Age seems like a relevant predictor for survival right? We'd want to keep the variables, but it has 177 missing values. Yikes!! We are going to need to find a way to approximate for those missing values!
'''
titanic_drop = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],1)

#print(titanic_drop.head())

#sb.boxplot(x='Pclass', y='Age', data=titanic_drop, palette='hls')


def age_approx(cols):
	Age = cols[0]
	Pclass = cols[1]

	if pd.isnull(Age):
		if Pclass == 1:
			return 37
		elif Pclass == 2:
			return 29
		else:
			return 24
	else:
		return Age

titanic_drop['Age'] = titanic_drop[['Age', 'Pclass']].apply(age_approx, axis = 1)
Age_miss = titanic_drop.isnull().sum()
print(Age_miss)

titanic_drop.dropna(inplace=True)
Embarked_miss = titanic_drop.isnull().sum()
print(Embarked_miss)

'''
Converting categorical variables to a dummy indicators
'''
gender = pd.get_dummies(titanic_drop['Sex'], drop_first=True)
print(gender.head())

embark_location = pd.get_dummies(titanic_drop['Embarked'], drop_first=True)
print(embark_location.head())

#print(titanic_drop.head())
titanic_drop = titanic_drop.drop(['Sex','Embarked'],1)
print(titanic_drop.head())

titanic_dmy = pd.concat([titanic_drop, gender, embark_location], axis=1)
print(titanic_dmy.head())









plt.show()
