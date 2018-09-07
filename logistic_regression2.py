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

'''
Checking for independence between features
'''
#sb.heatmap(titanic_dmy.corr())

'''
Fare and Pclass are not independent of each other, 
so I am going to drop these.
'''

titanic_dmy_drop = titanic_dmy.drop(['Fare', 'Pclass'], 1)
print(titanic_dmy_drop.head())

titanic_dmy.info()

X = titanic_dmy.ix[:,(1,2,3,4,5,6)].values
y = titanic_dmy.ix[:,0].values
#print(y)

'''
train_data：所要划分的样本特征集

train_target：所要划分的样本结果

test_size：样本占比，如果是整数的话就是样本的数量

random_state：是随机数的种子。

随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。

随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：

种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。


'''

'''
 避免过拟合，采用交叉验证，验证集占训练集30%
 train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和test data
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
#print(y_test,'gf')
LogReg = LogisticRegression()
x = LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
cla = classification_report(y_test, y_pred)
print(confusion_matrix)
print(cla)


