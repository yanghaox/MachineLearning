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

cat_vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(cat_list)
    data = data1
cat_vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars = data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values
#print(data_final.columns.values)
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
'''
Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model 
and choose either the best or worst performing feature, 
setting the feature aside and then repeating the process with the rest of the features.
 This process is applied until all features in the dataset are exhausted. 
 The goal of RFE is to select features by recursively considering smaller and smaller sets of features.
'''
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import  LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y] )
print(rfe.support_)
print(rfe.ranking_)

cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no",
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed",
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"]
X=data_final[cols]
y=data_final['y']

'''
The p-values for most of the variables are smaller than 0.05, therefore,
 most of them are significant to the model.
'''
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())

'''
Logistic Regression Model Fitting
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: '
      '{:.2f}'.format(logreg.score(X_test, y_test)))

'''
Cross validation attempts to avoid overfitting while still producing a prediction for each observation dataset. 
We are using 10-fold Cross-Validation to train our Logistic Regression model.
'''

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

'''
The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.

The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
The recall is intuitively the ability of the classifier to find all the positive samples.

The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, 
where an F-beta score reaches its best value at 1 and worst score at 0.

The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important.

The support is the number of occurrences of each class in y_test.
'''

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


'''
The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. The dotted line represents the ROC curve of a purely random classifier; 
a good classifier stays as far away from that line as possible (toward the top-left corner).
'''
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()