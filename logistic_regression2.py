'''
the dataset from the tatannic

'''



import numpy as np
import pandas as pd
import  seaborn as sb
import  matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import  rcParams
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

#matplotlib inline
rcParams['figure.figsize'] = 10, 8
