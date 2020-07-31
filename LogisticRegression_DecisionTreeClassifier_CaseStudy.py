
# coding: utf-8

# In[1]:


#Q1
#Logistic Regression - will pay (0), will not pay (1). Considering values to be either 0 or 1
from sklearn.model_selection import cross_val_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
dataset_loan = pd.read_csv("loan_borowwer_data.csv")
dataset_loan
#now split our data into train and test
train, test = train_test_split(dataset_loan, test_size = 0.3)# in this our main data is splitted into train and test
# we can check their dimension
print(train.shape)
print(test.shape)


# In[2]:


#correlation
corr = dataset_loan.corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True,
            cmap= 'coolwarm')
plt.show()


# In[4]:


prediction_var = ['credit.policy', 'log.annual.inc', 'dti', 'days.with.cr.line', 'revol.bal', 'revol.util', 'fico', 'pub.rec']
train_X = train[prediction_var]# taking the training data input
train_y = train['not.fully.paid']# This is output of our training data
# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_y =test['not.fully.paid']   #output value of test dat
logistic = LogisticRegression()
logistic.fit(train_X,train_y)
temp=logistic.predict(test_X)
print(metrics.accuracy_score(temp,test_y)) # to check the accuracy

