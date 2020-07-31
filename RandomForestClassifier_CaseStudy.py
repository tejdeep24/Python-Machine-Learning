
# coding: utf-8

# In[109]:


#Q1
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib
import seaborn as sns # this is used for the plot the graph
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
dataset_horse = pd.read_csv('horse.csv')
#dataset_horse.isnull()


# In[110]:


#Q2
dataset_horse = pd.get_dummies(data=dataset_horse)
dataset_horse


# In[111]:


#Q3
from sklearn.preprocessing import Imputer
# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)

# Train the imputor on the df dataset
imputer = imputer.fit(dataset_horse)

# Apply the imputer to the df dataset
imputed_df = imputer.transform(dataset_horse.values)
imputed_df


# In[112]:


#Q4 - Decision Tree Classifier
dataset_horse = dataset_horse.fillna(value = 0, axis = 1)
pred_columns = dataset_horse[:]
pred_columns.drop(['hospital_number'],axis=1,inplace=True)
prediction_var = pred_columns.columns
train, test = train_test_split(dataset_horse, test_size = 0.3)# in this our main data is splitted into train and test
train_X = train[prediction_var]# taking the training data input
train_y= train['outcome_lived']
test_X= test[prediction_var] # taking test data inputs
test_y =test['outcome_lived']
model = tree.DecisionTreeClassifier()
model.fit(train_X,train_y)# now fit our model for traiing data
prediction=model.predict(test_X)# predict for the test data
print(metrics.accuracy_score(prediction,test_y)) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values


# In[113]:


#Q5 - Random Tree Classifier
#RandomForest classifier
model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train_X,train_y)# now fit our model for traiing data
prediction=model.predict(test_X)# predict for the test data
#prediction will contain the predicted value by our model predicted values of diagnosis column for test inputs
print(metrics.accuracy_score(prediction,test_y)) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values

