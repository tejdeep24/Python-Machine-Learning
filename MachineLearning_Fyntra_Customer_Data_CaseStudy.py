
# coding: utf-8

# In[41]:


#Q1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
customer_dataset = pd.read_csv('FyntraCustomerData.csv')
customer_dataset
#sns.jointplot(x='Time_on_Website',y='Yearly_Amount_Spent',data=customer_dataset)


# In[9]:


#Q2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
customer_dataset = pd.read_csv('FyntraCustomerData.csv')
sns.jointplot(x='Time_on_App',y='Yearly_Amount_Spent',data=customer_dataset)


# In[16]:


#Q3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
customer_dataset = pd.read_csv('FyntraCustomerData.csv')
sns.pairplot(customer_dataset, height=2.5)


# In[17]:


#Q4
sns.regplot(x='Length_of_Membership', y='Yearly_Amount_Spent', data=customer_dataset)


# In[37]:


#Q5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
customer_dataset = pd.read_csv('FyntraCustomerData.csv').select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
x = customer_dataset.drop('Yearly_Amount_Spent', axis = 1)
y = customer_dataset['Yearly_Amount_Spent']
from sklearn.cross_validation import train_test_split
#testing data size is of 25% of entire data
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size = 0.25, random_state =85)
from sklearn.linear_model import LinearRegression
#fitting our model to train and test
lm = LinearRegression()
model = lm.fit(x_train,y_train)
y_pred = lm.predict(x_test)
from sklearn import metrics
mean_square_error = metrics.mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mean_square_error)
#if random_state is not mentioned every time when code is excuted different train and test values will be generated


# In[30]:


#Q6
plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[38]:


#Q7
from math import sqrt
rmse = sqrt(mean_square_error)
print('Root Mean Squared Error:', rmse)


# In[39]:


#Q8
lm.coef_

#Time on website

