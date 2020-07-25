
# coding: utf-8

# In[38]:


#Q1
import pandas as pd
import matplotlib.pyplot as plt
cereal_dataset = pd.read_csv('cereal.csv')
cereal_dataset


# In[41]:


#Q2
import pandas as pd
import matplotlib.pyplot as plt
cereal_dataset = pd.read_csv('cereal.csv')
cereal_dataset["Manufacturers"] = cereal_dataset["mfr"].map({'N': 'Nabisco','Q': 'Quaker Oats','K': 'Kelloggs','R': 'Raslston Purina','G': 'General Mills' ,'P' :'Post' ,'A':'American Home Foods Products'})
cereal_dataset = cereal_dataset.groupby(['Manufacturers'])['name'].count().reset_index(name = 'cereals_count')
cereal_dataset
import matplotlib.pyplot as plt
plt.bar(cereal_dataset['Manufacturers'], cereal_dataset['cereals_count'], label='Cereals Count', color='red')
plt.xlabel('Cereals Count')
plt.ylabel('Manufacturers')
plt.title('Cereals Count vs Manufacturers')
plt.legend()
plt.show()


# In[59]:


#Q3
import pandas as pd
cereal_dataset = pd.read_csv('cereal.csv').select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
x = cereal_dataset.drop('rating', axis = 1)
y = cereal_dataset['rating']
from sklearn.cross_validation import train_test_split
#testing data size is of 25% of entire data
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size = 0.25, random_state =5)
from sklearn.linear_model import LinearRegression
#fitting our model to train and test
lm = LinearRegression()
model = lm.fit(x_train,y_train)
y_pred = lm.predict(x_test)
from sklearn import metrics
print(metrics.mean_squared_error(y_test, y_pred))

