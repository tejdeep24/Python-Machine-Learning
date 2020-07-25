
# coding: utf-8

# In[42]:


#Q1 - a
import pandas as pd
prisoners_dataset = pd.read_csv('prisoners.csv')
prisoners_dataset.head(5)
prisoners_dataset.tail(5)


# In[44]:


#Q1 - b
prisoners_dataset.describe()
#prisoners_dataset.loc[]


# In[41]:


#Q2 - a
import pandas as pd
prisoners_dataset = pd.read_csv('prisoners.csv')
prisoners_dataset['total_benefitted'] = sum([prisoners_dataset['No. of Inmates benefitted by Elementary Education'], prisoners_dataset['No. of Inmates benefitted by Adult Education'], prisoners_dataset['No. of Inmates benefitted by Higher Education'], prisoners_dataset['No. of Inmates benefitted by Computer Course']])
prisoners_dataset


# In[54]:


#Q2 - b
import pandas as pd
prisoners_dataset = pd.read_csv('prisoners.csv')
prisoners_dataset = prisoners_dataset.drop(['STATE/UT', 'YEAR'], axis= 1)
prisoners_dataset.loc['totals'] = prisoners_dataset.sum()
prisoners_dataset


# In[88]:


#Q3 - a
import pandas as pd
prisoners_dataset = pd.read_csv('prisoners.csv')
prisoners_dataset['total_benefitted'] = sum([prisoners_dataset['No. of Inmates benefitted by Elementary Education'], prisoners_dataset['No. of Inmates benefitted by Adult Education'], prisoners_dataset['No. of Inmates benefitted by Higher Education'], prisoners_dataset['No. of Inmates benefitted by Computer Course']])
prisoners_dataset

import matplotlib.pyplot as plt
plt.bar(prisoners_dataset['STATE/UT'], prisoners_dataset['total_benefitted'], label='Total Benefitted')
plt.xlabel('State')
plt.ylabel('Total Benefitted')
plt.title('State vs Total Benefitted')
plt.legend()
plt.show()


# In[94]:


#Q3 - b
import pandas as pd
prisoners_dataset = pd.read_csv('prisoners.csv')
prisoners_dataset = prisoners_dataset.drop(['STATE/UT', 'YEAR'], axis= 1)
prisoners_dataset

import matplotlib.pyplot as plt
y = ['No. of Inmates benefitted by Elementary Education','No. of Inmates benefitted by Adult Education','No. of Inmates benefitted by Higher Education','No. of Inmates benefitted by Computer Course']
x = [prisoners_dataset['No. of Inmates benefitted by Elementary Education'].sum(), prisoners_dataset['No. of Inmates benefitted by Adult Education'].sum(), prisoners_dataset['No. of Inmates benefitted by Higher Education'].sum(), prisoners_dataset['No. of Inmates benefitted by Computer Course'].sum()]
plt.pie(x, labels=y, startangle=90, shadow=True, counterclock = False, labeldistance=1.1, radius = 1.5, rotatelabels=True ,center=(0, 0), frame=None, explode=None, autopct='%.2f')
plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.5))
plt.show()

