#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[30]:


df=pd.read_csv('creditcard.csv')


# In[31]:


df.head()


# In[32]:


from sklearn.preprocessing import StandardScaler 


# In[33]:


df['ScaledTime']=StandardScaler().fit_transform(df['Time'].values.reshape(-1,1))
df['ScaledAmount']=StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))


# In[34]:


df.head()


# In[35]:


x=df.iloc[:,:-3]
y=df.iloc[:,-3]


# In[36]:


LogReg=LogisticRegression()
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)


# In[37]:


LogReg.fit(x_train,y_train)


# In[38]:


y_pred=LogReg.predict(x_test)


# In[40]:


print(accuracy_score(y_pred,y_test))


# In[43]:


from sklearn.metrics import confusion_matrix
Genuine =df[df['Class']== 0]
Fraud=df[df['Class']== 1] 


# In[48]:


LABELS = ['Genuine', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(22, 22)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 


# In[ ]:




