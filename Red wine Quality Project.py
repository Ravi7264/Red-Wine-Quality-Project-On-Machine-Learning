#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[3]:


wine=pd.read_csv('winequality-red.csv')
wine.head()


# In[4]:


wine.info()


# In[5]:


wine.describe()


# In[6]:


wine.isnull().sum()


# In[7]:


wine.groupby('quality').mean()


# # Data Analysis

# In[8]:


plt.bar(wine['quality'],wine['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[9]:


wine.plot(kind='box',subplots = True, layout= (4,4),sharex = False)


# In[10]:


wine['fixed acidity'].plot(kind ='box')


# # Histogram

# In[11]:


wine.hist(figsize=(10,10),bins=50)
plt.show()


# # Feature Selection

# In[12]:


wine.sample(5)


# In[13]:


wine['quality'].unique()


# In[14]:


#If wine quality is 7 or above then will consider as good quality wine
wine['goodquality']= [1 if x >= 7 else 0 for x in wine['quality']]
wine.head(20)


# In[15]:


# see total number of good vs bad wines samples
wine['goodquality'].value_counts()


# In[16]:


# Seperate dependent and indepedent veriables 
x = wine.drop(['quality','goodquality'], axis=1) #Made Independent veriables
y = wine['goodquality'] #Made depedent veriables


# In[17]:


x


# In[18]:


print(y)


# # Feature Importance

# In[19]:


from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(x,y)
score = classifiern.feature_importances_
print(score)


# # splitting Dataset

# In[20]:


#Trained and fit the model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)


# # Result

# In[21]:


model_res=pd.DataFrame(columns=['Model','Score'])


# # Logistics Regression

# In[22]:


from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[23]:


from sklearn.metrics import accuracy_score,confusion_matrix
#accuracy_score(y_test,y_pred)
model_res.loc[len(model_res)] = ['LogisticRegression', accuracy_score(y_test,y_pred)]
model_res


# # Using KNN :

# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[25]:


from sklearn.metrics import accuracy_score
model_res.loc[len(model_res)] = ['KNeighborsClassifier', accuracy_score(y_test,y_pred)]
model_res


# # Using SVC :

# In[26]:


from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[27]:


from sklearn.metrics import accuracy_score
print('Accuracy Score:',accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)] = ['SVC', accuracy_score(y_test,y_pred)]
model_res


# # Using Decision Tree:

# In[28]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion ='entropy',random_state=7)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[29]:


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)] = ['DecisionTreeClassifier',accuracy_score(y_test,y_pred)]
model_res


# # Using GaussianNB

# In[30]:


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(x_train,y_train)
y_pred = model3.predict(x_test)


# In[31]:


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)] = ['GaussianNB',accuracy_score(y_test,y_pred)]
model_res


# # Using Random Forest:

# In[32]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)


# In[33]:


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
model_res.loc[len(model_res)] = ['RandomForestClassifier',accuracy_score(y_test,y_pred)]
model_res


# In[34]:


model_res = model_res.sort_values(by='Score', ascending=False)
model_res


# In[ ]:





# In[ ]:




