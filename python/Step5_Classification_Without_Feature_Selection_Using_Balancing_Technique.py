#!/usr/bin/env python
# coding: utf-8

# In[1]:


# If you do not already have this package install it by uncommenting the below line:
#pip install scikit-learn-intelex


# In[2]:


# pip install --upgrade scikit-learn


# In[3]:


from sklearnex import patch_sklearn
patch_sklearn()


# In[4]:


# import relevant libraries:
import random
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold


# In[5]:


############## Importing Dataset ##############
dataset = pd.read_csv('Customer_Churn_Final_Dataset.csv')
dataset.head()


# In[6]:


dataset.shape


# In[7]:


# Separate input features and target:
    
target_variable = dataset["Churn"]
independent_features = dataset.drop(columns="Churn")
independent_features


# In[8]:


X = independent_features
y = target_variable


# ### Balancing the Imbalanced Data

# In[9]:


# As you can clearly see below, the dataset is quite imbalanced:


# In[10]:


# The Number of Customer's Churning is very less than the Customer's Churning (more than half a difference)
dataset["Churn"].value_counts()


# In[11]:


# Target variable distribution is showing that the dataset is imbalanced:
sns.countplot(x=dataset.Churn)


# In[12]:


# Building Gaussion Naive Bayes model
model = GaussianNB()

# define resampling
resample = SMOTEENN()
# define pipeline
pipeline = Pipeline(steps=[('r', resample), ('m', model)])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10)


# In[13]:


# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='f1_macro', cv=cv)
# summarize performance
print("F1 Score: %.3f%%" % (scores.mean()*100.0))


# In[14]:


# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='precision', cv=cv)
# summarize performance
print("Precision: %.3f%%" % (scores.mean()*100.0))


# In[15]:


# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='recall', cv=cv)
# summarize performance
print("Recall: %.3f%%" % (scores.mean()*100.0))

