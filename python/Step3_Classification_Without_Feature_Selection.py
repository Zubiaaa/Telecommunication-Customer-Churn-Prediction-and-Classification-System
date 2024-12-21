#!/usr/bin/env python
# coding: utf-8

# In[1]:


# If you do not already have this package install it by uncommenting the below line:
#pip install scikit-learn-intelex


# In[2]:


# Below 2 lines accelerate sklearn algorithms while using the familiar scikit-learn package and getting the same results
from sklearnex import patch_sklearn
patch_sklearn()


# In[3]:


# import relevant libraries:
import random
import operator
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score


# In[4]:


# importing Dataset
dataset = pd.read_csv('Customer_Churn_Final_Dataset.csv')
dataset.head()


# In[5]:


# Separate input features and target:
    
target_variable = dataset["Churn"]
independent_features = dataset.drop(columns="Churn")
independent_features


# In[6]:


X = independent_features
y = target_variable


# In[7]:


# Trying a validation technique, K-fold Cross-validation
kfold = KFold(n_splits=10)
# Building Gaussion Naive Bayes model
model = GaussianNB()
# Using f1_score as it is an imbalanced dataset
score_kfold = cross_val_score(model, X, y, scoring='f1_macro', cv=kfold)
print("F1 Score: %.2f%%" % (score_kfold.mean()*100.0)) 


# In[8]:


# evaluate model
scores = cross_val_score(model, X, y, scoring='precision', cv=kfold)
# summarize performance
print("Precision: %.3f%%" % (scores.mean()*100.0))


# In[9]:


# evaluate model
scores = cross_val_score(model, X, y, scoring='recall', cv=kfold)
# summarize performance
print("Recall: %.3f%%" % (scores.mean()*100.0))

