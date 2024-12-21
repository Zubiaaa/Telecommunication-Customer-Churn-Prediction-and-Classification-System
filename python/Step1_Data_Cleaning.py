#!/usr/bin/env python
# coding: utf-8

# In[1]:


############## Importing Libraries ##############
import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


# Strings "?" to recognize as NA/NaN
missing_value_formats = ["?"]


# In[3]:


############## Importing Dataset ##############
dataset = pd.read_csv('cell2celltrain_Small_6k.csv', na_values = missing_value_formats)


# ## General Data Exploration

# In[4]:


dataset.head()


# In[5]:


dataset.columns


# In[6]:


dataset.shape


# In[7]:


dataset.dtypes


# In[8]:


dataset.columns.to_series().groupby(dataset.dtypes).groups


# ### Looking for Null Values

# In[9]:


# Prints information about the DataFrame
dataset.info()


# In[10]:


# Yes, there are null values in the columns marked as True below
dataset.isna().any()


# In[11]:


# How many Null values are there in each column?
dataset.isna().sum()


# In[12]:


# How many Unique values are there in each column and what are those unique values?
for i in dataset.columns:
    print(f"Unique {i}'s count: {dataset[i].nunique()}")
    print(f"{dataset[i].unique()}\n")


# In[13]:


# Does the dataset upholds any class imbalance issues?
dataset["Churn"].value_counts()


# ## Dealing with Missing Values

# In[14]:


dataset["MonthlyRevenue"].value_counts()


# In[15]:


# 0.329% of the Missing value in "MonthlyRevenue" column
# Replace the missing value with the mean 

dataset['MonthlyRevenue'] = dataset['MonthlyRevenue'].fillna(dataset['MonthlyRevenue'].mean()).round(2)

# Checking if still there is any Null value left
dataset["MonthlyRevenue"].isnull().sum()


# In[16]:


dataset["MonthlyMinutes"].value_counts()


# In[17]:


# 0.329% of the Missing value in "MonthlyMinutes" column
# Replace the missing value with the mean 

dataset['MonthlyMinutes'] = dataset['MonthlyMinutes'].fillna(dataset['MonthlyMinutes'].mean()).round(1)

# Checking if still there is any Null value left
dataset["MonthlyMinutes"].isnull().sum()


# In[18]:


dataset["TotalRecurringCharge"].value_counts()


# In[19]:


# 0.329% of the Missing value in "TotalRecurringCharge" column
# Replace the missing value with the mean 

dataset['TotalRecurringCharge'] = dataset['TotalRecurringCharge'].fillna(dataset['TotalRecurringCharge'].mean()).round(1)

# Checking if still there is any Null value left
dataset["TotalRecurringCharge"].isnull().sum()


# In[20]:


dataset["DirectorAssistedCalls"].value_counts()


# In[21]:


# 0.329% of the Missing value in "DirectorAssistedCalls" column
# Replace the missing value with the mean 

dataset['DirectorAssistedCalls'] = dataset['DirectorAssistedCalls'].fillna(dataset['DirectorAssistedCalls'].mean()).round(2)

# Checking if still there is any Null value left
dataset["DirectorAssistedCalls"].isnull().sum()


# In[22]:


dataset["OverageMinutes"].value_counts()


# In[23]:


# 0.329% of the Missing value in "OverageMinutes" column
# Replace the missing value with the mean 

dataset['OverageMinutes'] = dataset['OverageMinutes'].fillna(dataset['OverageMinutes'].mean()).round(1)

# Checking if still there is any Null value left
dataset["OverageMinutes"].isnull().sum()


# In[24]:


dataset["RoamingCalls"].value_counts()


# In[25]:


# 0.329% of the Missing value in "RoamingCalls" column
# Replace the missing value with the mean 

dataset['RoamingCalls'] = dataset['RoamingCalls'].fillna(dataset['RoamingCalls'].mean()).round(1)

# Checking if still there is any Null value left
dataset["RoamingCalls"].isnull().sum()


# In[26]:


dataset["PercChangeMinutes"].value_counts()


# In[27]:


# 0.658% of the Missing value in "PercChangeMinutes" column
# Replace the missing value with the mean 

dataset['PercChangeMinutes'] = dataset['PercChangeMinutes'].fillna(dataset['PercChangeMinutes'].mean()).round(1)

# Checking if still there is any Null value left
dataset["PercChangeMinutes"].isnull().sum()


# In[28]:


dataset["PercChangeRevenues"].value_counts()             


# In[29]:


# 0.658% of the Missing value in "PercChangeRevenues" column
# Replace the missing value with the mean 

dataset['PercChangeRevenues'] = dataset['PercChangeRevenues'].fillna(dataset['PercChangeRevenues'].mean()).round(1)

# Checking if still there is any Null value left
dataset["PercChangeRevenues"].isnull().sum()


# In[30]:


dataset["ServiceArea"].value_counts()


# In[31]:


# 0.047% of the Missing value in "ServiceArea" column
# Replace the missing value with the mean 

dataset['ServiceArea'] = dataset['ServiceArea'].fillna(dataset['ServiceArea'].value_counts().index[0])

# Checking if still there is any Null value left
dataset["ServiceArea"].isnull().sum()


# In[32]:


dataset["AgeHH1"].value_counts()             


# In[33]:


# 1.72% of the Missing value in "AgeHH1" column
# Replace the missing value with the mean 

dataset['AgeHH1'] = dataset['AgeHH1'].fillna(dataset['AgeHH1'].mean()).round(1)

# Checking if still there is any Null value left
dataset["AgeHH1"].isnull().sum()


# In[34]:


dataset["AgeHH2"].value_counts()             


# In[35]:


# 1.72% of the Missing value in "AgeHH2" column
# Replace the missing value with the mean 

dataset['AgeHH2'] = dataset['AgeHH2'].fillna(dataset['AgeHH2'].mean()).round(1)

# Checking if still there is any Null value left
dataset["AgeHH2"].isnull().sum()


# In[36]:


dataset["Homeownership"].value_counts()


# In[37]:


# 33.244% of the Missing value in "Homeownership" column
# Replace the missing value with the mean 

dataset['Homeownership'] = dataset['Homeownership'].fillna(dataset['Homeownership'].value_counts().index[0])

# Checking if still there is any Null value left
dataset["Homeownership"].isnull().sum()


# In[38]:


dataset["HandsetPrice"].value_counts()


# In[39]:


# 56.411% (less than 60%) of the Missing value in "HandsetPrice" column
# Replace the missing value with the mean 

dataset['HandsetPrice'] = dataset['HandsetPrice'].fillna(dataset['HandsetPrice'].value_counts().index[0])

# Checking if still there is any Null value left
dataset["HandsetPrice"].isnull().sum()


# In[40]:


dataset["MaritalStatus"].value_counts()


# In[41]:


# 38.197% of the Missing value in "MaritalStatus" column
# Replace the missing value with the mean 

dataset['MaritalStatus'] = dataset['MaritalStatus'].fillna(dataset['MaritalStatus'].value_counts().index[0])

# Checking if still there is any Null value left
dataset["MaritalStatus"].isnull().sum()


# In[42]:


dataset["Churn"].value_counts()


# In[43]:


# Target variable distribution is showing that the dataset is imbalanced:
sns.countplot(x=dataset.Churn)


# In[44]:


# To save the cleaned dataset to your preferred location, please change the file location
# This dataset will be used in Step 2: Data Preprocessing

dataset.to_csv(r'C:\Users\Admin\Downloads\Dataset.csv', index = False)

