#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install category-encoders


# In[2]:


############## Importing Libraries ##############
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[3]:


############## Importing Dataset ##############
dataset = pd.read_csv('Dataset.csv')
dataset.head()


# # Data Preprocessing

# ### General Exploration and Encoding Categorical Data

# In[4]:


dataset.columns


# In[5]:


dataset.shape


# In[6]:


dataset.columns.to_series().groupby(dataset.dtypes).groups


# In[7]:


dataset.dtypes


# In[8]:


dataset.dtypes.value_counts()


# In[9]:


dataset.select_dtypes(include=['object']).describe()


# In[10]:


for i in dataset.columns:
    print(f"Unique {i}'s count: {dataset[i].nunique()}")
    print(f"{dataset[i].unique()}\n")


# In[11]:


dataset["ServiceArea"].value_counts()


# In[12]:


dataset["ServiceArea"].duplicated().any()


# In[13]:


dataset.columns.get_loc("ServiceArea")


# In[14]:


dataset["ServiceArea"].dtype


# In[15]:


dataset["ServiceArea"] = dataset["ServiceArea"].astype("category")


# In[16]:


dataset["ServiceArea"] = pd.DataFrame(data = dataset["ServiceArea"].cat.codes, columns= ["ServiceArea"])


# In[17]:


dataset["ServiceArea"].head()


# In[18]:


dataset['ServiceArea'].value_counts()


# In[19]:


# all categorical variables take on 2 values (mostly yes/no) — therefore are transformed to binary
# and variables with more than 2 values are transformed to integers


# In[20]:


# Below are all the steps to Transform categorical variables into Binary/Integers:


# In[21]:


# creating instance of labelencoder
le = LabelEncoder()


# In[22]:


dataset['ChildrenInHH'].value_counts()


# In[23]:


le.fit(dataset['ChildrenInHH'])
dataset['ChildrenInHH'] = le.transform(dataset['ChildrenInHH'])


# In[24]:


dataset['HandsetRefurbished'].value_counts()


# In[25]:


le.fit(dataset['HandsetRefurbished'])
dataset['HandsetRefurbished'] = le.transform(dataset['HandsetRefurbished'])


# In[26]:


dataset['HandsetWebCapable'].value_counts()


# In[27]:


le.fit(dataset['HandsetWebCapable'])
dataset['HandsetWebCapable'] = le.transform(dataset['HandsetWebCapable'])


# In[28]:


dataset['TruckOwner'].value_counts()


# In[29]:


le.fit(dataset['TruckOwner'])
dataset['TruckOwner'] = le.transform(dataset['TruckOwner'])


# In[30]:


dataset['RVOwner'].value_counts()


# In[31]:


le.fit(dataset['RVOwner'])
dataset['RVOwner'] = le.transform(dataset['RVOwner'])


# In[32]:


dataset['Homeownership'].value_counts()


# In[33]:


le.fit(dataset['Homeownership'])
dataset['Homeownership'] = le.transform(dataset['Homeownership'])


# In[34]:


dataset['BuysViaMailOrder'].value_counts()


# In[35]:


le.fit(dataset['BuysViaMailOrder'])
dataset['BuysViaMailOrder'] = le.transform(dataset['BuysViaMailOrder'])


# In[36]:


dataset['RespondsToMailOffers'].value_counts()


# In[37]:


le.fit(dataset['RespondsToMailOffers'])
dataset['RespondsToMailOffers'] = le.transform(dataset['RespondsToMailOffers'])


# In[38]:


dataset['OptOutMailings'].value_counts()


# In[39]:


le.fit(dataset['OptOutMailings'])
dataset['OptOutMailings'] = le.transform(dataset['OptOutMailings'])


# In[40]:


dataset['NonUSTravel'].value_counts()


# In[41]:


le.fit(dataset['NonUSTravel'])
dataset['NonUSTravel'] = le.transform(dataset['NonUSTravel'])


# In[42]:


dataset['OwnsComputer'].value_counts()


# In[43]:


le.fit(dataset['OwnsComputer'])
dataset['OwnsComputer'] = le.transform(dataset['OwnsComputer'])


# In[44]:


dataset['HasCreditCard'].value_counts()


# In[45]:


le.fit(dataset['HasCreditCard'])
dataset['HasCreditCard'] = le.transform(dataset['HasCreditCard'])


# In[46]:


dataset['NewCellphoneUser'].value_counts()


# In[47]:


le.fit(dataset['NewCellphoneUser'])
dataset['NewCellphoneUser'] = le.transform(dataset['NewCellphoneUser'])


# In[48]:


dataset['NotNewCellphoneUser'].value_counts()


# In[49]:


le.fit(dataset['NotNewCellphoneUser'])
dataset['NotNewCellphoneUser'] = le.transform(dataset['NotNewCellphoneUser'])


# In[50]:


dataset['OwnsMotorcycle'].value_counts()


# In[51]:


le.fit(dataset['OwnsMotorcycle'])
dataset['OwnsMotorcycle'] = le.transform(dataset['OwnsMotorcycle'])


# In[52]:


dataset['MadeCallToRetentionTeam'].value_counts()


# In[53]:


le.fit(dataset['MadeCallToRetentionTeam'])
dataset['MadeCallToRetentionTeam'] = le.transform(dataset['MadeCallToRetentionTeam'])


# In[54]:


dataset['MadeCallToRetentionTeam'].dtypes


# In[55]:


dataset['CreditRating'].value_counts()


# In[56]:


dataset['CreditRating'] = dataset['CreditRating'].apply(lambda v : int(v.split('-')[0]) if v else np.nan)


# In[57]:


dataset["CreditRating"].dtypes


# In[58]:


dataset['PrizmCode'].value_counts()


# In[59]:


# converting type of column to 'category' as by default it's 'object'
dataset['PrizmCode'] = dataset['PrizmCode'].astype('category')


# In[60]:


le.fit(dataset['PrizmCode'])
dataset['PrizmCode'] = le.transform(dataset['PrizmCode'])


# In[61]:


dataset['Occupation'].value_counts()


# In[62]:


# converting type of column to 'category' as by default it's 'object'
dataset['Occupation'] = dataset['Occupation'].astype('category')


# In[63]:


le.fit(dataset['Occupation'])
dataset['Occupation'] = le.transform(dataset['Occupation'])


# In[64]:


dataset['MaritalStatus'].value_counts()


# In[65]:


le.fit(dataset['MaritalStatus'])
dataset['MaritalStatus'] = le.transform(dataset['MaritalStatus'])


# In[66]:


dataset['Churn'].value_counts()


# In[67]:


le.fit(dataset['Churn'])
dataset['Churn'] = le.transform(dataset['Churn'])


# In[68]:


# 0 is 'No' and 1 is 'Yes'
dataset['Churn'].value_counts()


# In[69]:


# As you can clearly see below, the dataset is quite imbalanced:


# In[70]:


# The Number of Customer's Churning is very less than the Customer's Churning (more than half a difference)
dataset["Churn"].value_counts()


# In[71]:


# Target variable distribution is showing that the dataset is imbalanced:
sns.countplot(x=dataset.Churn)


# In[72]:


dataset.head()


# In[73]:


# To save the final dataset to your preferred location, please change the file location
# This dataset will be used in Step 3: Classification without Feature Selection

dataset.to_csv(r'C:\Users\Admin\Downloads\Customer_Churn_Final_Dataset.csv', index = False)

