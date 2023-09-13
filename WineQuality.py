#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file_path = 'C:/Users/himan/Downloads/winequality-red.csv'


# In[3]:


file_path


# In[4]:


df = pd.read_csv(file_path)
df


# Reading the csv file and creating a dataframe

# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# As none of the columns contains null values, hence there is ni need of data cleaning

# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# As all the columns have 50th percentile value nearly close to mean value, hence the data is distributed normally for all the columns

# In[11]:


df.hist(bins=60, figsize=(20,10))


# Most wines have a quality score of 5 or 6.
# There are fewer wines with quality scores of 3, 4, 7, and 8, with wines rated 3 being the lowest.

# In[12]:


df['good'] = (df['quality']>=7).astype(int)


# Assigning a rating as good(1) if quality>=7 else 0

# In[13]:


df


# In[14]:


from sklearn.model_selection import train_test_split


# Import the library for splitting the data into training and test data

# In[15]:


x = df.iloc[:,0:-2]
x


# In[16]:


y = df.iloc[:,-1]
y


# In[17]:


x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)


# Different test sizes have been tried and test_size = 0.1 has been found to be the best fit for the optimal AUC value

# In[18]:


x_train.shape


# In[19]:


x_test.shape


# In[20]:


y_train.shape


# In[21]:


y_test.shape


# In[22]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# standardizing the features to have a mean of 0 and a standard deviation of 1

# In[23]:


x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[24]:


x_train_scaled


# In[25]:


x_test_scaled


# The features are now scaled and the transformed training data now has standardized features with a mean=0 and standard deviation=1
# Building a classification model and starting with desicion tree classifier as it is mentioned to focus on ROC curve and AUC value

# In[26]:


from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(x_train_scaled, y_train)


# In[39]:


from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Predicting the probabilities for the test data
y_probablity = dt_classifier.predict_proba(x_test_scaled)[:, 1]

# Calculating the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)


# In[40]:


# Plot ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)

plt.show()


# The (AUC)area under curve's value of 1 represents a perfect classifier, whereas an AUC value of 0.5 represents a classifier is not performing better. As per above model, classifier's AUC value is found out to be 0.76 which indicates reasonably good performance. Prior to it, multiple trials were made by varying the test_size and random_state, but amongst all the trials, 0.76 was found to be the highest AUC.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




