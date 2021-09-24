#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv("C:/Users/dell/Desktop/Projects/Vehicle/car data.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[13]:


data.columns.unique()


# In[26]:


print(data["Seller_Type"].unique())
print(data["Transmission"].unique())
print(data["Owner"].unique())


# In[21]:


data.dtypes


# In[28]:


#check if there is any null values in the data set.

data.isnull().sum()


# In[29]:


data.describe()


# In[30]:


data.columns


# In[34]:


final_dataset = data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[36]:


final_dataset.head()


# In[39]:


final_dataset["Current_Year"] = 2020


# In[40]:


final_dataset.head()


# In[43]:


final_dataset["No. of years"] = final_dataset["Current_Year"]-final_dataset["Year"]


# In[45]:


final_dataset.head()


# In[46]:


final_dataset.drop(["Year","Current_Year"], axis = 1, inplace = True)


# In[47]:


final_dataset.head()


# In[48]:


final_dataset = pd.get_dummies(final_dataset, drop_first= True)


# In[49]:


final_dataset


# In[51]:


final_dataset.shape


# In[53]:


final_dataset.corr()


# In[54]:


import seaborn as sns


# In[55]:


sns.pairplot(final_dataset)


# In[58]:


cormat = final_dataset.corr()


# In[109]:


top_Corr_features = cormat.index
plt.figure(figsize= (10,10))
sns.heatmap(final_dataset[top_Corr_features].corr(), annot = True, cmap= "RdYlGn")


# In[110]:


x = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()


# In[116]:


model.fit(x,y)


# In[119]:


print(model.feature_importances_)


# In[121]:


feat_importance = pd.Series(model.feature_importances_,index = x.columns)


# In[127]:


# feat_importance.nlargest().plot(kind ="barh")
# plt.show()


# In[128]:


x_train,x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2)


# In[130]:


from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()


# In[176]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num =12)]
max_features = ["auto","sqrt"]
max_depth = [int(x) for x in np.linspace(start = 5, stop =30, num = 6)]
min_sample_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


# In[177]:


random_forest = RandomForestRegressor(n_estimators,max_features,max_depth,min_sample_split,min_samples_leaf)


# In[178]:


from sklearn.model_selection import RandomizedSearchCV


# In[183]:


random_grid = {"n_estimators": n_estimators, 
               "max_features" : max_features,
               "max_depth" : max_depth, 
               "min_samples_leaf" : min_samples_leaf               
              }

# print(random_grid)


# In[184]:


rf = RandomForestRegressor()


# In[185]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = "neg_mean_squared_error", n_iter = 10, cv = 10, random_state = 42, verbose = 3, n_jobs=1)


# In[186]:


rf_random.fit(x_train,y_train)


# In[187]:


prediction = rf_random.predict(x_test)


# In[189]:



# In[191]:


# sns.distplot(y_test-prediction)


# In[192]:


# plt.scatter(y_test,prediction)


# In[202]:


import pickle

file = open("C:/Users/dell/Desktop/Projects/Vehicle prediction/random_forest_model.pkl" , "wb")
pickle.dump(rf_random,file )


# In[ ]:




