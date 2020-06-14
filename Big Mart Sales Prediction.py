#!/usr/bin/env python
# coding: utf-8

# ## Importing the important packages

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# In[26]:


dataset = pd.read_csv('dataset/dataset_for_ML.csv')


# In[27]:


dataset.head()


# In[28]:


# Now we can see while importng we got an unnecessary column and we can get rid of that column by deleting that column.

del dataset['Unnamed: 0']


# In[29]:


# splitting the dataset into training and testing dataset.

train, test = train_test_split(dataset, test_size = 0.20, random_state = 2019)


# In[30]:


# checking the shape of training and testing dataset.

train.shape , test.shape


# In[31]:


# creating the label dataset for training and testing dataset.

train_label = train['Item_Outlet_Sales']
test_label = test['Item_Outlet_Sales']


# In[32]:


# removing the label dataset from training and testing dataset.

del train['Item_Outlet_Sales']
del test['Item_Outlet_Sales']


# In[33]:


# Now we can check the shape of training and testing dataset.

train.shape , test.shape


# In[ ]:





# ## Now after getting the dataset in accurate from, we can start implementing our ML approaches.

# In[37]:


# Importing Cross validation score and means square error from the SKLearn Library.

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# In[76]:


# Let's create a method to evaluate the performance of each model so, we don't have to write the duplicate code.

def modelfit(model, train, test, train_label, test_label, model_name, graph_flag ):
    
    # fitting the model on training dataset.
    model.fit( train, train_label)
    
    # checking the model on testing dataset.
    predict_model = model.predict(test)
    
    # let's find the mean square error
    mse = mean_squared_error(test_label, predict_model)
    
    #Printing Model Report:
    print ('\nModel Report For ', (model_name), '\n\n')
    print("RMSE : %.4g" % np.sqrt(mse))
    # let's find the 
    c_val_score = score = np.sqrt(-(cross_val_score(model, train, train_label, cv = 10, scoring = 'neg_mean_squared_error')))
    print( 'CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g' % (np.mean(c_val_score), np.std(c_val_score), np.min(c_val_score), np.max(c_val_score)))
    if graph_flag:
        fig, axes = plt.subplots(1, 1, figsize=(12,8))
        coef1 = pd.Series(model.coef_, train.columns).sort_values()
        coef1.plot(kind='bar', title='Model Coefficients')


# In[ ]:





# ## Linear Regression Model

# In[77]:


# Importing the Linear Regression Model.

from sklearn.linear_model import LinearRegression


# Initialize the Empty linear Regression Model.

linear_regression = LinearRegression()
# Checking the performance of the model.
modelfit(linear_regression, train, test, train_label, test_label, "Linear Regression", True)


# In[ ]:





# ## Ridge Model.

# In[79]:


# Importing the Ridge Model.
from sklearn.linear_model import Ridge

# Initializing the Ridge Model
ridge_model = Ridge(alpha = 0.05, solver = 'cholesky')

# Evaluating the Ridge Model.
modelfit(ridge_model, train, test, train_label, test_label, "Ridge Model", True)


# In[ ]:





# ## LASSO Model

# In[80]:


# Importing the Lasso Model

from sklearn.linear_model import Lasso

# Initializing the Lasso model
lasso_model = Lasso(alpha = 0.01)

# Evaluating the Lasso Model
modelfit(lasso_model, train, test, train_label, test_label, "Lasso Model", True)


# In[ ]:





# ## Bagging Regression Model

# In[87]:


# Importing the Bagging Regressor Model

from sklearn.ensemble import BaggingRegressor

# Initializing the BaggingRegressor Model
br = BaggingRegressor(max_samples = 70)

# Evaluating the bagging Regressor Model
modelfit(br, train, test, train_label, test_label,"BaggingRegressor Model", False)


# In[ ]:





# ## SVR Model

# In[78]:


# Importing the SVR

from sklearn.svm import SVR

# Initializing the SVR Model
svr = SVR(epsilon = 15, kernel = 'linear')

# Evaluating the SVR Model.
modelfit(svr, train, test, train_label, test_label, 'SVR Model', False)


# In[ ]:





# ## Decision Tree

# In[82]:


# Importing the DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

# Initializing the Decision Tree Regressor

dtr = DecisionTreeRegressor()

# Evaluating the Decision Tree Regressor

modelfit(dtr, train, test, train_label, test_label, 'DTR Model', False)


# In[ ]:





# ## Random Forest

# In[85]:


# Importing the RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

#Initializing the RandomForestRegressor

rf = RandomForestRegressor()

# Evaluating the RandomForestRegressor

modelfit(rf, train, test, train_label, test_label, 'RandomForestRegressor', False)


# In[ ]:





# ## ADA Boosting

# In[89]:


# importing adaptive boosting regressor algorithm

from sklearn.ensemble import AdaBoostRegressor

# initializing adaboosting algorithm
ada = AdaBoostRegressor()
# evaluating adaboosting
modelfit(ada, train, test, train_label, test_label, 'Adaptive Boosting Model', False)


# In[ ]:





# ## Gradient Boosting

# In[92]:


# Importing Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

# Initializing Gradient Boosting

gbr = GradientBoostingRegressor()

# Evaluatin Gradient Boosting Algorithm
modelfit(gbr, train, test, train_label, test_label, 'Gradient Boosting Model', False)


# In[ ]:





# # Now after checking the performance of each algorithm lets trained out whole model on complete training set and import testing dataset and preprocess it for getting the predictions.

# In[94]:


# importing the test dataset.

df_test = pd.read_csv('dataset/test.csv')


# In[95]:



# checking the shape of test dataset.
df_test.shape


# In[98]:


# checking the top 5 rows of testing dataset.

df_test.head()


# In[102]:


# making a list of the attributes which can effect the sales.

attributes = ['Item_MRP',
 'Outlet_Type',
 'Outlet_Size',
 'Outlet_Location_Type',
 'Outlet_Establishment_Year',
 'Outlet_Identifier',
 'Item_Type']


# In[104]:


# limiting the dataset to only those attributes which can effect the sale of the outlet.

df_test = df_test[attributes]


# In[106]:


# checking the shape of the sales.

df_test.shape


# In[109]:


# checking the info about the test dataset

df_test.info()


# In[110]:


# changing the data type of the attributes to the categorical data type.

df_test.Item_MRP = pd.cut(df_test.Item_MRP,bins=[25,75,140,205,270],labels=['a','b','c','d'],right=True)
df_test.Item_Type = df_test.Item_Type.astype('category')
df_test.Outlet_Size = df_test.Outlet_Size.astype('category')
df_test.Outlet_Identifier = df_test.Outlet_Identifier.astype('category')
df_test.Outlet_Establishment_Year = df_test.Outlet_Establishment_Year.astype('category')
df_test.Outlet_Type = df_test.Outlet_Type.astype('category')
df_test.Outlet_Location_Type = df_test.Outlet_Location_Type.astype('category')


# In[111]:


# again chacking the info of the test dataset.

df_test.info()


# In[112]:


# making a function to replace the null value in the outlet_size attribute

def function_replacing_null_Values(x):
    if x == 'OUT010' :
        return 'High'
    elif x == 'OUT045' :
        return 'Medium'
    elif x == 'OUT017' :
        return 'Medium'
    elif x == 'OUT013' :
        return 'High'
    elif x == 'OUT046' :
        return 'Small'
    elif x == 'OUT035' :
        return 'Small'
    elif x == 'OUT019' :
        return 'Small'
    elif x == 'OUT027' :
        return 'Medium'
    elif x == 'OUT049' :
        return 'Medium'
    elif x == 'OUT018' :
        return 'Medium'


# In[113]:


# using Function to fill null values.

df_test['Outlet_Size'] = df_test.Outlet_Identifier.apply(function_replacing_null_Values)


# In[114]:


# making hot encoded vectors from the categorical variables.

test_data = pd.get_dummies(df_test.iloc[:,0:6])


# In[123]:


# checking the shape of the test_data

test_data.shape


# ## Now after preprocessing the test dataset, we can get our complete dataset of training and lets train the gradient boosting algorithm on complete.

# In[ ]:





# In[116]:


# first lets create the labeled dataset from the training set.

train_label = dataset['Item_Outlet_Sales']


# In[118]:


# let's remove the labeled dataset from the training set.

del dataset['Item_Outlet_Sales']


# In[120]:


# after removing the dataset let's check the shape of training set.

dataset.shape


# In[122]:


# to verify let's check the shape of test set.

test_data.shape


# In[119]:


# and train label shape.

train_label.shape


# In[ ]:





# ## Let's train Gradient Boosting Algorithm and take the predictions and add it to the submission file.

# In[126]:


# Importing Gradient Boosting and GradientSearchCV

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Initializing Gradient Boosting

gb = GradientBoostingRegressor(max_depth = 7, n_estimators=200, learning_rate = 0.01)

# making a list of params for the Gradient Search CV
param = [{'min_samples_split': [5, 9, 13], 'max_leaf_nodes': [3, 5, 7, 9], 'max_features': [8, 10, 15, 18]}]

# initializing the GradientSearchCV with the params
gs = GridSearchCV(gb, param, cv = 5, scoring = 'neg_mean_squared_error')


# In[127]:



# fitting the model on training dataset.
gs.fit(dataset, train_label)


# In[128]:


gb = gs.best_estimator_


# In[129]:


# getting the output of testing dataset after training.

predict = gb.predict(test_data)


# In[130]:


# checking the shape of the predictions.

predict.shape


# In[133]:


# importing the submission dataset.

df_submission = pd.read_csv('dataset/submission.csv')


# In[134]:


# checking the shape of the submission dataset.

df_submission.shape


# In[136]:


# checking the top 10 rows of the submission dataset.

df_submission.head(10)


# In[139]:


# we can see, we have an unnecessary column Unnamed: 0, we can del that column

del df_submission['Unnamed: 0']


# In[137]:


# creating a new column in the submission dataset, for saving the predicted sales value.

df_submission['Item_Outlet_Sales_Predicted'] = predict


# In[140]:


# checking the top 10 values.

df_submission.head(10)


# In[141]:


df_submission.to_csv('predicted_dataset.csv')


# In[ ]:




