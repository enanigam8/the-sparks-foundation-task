#!/usr/bin/env python
# coding: utf-8

# # Task # 1 To Explore Supervised Machine Learning
# 
# In this notebook I am using Python Scikit-Learn library for machine learning to implement simple linear regression functions.
# 
# 

# # Simple Linear Regression
# 
# 1.In this regression task I will be predicting the percentage of marks that a student is expected to score based upon the number of hours studied. 
# 2.This is a simple linear regression tas involving just two variables.

# #    Author- ENA NIGAM

# # Importing Libraries

# In[3]:


# Importing all required libraries
import pandas as pd # used for data processing and analysis
import numpy as np  # used for various array's operation
import matplotlib.pyplot as plt  # used for data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Dataset

# In[9]:


#reading datwa remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print ("Data imported successfully")

s_data.head(5)


# In[10]:


s_data.shape # get no. of rows and columns from dataset


# In[11]:


s_data.describe() # get summary of statistical details pertaining to columns of dataset


# In[12]:


s_data.info()  # get concise summary/ basic information of dataset


# # Ploting Graph

# # Now ploting the data points in a 2D graph to eyeball our dataset and see 
# if we can manually find any relationship between the data.

# In[14]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Studied Hours vs Percentage Scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# # Preparing the data
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs
# In[15]:


X = s_data.iloc[:, :-1].values  # attributes as inputs
Y = s_data.iloc[:, 1].values  # labels as outputs


# In[16]:


print('inputs:\n')
print(X)


# In[17]:


print('outputs:\n')
print(Y)


# # Spiliting the dataset into Training and Testing datasets
# 
# 1.Now we have our attributes as inputs and labels as outputs.
# 
# 2.The next step is to split this data into training and test sets.
# 
# 3.for this We wiil be using Python's Scikit-Learn's built-in _train_testsplit() method:

# In[20]:


# importing the method called train_test_split from sklearn's model_selection
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# # Training the Algorithm

# 1.Now, We have splite the data into training and testing sets.
# 
# 2.And now finally we'll train our algorithm.

# In[21]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 

print("Training complete.")


# # Plotting Regression
# 1.As we have trained our algorithm of regression.
# 
# 2.now we'll plot it in a graph.

# In[22]:


# Plotting the regression line
line = regressor.coef_*X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y, color='red')
plt.plot(X, line, color='blue');
plt.show()


# # Making Predictions
# 
# 1.As we have trained our algorithm and see in a graph
# 
# 2.Now we'll make some predictions.

# In[23]:


# first printing the testing data (in hours)
print(X_test)


# In[24]:


# now we'll predict the scores (in presentage)
Y_predict = regressor.predict(X_test) # Predicting the scores
print(Y_predict)


# In[25]:


# Comparing Actual data vs Predicted data
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_predict})  
df


# # Predicting score if a student studies for 9.25 hours

# In[26]:


# now I will predcit with my own data
hours = 9.25
my_predict = regressor.predict([[hours]])
print("No of Hours a student sudies = {}".format(hours))
print("Predicted Score of marks in % = {}".format(my_predict[0]))


# # Evaluating the model
# 
# Finally we'll be evaluating the performance of algorithm.
# 
# By calcuating some metrices like mean square error, etc.

# In[28]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_predict))


# In[29]:


print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_predict))


# In[30]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)))


# # Checking Accurancy of the Model
# 
# R Squared value is close to 1, this is a good model

# In[31]:


regressor.score(X,Y)


# In[32]:


plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_test,Y_predict, color='red')
plt.show()


# Our model is 95.26% accurate
