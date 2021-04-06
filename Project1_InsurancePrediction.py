#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import relevent libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import warnings # ignore warnings
warnings.simplefilter("ignore")


# #### Data loading and Overview

# In[2]:


# Importing Dataset
data=pd.read_csv('C:\\Users\\LENOVO\\Desktop\\insurance.csv')


# In[3]:


# Data Overview
data.head() # show first five values of data


# In[4]:


data.shape # number of rows and columns


# In[5]:


data.info() # shape and properties of data


# The data containd individual medical cost billed by health insurance and some related variables like age,sex,bmi,number of children covered by insurance,smoker or non-smoker and residential area of the individual.
# Data has 7 variables and 1338 samples.

# #### Data Preprocessing

# In[6]:


# dealing with missing value
data.isnull().sum()


# In[7]:


# check for duplicate values
data.duplicated().sum()


# In[8]:


data=data.drop_duplicates()


# In[9]:


data.shape


# ##### Descriptive Statistics

# In[10]:


data.describe().T


# The age of individuals ranges from 18 to 64 with an average of 39. The maximum insurance cost is 63770 though 75% falls under 16657.

# In[11]:


#removing outliers 
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[12]:


df = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape


# In[13]:


df.describe()


# ##### Finding Predictor variables
# 

# Since our response variable is charges, we are checking the relation of charges with other variables through different plots.

# ###### correlation between age and charges by plotting scatterplot

# In[14]:


plt.title('Relation between Age and Charges')
sns.scatterplot(x=df['age'],y=df['charges'])
plt.show()
plt.title('Regression between Age and Charges')
sns.regplot(x=df['age'],y=df['charges'])
plt.show()


# We can notice that older people tend to pay slightly more but to make it more clearer we can draw a regression link. The regression line shows positive correlation. Hence, Age plays small role in predicting insurance price.

# ###### Finding correlation between BMI and Charges by ploting a scatter plot.

# In[15]:


plt.title('Relation between BMI and Charges')
sns.regplot(x=df['bmi'],y=df['charges'])
plt.show()


# The regression line and the scatter plot shoes that there is no significant relation between bmi and charges

# ###### Finding correlation between Smokers and Charges by ploting a categorical plot.

# The above regression line shows that charges for smokers the charge is higher than non smokers. 

# In[16]:


sns.swarmplot(x=df['smoker'],y=df['charges'])


# On average, non-smokers are charged less than smokers, and the customers who pay the most are smokers whereas the customers who pay the least are non-smokers. Hence, smoking habits determine the insurance charges.

# ###### Finding correlation between Children and Charges by ploting a bar plot.

# In[17]:


sns.barplot(x=df['children'], y=df['charges'])


# We can easily say that person having 2 and 3 children tend to pay more. Surprisingly person having 5 children pays the least insurance charges. Hence, definately children is a predictor variable.

# ###### Finding correlation between Sex and Charges by ploting a swarm plot.

# In[18]:


sns.swarmplot(x=df['sex'],y=df['charges'])
plt.show()


# We cannot find much difference between cost paid by mail and female in first plot and sex is alomst equally scattered.

# ###### correlation between region and charges by plotting boxplot

# In[19]:


sns.barplot(x=df['region'], y=df['charges'])
plt.show()


# there is a slight increase in charges from southwest to northeast.

# Hence for the response variable charges, I am taking age, smoker,children and region as predictor variables.

# ### Multiple Linear Regression

# In[20]:


# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[21]:


df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])


# In[22]:


df.head(5)


# In[23]:


x=df.drop(columns=['sex','bmi','charges'])
y=df['charges'].values.reshape(-1,1)


# In[24]:


x.shape


# In[25]:


from sklearn.linear_model import LinearRegression # to fit the model
from sklearn.metrics import mean_squared_error # to find mean square error for model accuracy
from sklearn.model_selection import train_test_split# spliting main data for training and testing


# In[26]:


reg=LinearRegression() #initialise regression model


# In[27]:


# We are splitting the data into two parts- test data and train data.
# using train data we create regression model
# using test data, we van calculate accuracy of our data.
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size = 0.2 , random_state = 51)


# In[28]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[29]:


#Train the model with training data
reg.fit(x_train, y_train)


# In[30]:


# print the coefficients for each columns of model
print(reg.coef_) # f(x)= ax+b =y, a is coeff
rc=reg.coef_


# In[31]:


a1=pd.DataFrame(np.array(x.columns),columns=['predictors'])
a2=pd.DataFrame(rc).T.rename(columns={0:'coeff'})
pd.concat([a1 , a2] , axis=1)


# In[32]:


y_pred= reg.predict(x_test)


# In[33]:


y_pred_df = pd.DataFrame(y_pred, columns=["Predicted Values" ])
y_test_df = pd.DataFrame(np.array(y_test), columns=["Real Values"])
avp=pd.concat([y_test_df , y_pred_df] , axis=1)

avp.head(10)


# So we can see the difference between the actual value and predicted value here.

# ##### Finding Model Accuracy
# + There are differetnt ways to find the accuracy. Here i am plotting actual vs predicted plot and finding mean squared error and r2 score.

# In[35]:


plt.title('Regression line/ actual vs predicted')
sns.regplot(x=y_test,y=y_pred)
plt.show()


# The above regression figure shows that all the data points upto 15000 falls on or near the regression line. But charges>15000 falls far away from regression line.

# In[36]:


# calculating mean square error.
mean_squared_error(y_test,y_pred)


# For a better fit model the mse will be smaller. but hear mse is very large.

# In[37]:


# checking model performance by finding r2 wrt test data.
r2=reg.score(x_test,y_test)
print(r2)


# r2 score shows that our model is 59.7% accurate. That means, 59.7% variability in response variable is predicted by predictor variables. 

# In[ ]:




