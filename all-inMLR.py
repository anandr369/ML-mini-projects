#importing the required libraries
import pandas as pd
import numpy as np

#fetching the dataset into pandas dataframe
dataset = pd.read_csv('insurance.csv')

#Converting the categorical method into numeric 
dataset['sex'] = dataset['sex'].astype('category')
dataset['smoker'] = dataset['smoker'].astype('category')
dataset['region'] = dataset['region'].astype('category')
dataset = pd.get_dummies(dataset)

#Getting the dependent and independent variables
X= dataset[['age','bmi','children','sex_female','smoker_no','smoker_yes','sex_male','region_northeast','region_northwest','region_southeast','region_southwest']]
Y= dataset['charges']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression to the training set
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 50).fit(X_train, Y_train)
y_pred = rfr.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(Y_test, y_pred)))
print('RMSE is {}'.format(rmse))
print("\n")





