# multiple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


print("######################################################")

# Encoding categorical data
# encoding independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X.astype(float)

# Avoid Dummy Variable trap

X = X[:, 1:]

print("######################################################")

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building optimal model using Backward Elimination
# Since in Above regression model we have used all independent variables as equal,
# But what if there are some independent variable whose significance is higher that the others
# and have greater impact on the dependent variable, while some are statistically insignificant.
# So if we remove the statistically insignificant variable we would still have greaat prediction.
# So the goal here is, to find an optimal team of independent variables such that each independent
# variable has a powerful impact in determining the dependent variable profit which can be both posotive or negative
# Hence we use backward elemination

import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X , axis=1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()