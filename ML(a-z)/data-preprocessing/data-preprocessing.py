import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :3].values
Y = dataset.iloc[:, -1].values

print(X)
print(Y)

print("######################################################")

########################################################
# Taking care of missing data

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("After filling missing data")
print(X)

print("######################################################")

########################################################
#Encoding categorical data
#Encoding the Independent Variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("Encoding the Independent Variable")
print(X)

#Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(Y)
print("Encoding the Dependent Variable")
print(y)
print("######################################################")


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

print(X_train)
print("----------------------------")
print(X_test)
print("----------------------------")
print(y_train)
print("----------------------------")
print(y_test)

print("######################################################")
### Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print("----------------------------")
print(X_test)