# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt #create nice charts
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # in order to take all row data -> [ : ,  all the columns except the last one -> :-1]
y = dataset.iloc[:, -1].values # all th rows and the last column

# Taking care of missing data
# how to identify missing data
missing_data = dataset.isnull().sum()

# missing data is replaced by average data of the column in which data is missing
# np.nan -> blank cells
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

# Encoding the Independent Variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Encoding the Dependent Variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
# Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:, 3:] = scaler.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scaler.transform(X_test[:, 3:]) #new data transformed with the same scaler obtained during training machine

print(X_train)
print(X_test)

