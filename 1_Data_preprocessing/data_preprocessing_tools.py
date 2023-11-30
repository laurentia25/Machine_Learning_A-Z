# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt #create nice charts
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # in order to take all row data -> [ : ,  all the columns except the last one -> :-1]
y = dataset.iloc[:, -1].values # all th rows and the last column
print(dataset)
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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Encoding the Dependent Variable
# Splitting the dataset into the Training set and Test set
# Feature Scaling
