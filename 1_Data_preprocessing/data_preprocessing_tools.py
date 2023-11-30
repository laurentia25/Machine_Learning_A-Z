# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt #create nice charts
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # in order to take all row data -> [ : ,  all the columns except the last one -> :-1]
y = dataset.iloc[:, -1].values # all th rows and the last column
print(x)
print(y)
# Taking care of missing data
# Encoding categorical data
# Encoding the Independent Variable
# Encoding the Dependent Variable
# Splitting the dataset into the Training set and Test set
# Feature Scaling
