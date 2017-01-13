# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 16:25:57 2017

@author: CHOU_H
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
#%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
# Display a description of the dataset
display(data.describe())

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [10,73,345]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)


# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.copy()
col_name = new_data.columns
y_all = new_data[col_name]

# TODO: Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = (None, None, None, None)

# TODO: Create a decision tree regressor and fit it to the training set
regressor = None

# TODO: Report the score of the prediction using the testing set
score = None