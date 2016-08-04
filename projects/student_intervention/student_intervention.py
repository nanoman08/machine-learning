# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:54:21 2016

@author: hungtaochou
"""

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

print student_data.describe()

print student_data.shape

print student_data.head()

print sum(student_data['passed']=='yes'), sum(student_data['passed']=='no')
