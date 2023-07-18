# Predicting Airbnb Listed Prices in Melbourne 🏘️
---

# Project Summary
In this project, we try to predict the prices of Airbnb listings in Melbourne based on the characteristics of the properties. 

The goal is to develop accurate and reliable forecasting models that can estimate the prices of the listings using the provided dataset. 

In the real world, accurate price forecasting for Airbnb listings can have significant implications for both hosts and guests. For hosts, it helps them set competitive prices, optimize their revenue, and attract potential guests. For guests, it helps them evaluate affordability, compare listings, and make informed decisions for optimal accommodation choices that provide the best value for their money. 

Additionally, the techniques and methodologies used in this assignment can be extended to other similar domains where price forecasting plays a crucial role.


# Libraries and Packages
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
```


# Notebook Structure
- **Main Notebook** - This is where the codes for the main part of the analysis and predictive modeling are
- ** Data Preparation** - This notebook is where we cleaned all the files for the data
- **Forecasting Model & Results** -  This is where we deep-dive into the analysis of all the variables and commenting on the results 
