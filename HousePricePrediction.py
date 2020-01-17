#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:29:37 2019

@author: Rifat

Problem Statement:
    House Prices: Advanced Regression Techniques; Predict sales prices and practice feature engineering, RFs, and gradient boosting

Data Source:
    https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
"""

#======================import libs=====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
plt.rcParams['figure.figsize'] = (10.0, 8.0)



#===================== Data Import========================
#1 import data
dataset = pd.read_csv("dataset/dataset.csv")

#===================== Data observation========================
#analyze the data
print(dataset.shape)

dataset.head()

dataset.info()

#Check the missing missing values:
dataset.apply(lambda x: sum(x.isnull()))

#===================== Data Processing ========================
# seperate the numeric and categorical features
numeric_features = [f for f in dataset.columns if dataset[f].dtype != object]
categorical_features = [f for f in dataset.columns if dataset[f].dtype == object]
all_features=numeric_features
print ("There are {} numeric and {} categorical features in the dataset".format(len(numeric_features),len(categorical_features)))

#impute the missing Null values with  mean value
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy ='mean')
dataset[numeric_features]=imputer.fit_transform(dataset[numeric_features])

col_miss_values=list(dataset.columns[dataset.isnull().any()])

for col in col_miss_values:
    mode=stats.mode(dataset[col]).mode
    dataset[col].fillna(mode[0], inplace=True)


#================FEATURE ENGINEERING========================

#Correlation analysis
corr_threshold=0.8
corr=dataset.corr()

#identify the  features those are strongly correlated with the target variable
cols = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if abs(corr.iloc[i,j]) >= corr_threshold:
            if cols[j]:
                cols[j] = False
            
all_cols=corr.columns.tolist()
sel_cols = corr.columns[cols].tolist()
#remove some of the features those are strongly corelated with the target variable
not_sel_cols=list(set(all_cols)-set(sel_cols)) #['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd']


dataset["ExterCond"] = dataset["ExterCond"].map({ np.nan: 0,  "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('ExterCond')

dataset["BsmtCond"] = dataset["BsmtCond"].map({ np.nan: 0,   "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('BsmtCond')

dataset["BsmtExposure"] = dataset["BsmtExposure"].map({ np.nan: 0,   "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)
all_features.append('BsmtExposure')

dataset["BsmtFinType1"] = dataset["BsmtFinType1"].map({ np.nan: 0,   'GLQ':6, 'ALQ':4, 'Unf':5, 'Rec':3, 'BLQ':1, 'LwQ':2}).astype(int)
all_features.append('BsmtFinType1')

dataset["BsmtFinType2"] = dataset["BsmtFinType2"].map({ np.nan: 0,   'GLQ':6, 'ALQ':4, 'Unf':5, 'Rec':3, 'BLQ':1, 'LwQ':2}).astype(int)
all_features.append('BsmtFinType2')

dataset["BsmtQual"] = dataset["BsmtQual"].map({ np.nan: 0,   "Fa": 1, "TA": 2, "Gd": 3, "Ex":4}).astype(int)
all_features.append('BsmtQual')

dataset["ExterQual"] = dataset["ExterQual"].map({ np.nan: 0,   "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('ExterQual')

#pivot=dataset.pivot_table(index='Fence', values='SalePrice', aggfunc=np.median)
#pivot.plot(kind='bar', color='red')
dataset["Fence"] = dataset["Fence"].map({ np.nan: 0, 'MnPrv':2, 'GdWo':3, 'GdPrv':4, 'MnWw':1}).astype(int)
all_features.append('Fence')

dataset["FireplaceQu"] = dataset["FireplaceQu"].map({ np.nan: 0,   "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('FireplaceQu')

#pivot=dataset.pivot_table(index='Functional', values='SalePrice', aggfunc=np.median)
#pivot.plot(kind='bar', color='red')
dataset["Functional"] = dataset["Functional"].map({ np.nan: 0, 'Typ':7 ,'Min1':4, 'Maj1':6 ,'Min2':5 ,'Mod':3 ,'Maj2' :1, 'Sev':2}).astype(int)
all_features.append('Functional')

dataset["GarageCond"] = dataset["GarageCond"].map({ np.nan: 0,  "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('GarageCond')

#pivot=dataset.pivot_table(index='GarageFinish', values='SalePrice', aggfunc=np.median)
#pivot.plot(kind='bar', color='red')

dataset["GarageFinish"] = dataset["GarageFinish"].map({ np.nan: 0, 'RFn':2, 'Unf':1, 'Fin':3}).astype(int)
all_features.append('GarageFinish')

dataset["GarageQual"] = dataset["GarageQual"].map({ np.nan: 0,  "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('GarageQual')


dataset["HeatingQC"] = dataset["HeatingQC"].map({ np.nan: 0,  "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('HeatingQC')


dataset["KitchenQual"] = dataset["KitchenQual"].map({ np.nan: 0,   "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('KitchenQual')

dataset["PoolQC"] = dataset["PoolQC"].map({ np.nan: 0,   "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex":5}).astype(int)
all_features.append('PoolQC')

#define the onehotencoder to then splits the categorical columns into multiple columns
def oneHotEncoding(df, column_name):
       onehot_df = pd.DataFrame(index = df.index)
       onehot_df[column_name] = df[column_name]
       dummies = pd.get_dummies(onehot_df[column_name], prefix="_"+column_name)
       onehot_df = onehot_df.join(dummies)
       onehot_df = onehot_df.drop([column_name], axis=1)
      
       for f in list(dummies.columns):
           all_features.append(f)
       return  onehot_df  

onehotfeatures=['Alley','BldgType','CentralAir','Condition1','Condition2','Electrical','Foundation','GarageType','Heating','HouseStyle','LandContour',
               'LandSlope','LotConfig','LotShape','MSZoning','MasVnrType','MiscFeature','RoofStyle','SaleCondition','SaleType', 'Street','Utilities','PavedDrive']

for cf in onehotfeatures:
    onehot_df = oneHotEncoding(dataset,cf)
    dataset = dataset.join(onehot_df) 


data2 = dataset[all_features]

#Spliting the dataset into training and test data
from sklearn.model_selection import train_test_split

X = data2.loc[:, data2.columns != 'SalePrice']
y = data2[['SalePrice']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 0)


numeric_features.remove('SalePrice')

#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = X_train[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
X_train[skewed] = np.log1p(X_train[skewed])
X_test[skewed] = np.log1p(X_test[skewed])

#transform the target variable
y_train = np.log(y_train)
y_test = np.log(y_test)
#check if any columns is missing any values 
col_miss_values=list(data2[numeric_features].columns[data2[numeric_features].isnull().any()])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(X_train[numeric_features])
scaled = sc.transform(X_train[numeric_features])

for i, col in enumerate(numeric_features):
       X_train[col] = scaled[:,i]


scaled = sc.fit_transform(X_test[numeric_features])

for i, col in enumerate(numeric_features):
      X_test[col] = scaled[:,i]
      
      
#PCA========================
       
X_train = X_train[all_features]  
from sklearn.decomposition import PCA
pca= PCA(n_components=100)   
X_train= pca.fit_transform(X_train)
X_test= pca.fit_transform(X_test[all_features])
explainedVarience=pca.explained_variance_ratio_


## Run models and evaluate ##########################################################

def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
#lasso Regression
from sklearn.linear_model import Lasso

#found this best alpha through cross-validation
best_alpha = 0.00099

lassoclassifier = Lasso(alpha=best_alpha, max_iter=50000)
lassoclassifier.fit(X_train, y_train)

#run the prediction on the training set to get an idea of the accuracy
y_pred = lassoclassifier.predict(X_train)
print  ("Lasso RMSE score on the training data: ", rmse(y_train, y_pred))

#Now predict the sales value on the test set
y_pred = lassoclassifier.predict(X_test)
print  ("Lasso RMSE score on the test data: ", rmse(y_test, y_pred))

#XGBOOST
import xgboost as xgb
xgbclassifier = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

xgbclassifier.fit(X_train, y_train)

y_pred = xgbclassifier.predict(X_train)
print  ("XGBOOST RMSE score on the training data: ", rmse(y_train, y_pred))


y_pred = xgbclassifier.predict(X_test)
print  ("XGBOOST RMSE score on the test data: ", rmse(y_test, y_pred))

##Random Forest
from sklearn.ensemble import RandomForestRegressor
rfclassifier = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4)
rfclassifier.fit(X_train, y_train)

y_pred = rfclassifier.predict(X_train)
print  ("Random Forest RMSE score on the training data: ", rmse(y_train, y_pred))


y_pred = rfclassifier.predict(X_test)
print  ("Random Forest RMSE score on the test data: ", rmse(y_test, y_pred))
    
##svm
from sklearn.svm import SVR
svmclassifier = SVR(kernel='rbf')
svmclassifier.fit(X_train, y_train)

y_pred = svmclassifier.predict(X_train)
print  ("SVM score on the training data: ", rmse(y_train, y_pred))


y_pred = svmclassifier.predict(X_test)
print  ("SVM score on the test data: ", rmse(y_test, y_pred))
    


## Findings ###############################################
"""
We used four different models in this project and used Root Mean Square Error(RMSE) to measure the performance.
We first trained the dataset using those model and run test on the training data just to get an idea how the algorithm 
worked. Then we applied the model to the test data and calculated the RMSE. In the following we will share the test results of each model and 
see which model works best for this dataset.
 
1. Lasso RMSE score on the training data:  0.11129651807124671
   Lasso RMSE score on the test data:  0.2370552270655918
   
2. XGBOOST score on the training data:  0.02846106462105381
   XGBOOST score on the test data:  0.24029542986729605
   
3. Random Forest score on the training data:  0.2099405255972305
   Random Forest score on the test data:  0.20091598600452215
   
4. SVM score on the training data:  0.07815520474169338
   SVM score on the test data:  0.26740156770292517

After analyzing the score we find that XGBOOST and SVM model worked very well in the training data but the RMSE is 
not very good for test data. Random Forest model has the lowest RMSE and it performed similar to both training and test dataset.
   
"""











