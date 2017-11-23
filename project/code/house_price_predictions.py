#************************************************************
# module: house_price_predictions - kaggle data sets
#***********************************************************

#Loading libraries 
import os
import numpy as np 
import pandas as pd

#local libraries
import analysis as exploratory

import xgboost_algo as xgb
import ridge_algo as ridge
import lasso_algo as lasso
import neural_network_algo as nn
import random_forest_algo as random_forest
import svm_algo as svm


#os.system('cd /home/mcheruvu/i523/project/code')

print(os.getcwd())
                       
#****************************************************************
# method: load_datasets(path)
# purpose: load training and test house pricing data sets
#*****************************************************************
def load_datasets():
    #loading data
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))    
    print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))

    train["is_train"] = 1
    test["is_train"] = 0
   
    #combine the data set
    alldata = train.append(test)
    alldata.shape
    alldata.info()
    
    return alldata
#end load_datasets
    
#****************************************************************
# method: feature_engineering(df)
# purpose: add/remove features
#*****************************************************************
def feature_engineering(df):
    print('hello')
#end feature_engineering
    
#****************************************************************
# method: save_predicted_data(id_vector, sale_price_vector, save_as)
# purpose: save predicted data set for kaggle submission
#*****************************************************************
def save_predicted_data(id_vector, sale_price_vector, save_as):
    
    df_predict = pd.DataFrame({'Id': id_vector, 'SalePrice': np.exp(sale_price_vector)})
    
    df_predict.to_csv('../data/' + save_as + ".csv", header=True, index=False)

#end save_predicted_data


#*****************************************************************
# kaggle house pricing prediction challenge
#******************************************************************

df = load_datasets()

num_variables = [f for f in df.columns if df.dtypes[f] != 'object']
num_variables.remove('Id')
num_variables.remove('is_train')

cat_variables = [f for f in df.columns if df.dtypes[f] == 'object']

target_col = 'SalePrice'
row_id_col = 'Id'

exploratory.explore_and_preprocess(df, num_variables, cat_variables, target_col, row_id_col)

train = df[df["is_train"] == 1]
train_sale_price = train['SalePrice']
test = df[df["is_train"] == 0]

train = train[num_variables]
test = test[num_variables]

#************************************************************************
# let us do feature engineering - add new features and remove unwanted features
#**************************************************************************

feature_engineering(df)

#****************************************************************************
# run various algorithms to get different Sale Price predictions
#**************************************************************************

#XGBoosting algorithm
xgb_prediction = xgb.train_and_predict(train, train_sale_price, test)
#submit to kaggle
save_predicted_data(test['Id'], xgb_prediction, "kaggle_xgb_submission")


#Ridge algorithm
ridge_prediction = ridge.train_and_predict(train, train_sale_price, test)
#submit to kaggle
save_predicted_data(test['Id'], ridge_prediction, "kaggle_ridge_submission")


#Lasso algorithm
lasso_prediction = lasso.train_and_predict(train, train_sale_price, test)
#submit to kaggle
save_predicted_data(test['Id'], lasso_prediction, "kaggle_lasso_submission")


#random forest algorithm
random_forest_prediction = random_forest.train_and_predict(train, train_sale_price, test)
#submit to kaggle
save_predicted_data(test['Id'], random_forest_prediction, "kaggle_random_forest_submission")


#neural network algorithm
nn_prediction = nn.train_and_predict(train, train_sale_price, test)
#submit to kaggle
save_predicted_data(test['Id'], nn_prediction, "kaggle_neural_network_submission")


#SVM algorithm
svm_prediction = xgb.train_and_predict(train, train_sale_price, test)
#submit to kaggle
save_predicted_data(test['Id'], svm_prediction, "kaggle_svm_submission")

#ensemble best 3 algorithms
ensemble_prediction = (xgb_prediction + lasso_prediction + nn_prediction) / 3
#submit to kaggle
save_predicted_data(test['Id'], ensemble_prediction, "kaggle_ensemble_submission")

#****************************************************************
# method: munge_onehot(df)
# purpose: 
#*****************************************************************
def munge_onehot(df):

       onehot_df = pd.DataFrame(index = df.index)

       onehot_df = onehot(onehot_df, df, "MSSubClass", None)
       onehot_df = onehot(onehot_df, df, "MSZoning", "RL")
       onehot_df = onehot(onehot_df, df, "LotConfig", None)
       onehot_df = onehot(onehot_df, df, "Neighborhood", None)
       onehot_df = onehot(onehot_df, df, "Condition1", None)
       onehot_df = onehot(onehot_df, df, "BldgType", None)
       onehot_df = onehot(onehot_df, df, "HouseStyle", None)
       onehot_df = onehot(onehot_df, df, "RoofStyle", None)
       onehot_df = onehot(onehot_df, df, "Exterior1st", "VinylSd")
       onehot_df = onehot(onehot_df, df, "Exterior2nd", "VinylSd")
       onehot_df = onehot(onehot_df, df, "Foundation", None)
       onehot_df = onehot(onehot_df, df, "SaleType", "WD")
       onehot_df = onehot(onehot_df, df, "SaleCondition", "Normal")

       #Fill in missing MasVnrType for rows that do have a MasVnrArea.
       temp_df = df[["MasVnrType", "MasVnrArea"]].copy()
       idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))
       temp_df.loc[idx, "MasVnrType"] = "BrkFace"
       onehot_df = onehot(onehot_df, temp_df, "MasVnrType", "None")

       onehot_df = onehot(onehot_df, df, "LotShape", None)
       onehot_df = onehot(onehot_df, df, "LandContour", None)
       onehot_df = onehot(onehot_df, df, "LandSlope", None)
       onehot_df = onehot(onehot_df, df, "Electrical", "SBrkr")
       onehot_df = onehot(onehot_df, df, "GarageType", "None")
       onehot_df = onehot(onehot_df, df, "PavedDrive", None)
       onehot_df = onehot(onehot_df, df, "MiscFeature", "None")
       onehot_df = onehot(onehot_df, df, "Street", None)
       onehot_df = onehot(onehot_df, df, "Alley", "None")
       onehot_df = onehot(onehot_df, df, "Condition2", None)
       onehot_df = onehot(onehot_df, df, "RoofMatl", None)
       onehot_df = onehot(onehot_df, df, "Heating", None)

       # we'll have these as numerical variables too
       onehot_df = onehot(onehot_df, df, "ExterQual", "None")
       onehot_df = onehot(onehot_df, df, "ExterCond", "None")
       onehot_df = onehot(onehot_df, df, "BsmtQual", "None")
       onehot_df = onehot(onehot_df, df, "BsmtCond", "None")
       onehot_df = onehot(onehot_df, df, "HeatingQC", "None")
       onehot_df = onehot(onehot_df, df, "KitchenQual", "TA")
       onehot_df = onehot(onehot_df, df, "FireplaceQu", "None")
       onehot_df = onehot(onehot_df, df, "GarageQual", "None")
       onehot_df = onehot(onehot_df, df, "GarageCond", "None")
       onehot_df = onehot(onehot_df, df, "PoolQC", "None")
       onehot_df = onehot(onehot_df, df, "BsmtExposure", "None")
       onehot_df = onehot(onehot_df, df, "BsmtFinType1", "None")
       onehot_df = onehot(onehot_df, df, "BsmtFinType2", "None")
       onehot_df = onehot(onehot_df, df, "Functional", "Typ")
       onehot_df = onehot(onehot_df, df, "GarageFinish", "None")
       onehot_df = onehot(onehot_df, df, "Fence", "None")
       onehot_df = onehot(onehot_df, df, "MoSold", None)

       # Divide  the years between 1871 and 2010 into slices of 20 years
       year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20))  for i in range(0, 7))
       yearbin_df = pd.DataFrame(index = df.index)
       yearbin_df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)
       yearbin_df["GarageYrBltBin"].fillna("NoGarage", inplace=True)
       yearbin_df["YearBuiltBin"] = df.YearBuilt.map(year_map)
       yearbin_df["YearRemodAddBin"] = df.YearRemodAdd.map(year_map)

       onehot_df = onehot(onehot_df, yearbin_df, "GarageYrBltBin", None)
       onehot_df = onehot(onehot_df, yearbin_df, "YearBuiltBin", None)
       onehot_df = onehot(onehot_df, yearbin_df, "YearRemodAddBin", None)

       return onehot_df

#end munge_onehot





