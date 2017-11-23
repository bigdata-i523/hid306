#******************************************************************
# module: XGB Boost - supervised machine learning algorthim
#*******************************************************************

#Loading libraries
import numpy as np 
import xgboost as xgb
import analysis

np.random.seed(1234)

_xgb_algo = xgb.XGBRegressor(colsample_bytree=0.2,
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
                       
#****************************************************************
# method: _train(train_data, train_target_vector)
# purpose: 
#*****************************************************************
def _train(train_data, train_target_vector):

   _xgb_algo.fit(train_data, train_target_vector)

#end train

#****************************************************************
# method: _predict(train_data, train_target_vector, test_data)
# purpose: 
#*****************************************************************
def _predict(train_data, train_target_vector, test_data):

    y_train_pred = _xgb_algo.predict(train_data)
    y_train = train_target_vector
    
    rmse_train = analysis.rmse(y_train, y_train_pred)
    
    print("XGBoost score on training set: ", rmse_train)
   
    y_test_pred = _xgb_algo.predict(test_data)   
        
    return y_test_pred
    
#end predict
    
#****************************************************************
# method: train_and_predict(train_data, train_target_vector, test_data)
# purpose: 
#*****************************************************************
def train_and_predict(train_data, train_target_vector, test_data):

   _train(train_data, train_target_vector)
   
   y_test_predict = _predict(train_data, train_target_vector, test_data)
   
   return y_test_predict

#end train_and_predict