#******************************************************************
# module: Lasso - supervised machine learning algorthim
#*******************************************************************

#Loading libraries
import numpy as np 
from sklearn.linear_model import Lasso
import analysis

np.random.seed(1234)

#found this best alpha value through cross-validation
_best_alpha = 0.00099

_lasso_algo = Lasso(alpha = _best_alpha, max_iter = 50000)
                       
#****************************************************************
# method: _train(train_data, train_target_vector)
# purpose: 
#*****************************************************************
def _train(train_data, train_target_vector):

   _lasso_algo.fit(train_data, train_target_vector)

#end train

#****************************************************************
# method: _predict(train_data, train_target_vector, test_data)
# purpose: 
#*****************************************************************
def _predict(train_data, train_target_vector, test_data):

    y_train_pred = _lasso_algo.predict(train_data)
    y_train = train_target_vector
    
    rmse_train = analysis.rmse(y_train, y_train_pred)
    
    print("Lasso score on training set: ", rmse_train)
   
    y_test_pred = _lasso_algo.predict(test_data)   
        
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