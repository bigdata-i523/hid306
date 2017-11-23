#******************************************************************
# module: Random Forest - supervised machine learning algorthim
#*******************************************************************

#Loading libraries
import numpy as np 
import analysis

np.random.seed(1234)
                       
#****************************************************************
# method: _train(train_data, train_target_vector)
# purpose: 
#*****************************************************************
def _train(train_data, train_target_vector):

   print('heelo')

#end train

#****************************************************************
# method: _predict(train_data, train_target_vector, test_data)
# purpose: 
#*****************************************************************
def _predict(train_data, train_target_vector, test_data):

    print('heelo') 
    y_test_pred = ''
        
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