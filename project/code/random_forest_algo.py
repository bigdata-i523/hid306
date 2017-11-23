#******************************************************************
# module: Random Forest - supervised machine learning algorthim
#*******************************************************************

#Loading libraries
import numpy as np 
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import analysis

np.random.seed(1234)
                       
#****************************************************************
# method: _train(train_data, train_target_vector)
# purpose: 
#*****************************************************************
def _train(train_data, train_target_vector, max_features = 15):
  
   X = train_data
   Y = train_target_vector
   seed = 7
   num_trees = 100
      
   kfold = model_selection.KFold(n_splits=10, random_state=seed)
   
   #model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
   model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
   
   results = model_selection.cross_val_score(model, X, Y, cv=kfold)
   
   print(results.mean())

#end train

#****************************************************************
# method: _predict(train_data, train_target_vector, test_data)
# purpose: 
#*****************************************************************
def _predict(train_data, train_target_vector, test_data):
    
    y_test_pred = ''
        
    return y_test_pred
    
#end predict
    
#****************************************************************
# method: _feature_ranking(train_data, train_target_vector)
# purpose: 
#*****************************************************************
def _feature_ranking(train_data, train_target_vector):

    print('hello')    
    
#end _feature_ranking
    
#****************************************************************
# method: _cross_validation(train_data, train_target_vector)
# purpose: 
#*****************************************************************
def _cross_validation(train_data, train_target_vector):

    print('hello')    
    
#end _cross_validation
    
    
#****************************************************************
# method: train_and_predict(train_data, train_target_vector, test_data)
# purpose: 
#*****************************************************************
def train_and_predict(train_data, train_target_vector, test_data):

   _train(train_data, train_target_vector)
   
   y_test_predict = _predict(train_data, train_target_vector, test_data)
   
   return y_test_predict

#end train_and_predict