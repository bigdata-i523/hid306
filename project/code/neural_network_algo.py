#******************************************************************
# module: Neural Network - supervised machine learning algorthim
#*******************************************************************

#Loading libraries
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.sckikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(1234)

#****************************************************************
# method: _model()
# purpose: 
#*****************************************************************
def _model():

  model = Sequential()
  model.add(Dense(20, input_dim = 398, init = 'normal', activation = 'relu'))
  model.add(Dense(10, init = 'normal', activation = 'relu'))
  model.add(Dense(1, init = 'normal'))
  
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  
  return model

#end _model

_neural_network_algo = KerasRegressor(build_fn = _model, nb_epoch = 1000, batch_size = 5, verbose = 0)
_scale = StandardScaler()

#****************************************************************
# method: _train(train_data, train_target_vector)
# purpose: 
#*****************************************************************
def _train(train_data, train_target_vector):
  
   x_train = _scale.fit_transform(train_data)
   
   keras_target = train_target_vector.as_matrix()
   
   _neural_network_algo.fit(x_train, keras_target)  

#end train

#****************************************************************
# method: _predict(train_data, train_target_vector, test_data)
# purpose: 
#*****************************************************************
def _predict(train_data, train_target_vector, test_data):

    x_test = _scale.fit_transform(test_data)
    
    y_test_pred = _neural_network_algo.predict(x_test)
    
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