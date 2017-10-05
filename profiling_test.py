import pandas as pd
import numpy as np
import tensorflow_example as te

train = pd.read_csv('input/processed_train.csv')
test = pd.read_csv('input/processed_test.csv')

train_x = train[[col for col in train.columns if col != 'SalePrice']]
train_y = np.log(train['SalePrice']).values.reshape(train_x.shape[0], 1)

train_length = int(len(train_x)*.9)
valid_x = train_x.iloc[train_length:,:]
valid_y = train_y[train_length:,:]

train_x = train_x.iloc[:train_length,:]
train_y = train_y[:train_length,:]

train_x = train_x.T
train_y = train_y.T
valid_x = valid_x.T
valid_y = valid_y.T

test_x = test.T

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape)

parameters = te.model(train_x, train_y, 
                   valid_x, valid_y,
                   test_x,
                   learning_rate=0.01,
                   lr_decay_step = 10000,
                   lr_decay_rate = 0.95,
                   bn_decay_rate = 0.9,
                   num_epochs=1, 
                   minibatch_size = 64,
                   layer_count = 8, 
                   hidden_neuron = [100, 100,  100,  100,  100,  100,  100,  100],
                   profiling = True)