# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:03:04 2016

@author: liujun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.utils import shuffle
import time
import theano
import theano.tensor as T
import lasagne as lg
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = lg.layers.Conv2DLayer
    MaxPool2DLayer = lg.layers.MaxPool2DLayer

#%%

TRAINF = 'training.csv'
TESTF = 'test.csv'

def load2d(test = False, cols = None):
    filename = TESTF if test else TRAINF
    df = pd.read_csv(filename)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if cols:
        df = df[list(cols) + ['Image']]
    print(df.count())
    df = df.dropna()
    X = np.vstack(df['Image'].values).astype(np.float32) / 255.0
    if not test:
        y = df[df.columns[:-1]].values.astype(np.float32)
        y = (y - 48.0) / 48.0
        X, y = shuffle(X, y, random_state = 0)
    else:
        y = None
    X = X.reshape(-1, 1, 96, 96)
    return X, y
    
#%%
class colors:  
    BLACK         = '\033[0;30m'  
    DARK_GRAY     = '\033[1;30m'  
    LIGHT_GRAY    = '\033[0;37m'  
    BLUE          = '\033[0;34m'  
    LIGHT_BLUE    = '\033[1;34m'  
    GREEN         = '\033[0;32m'  
    LIGHT_GREEN   = '\033[1;32m'  
    CYAN          = '\033[0;36m'  
    LIGHT_CYAN    = '\033[1;36m'  
    RED           = '\033[0;31m'  
    LIGHT_RED     = '\033[1;31m'  
    PURPLE        = '\033[0;35m'  
    LIGHT_PURPLE  = '\033[1;35m'  
    BROWN         = '\033[0;33m'  
    YELLOW        = '\033[1;33m'  
    WHITE         = '\033[1;37m'  
    DEFAULT_COLOR = '\033[00m'  
    RED_BOLD      = '\033[01;31m'  
    ENDC          = '\033[0m'  
#%%
import cPickle as pickle 
import os

def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lg.layers.get_all_param_values(model)
    with open(filename, 'w') as f:
        pickle.dump(data, f)     

def write_data(data, filename):
    """Pickels the parameters within a Lasagne model."""
    with open(filename, 'w') as f:
        pickle.dump(data, f)    

def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    if not os.path.exists(filename):
        print("{} not exists".format(filename))
        return
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lg.layers.set_all_param_values(model, data)
     
def read_data(filename):
    """Unpickles and loads parameters into a Lasagne model."""
    if not os.path.exists(filename):
        print("{} not exists".format(filename))
        return None
    with open(filename, 'r') as f:
        data = pickle.load(f)
    return data
        
#%%
img_size = 96

def net_cnn(input_var = None):
    network = lg.layers.InputLayer(shape=(None, 1, img_size, img_size),
                                        input_var=input_var)
    network = Conv2DLayer(
            network, 
            num_filters=32, 
            filter_size=(5, 5), 
            pad='same', 
            nonlinearity=lg.nonlinearities.rectify,
            W=lg.init.GlorotUniform())
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    network = Conv2DLayer(
            lg.layers.dropout(network, p=.1), 
            num_filters=64, 
            filter_size=(3, 3),
            pad='same', 
            nonlinearity=lg.nonlinearities.rectify)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    network = Conv2DLayer(
            lg.layers.dropout(network, p=.2), 
            num_filters=128, 
            filter_size=(3, 3),
            pad='same', 
            nonlinearity=lg.nonlinearities.rectify)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    network = lg.layers.DenseLayer(
            lg.layers.dropout(network, p=.3),
            num_units=500,
            nonlinearity=lg.nonlinearities.rectify)
    network = lg.layers.DenseLayer(
            lg.layers.dropout(network, p=.5),
            num_units=500,
            nonlinearity=lg.nonlinearities.rectify)
    network = lg.layers.DenseLayer(
            network,
            num_units=30,
            nonlinearity=None)
    return network

#%%

def iter_batch(X , y, batch_size, shuffle = False):
    if shuffle:
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
    for i in range(0, X.shape[0]-batch_size+1, batch_size):
        if shuffle:
            excerpt = index[i : i+batch_size]
        else:
            excerpt = slice(i, i+batch_size)
        yield X[excerpt], y[excerpt]
#%%
# graph
input_var = T.tensor4('in')
target_var = T.matrix('out')
learning_rate = theano.shared(np.array(0.1, dtype=theano.config.floatX))

print("Building model and compiling functions...")
network = net_cnn(input_var)
prediction = lg.layers.get_output(network)
loss = lg.objectives.squared_error(prediction, target_var)
loss = loss.mean()
params = lg.layers.get_all_params(network, trainable=True)
updates = lg.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

valid_prediction = lg.layers.get_output(network, deterministic=True)
valid_loss = lg.objectives.squared_error(valid_prediction, target_var)
valid_loss = valid_loss.mean()

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [valid_loss, valid_prediction])

#%%
print("Loading data...")
X ,y =load2d()

print("X.shape == {}; X.min == {}; X.max == {}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {}; y.max == {}".format(
    y.shape, y.min(), y.max()))

train_size = int(X.shape[0]*0.8)
valid_size = X.shape[0] - train_size
train_X = X[:train_size]
train_y = y[:train_size]
valid_X = X[train_size:]
valid_y = y[train_size:]

#%%
num_epochs = 3000
decay_rate = np.array(0.95, dtype=theano.config.floatX)
stop_rate = 0.001

read_model_data(network,'para.pickle')
train_loss = read_data('train_loss.pickle')
valid_loss = read_data('valid_loss.pickle')
if train_loss is None:
    train_loss = [] 
if valid_loss is None:
    valid_loss = [] 

start_epoch = len(train_loss)
#%%
print("Starting training from epoch {}...".format(start_epoch+1))
print("epoch \t| train_loss \t| valid_loss \t| time \t\t|")
print("---------------------------------------------------------")
for epoch in range(start_epoch, num_epochs):
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iter_batch(train_X, train_y, 50, shuffle=True):
        batch_X, batch_y = batch
        train_err += train_fn(batch_X, batch_y)
        train_batches +=1

    valid_err = 0
    valid_batches = 0    
    for batch in iter_batch(valid_X, valid_y, 50, shuffle=False):
        batch_X, batch_y = batch
        dloss, _ = val_fn(batch_X, batch_y)
        valid_err += dloss
        valid_batches +=1        
    
    if (epoch+1)%50==0:
        print("{} \t| {:.6f} \t| {:.6f} \t| {:.6f} s \t|".format(
            epoch+1, 
            train_err/train_batches,
            valid_err/valid_batches,
            time.time() - start_time))
        #print("  learning rate:\t\t{}".format(learning_rate.get_value()))
    
    if epoch%50==0 and learning_rate.get_value()>stop_rate:
        learning_rate.set_value(learning_rate.get_value()*decay_rate)
        
    if epoch>100 and valid_loss[epoch-100]<valid_err/valid_batches:
        break
    train_loss.append(train_err/train_batches)
    valid_loss.append(valid_err/valid_batches)
    if (epoch+1)%50==0:
        write_model_data(network, 'para.pickle')
        write_data(train_loss, 'train_loss.pickle')
        write_data(valid_loss, 'valid_loss.pickle')
    
    
#%%
write_model_data(network, 'para.pickle')
write_data(train_loss, 'train_loss.pickle')
write_data(valid_loss, 'valid_loss.pickle')
#%%
#import sys
#sys.exit()

#%%
train_loss = read_data('train_loss.pickle')
valid_loss = read_data('valid_loss.pickle')
n_epochs = range(1,len(train_loss)+1)
plt.plot(n_epochs, train_loss)
plt.plot(n_epochs, valid_loss)
plt.ylim(4e-4, 1e-2)
plt.yscale("log")
plt.show()

#%%
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

#%%

test_X, _ = load2d(test=True)

#%%
test_batch_X = test_X[0:50]
#%%
read_model_data(network,'para.pickle')
_, test_y = val_fn(test_batch_X, valid_y[0:50])

#%%
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1.5, bottom=0, top=1.5, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(test_batch_X[i], test_y[i], ax)

plt.show()