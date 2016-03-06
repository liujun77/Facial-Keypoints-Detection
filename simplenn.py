# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:18:30 2016

@author: liujun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.utils import shuffle

#%%

TRAINF = 'training.csv'
TESTF = 'test.csv'

def load(test = False, cols = None):
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
    return X, y
    
#%%
    X ,y =load()
#%%
    print("X.shape == {}; X.min == {}; X.max == {}".format(
    X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {}; y.max == {}".format(
    y.shape, y.min(), y.max()))
#%%

import datetime as dt
import theano
import theano.tensor as T
import lasagne as lg
#%%
img_size = 96


#%%
def net_mlp(input_var = None):
    l_in = lg.layers.InputLayer(shape=(None, img_size*img_size),
                                     input_var=input_var)
    l_hid1 = lg.layers.DenseLayer(
            l_in, num_units=100,
            nonlinearity=lg.nonlinearities.rectify,
            W=lg.init.GlorotUniform())
    l_out = lg.layers.DenseLayer(
            l_hid1, num_units=30,
            nonlinearity=None)
    return l_out

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
input_var = T.matrix('in')
target_var = T.matrix('out')

network = net_mlp(input_var)
prediction = lg.layers.get_output(network)
loss = lg.objectives.squared_error(prediction, target_var)
loss = loss.mean()
params = lg.layers.get_all_params(network, trainable=True)
updates = lg.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

valid_prediction = lg.layers.get_output(network)
valid_loss = lg.objectives.squared_error(valid_prediction, target_var)
valid_loss = valid_loss.mean()

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [valid_loss, valid_prediction])

#%%
train_size = 1700#int(X.shape[0]*0.8)
valid_size = 400#X.shape[0] - train_size
train_X = X[:train_size]
train_y = y[:train_size]
valid_X = X[train_size:]
valid_y = y[train_size:]

#%%
train_loss = []
valid_loss = []
num_epochs = 400

for epoch in range(num_epochs):
    train_err = 0
    train_batches = 0
    start_time = dt.datetime.now()
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
    
    #print("Epoch {} of {} took {}".format(
    #        epoch + 1, num_epochs, dt.datetime.now() - start_time))
    #print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    #print("  valid    loss:\t\t{:.6f}".format(valid_err / valid_batches))
    #print('train_batches = {}, valid_batches = {}'.format(train_batches, valid_batches))
    train_loss.append(train_err/train_batches)
    valid_loss.append(valid_err/valid_batches)

#%%

n_epochs = range(1,num_epochs+1)
plt.plot(n_epochs, train_loss)
plt.plot(n_epochs, valid_loss)
plt.ylim(1e-3, 1e-2)
plt.yscale("log")
plt.show()

#%%
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

#%%

test_X, _ = load(test=True)

#%%
test_batch_X = test_X[0:50]
#%%
_, test_y = val_fn(test_batch_X, valid_y[0:50])

#%%
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(test_batch_X[i], test_y[i], ax)

plt.show()













