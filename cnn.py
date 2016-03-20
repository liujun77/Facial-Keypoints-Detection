# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:03:04 2016

@author: liujun
"""

import sys
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
    print('import cuda support lib error!')
    Conv2DLayer = lg.layers.Conv2DLayer
    MaxPool2DLayer = lg.layers.MaxPool2DLayer

#%%

TRAINF = 'training.csv'
TESTF = 'test.csv'

def load2d(test = False, cols = None):
    filename = TESTF if test else TRAINF
    df = pd.read_csv(filename)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if test==False and len(cols)>0:
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

models = [['left_eye_center_x', 'left_eye_center_y',
          'right_eye_center_x', 'right_eye_center_y'],
         ['left_eye_inner_corner_x', 'left_eye_inner_corner_y', 
          'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
          'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
          'right_eye_outer_corner_x', 'right_eye_outer_corner_y'],
         ['left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
          'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
          'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
          'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y'],
         ['nose_tip_x', 'nose_tip_y'],
         ['mouth_left_corner_x', 'mouth_left_corner_y',
          'mouth_right_corner_x', 'mouth_right_corner_y',
          'mouth_center_top_lip_x', 'mouth_center_top_lip_y'],
         ['mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']]

SPECIALIST={'left_eye_center_x':0, 'left_eye_center_y':1,
            'right_eye_center_x':2, 'right_eye_center_y':3,
            'left_eye_inner_corner_x':4, 'left_eye_inner_corner_y':5, 
            'left_eye_outer_corner_x':6, 'left_eye_outer_corner_y':7,
            'right_eye_inner_corner_x':8, 'right_eye_inner_corner_y':9,
            'right_eye_outer_corner_x':10, 'right_eye_outer_corner_y':11,
            'left_eyebrow_inner_end_x':12, 'left_eyebrow_inner_end_y':13,
            'left_eyebrow_outer_end_x':14, 'left_eyebrow_outer_end_y':15,
            'right_eyebrow_inner_end_x':16, 'right_eyebrow_inner_end_y':17,
            'right_eyebrow_outer_end_x':18, 'right_eyebrow_outer_end_y':19,
            'nose_tip_x':20, 'nose_tip_y':21,
            'mouth_left_corner_x':22, 'mouth_left_corner_y':23,
            'mouth_right_corner_x':24, 'mouth_right_corner_y':25,
            'mouth_center_top_lip_x':26, 'mouth_center_top_lip_y':27,
            'mouth_center_bottom_lip_x':28, 'mouth_center_bottom_lip_y':29} 
            
para_files = ['m0.pickle','m1.pickle','m2.pickle','m3.pickle','m4.pickle','m5.pickle']
#%%

cols = models[1]
para_file = para_files[1]

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
    if cols==[] or len(cols)==data[11].shape[0]:
        lg.layers.set_all_param_values(model, data)
    else:
        data[10]=data[10][:,SPECIALIST[cols[0]]:SPECIALIST[cols[-1]]+1]
        data[11]=data[11][SPECIALIST[cols[0]]:SPECIALIST[cols[-1]]+1]
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

def net_cnn(input_var = None, num_outputs=30):
    network = lg.layers.InputLayer(shape=(None, 1, img_size, img_size),
                                        input_var=input_var)
    network = Conv2DLayer(
            network, 
            num_filters=32, 
            filter_size=(3, 3), 
            #pad='same', 
            nonlinearity=lg.nonlinearities.rectify,
            W=lg.init.GlorotUniform())
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    network = Conv2DLayer(
            lg.layers.dropout(network, p=.1), 
            num_filters=64, 
            filter_size=(2, 2),
            #pad='same', 
            nonlinearity=lg.nonlinearities.rectify)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    network = Conv2DLayer(
            lg.layers.dropout(network, p=.2), 
            num_filters=128, 
            filter_size=(2, 2),
            #pad='same', 
            nonlinearity=lg.nonlinearities.rectify)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    network = lg.layers.DenseLayer(
            lg.layers.dropout(network, p=.3),
            num_units=1000,
            nonlinearity=lg.nonlinearities.rectify)
    network = lg.layers.DenseLayer(
            lg.layers.dropout(network, p=.5),
            num_units=1000,
            nonlinearity=lg.nonlinearities.rectify)
    network = lg.layers.DenseLayer(
            network,
            num_units=num_outputs,
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

flip_ind = {'para.pickle':[
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25)],
        'm0.pickle':[(0, 2), (1, 3)],
        'm1.pickle':[(0, 4), (1, 5), (2, 6), (3, 7)],
        'm2.pickle':[(0, 4), (1, 5), (2, 6), (3, 7)],
        'm3.pickle':[],
        'm4.pickle':[(0, 2), (1, 3)],
        'm5.pickle':[]}

def rnd_flip(X, y):
    flip_indices = flip_ind[para_file]
    index = np.random.choice(X.shape[0], X.shape[0] // 2 , replace=False)
    #print(index)
    #plt.imshow(X[index[0]].reshape(96,96),cmap='gray')
    #plt.show()
    X[index] = X[index, :, :, ::-1]
    #plt.imshow(X[index[0]].reshape(96,96),cmap='gray')
    #plt.show()
    
    if y is not None:
        y[index, ::2] = y[index, ::2]*(-1)
        for a, b in flip_indices:
            y[index, a], y[index, b] = (y[index, b], y[index, a])
    return X, y
    
def float32(k):
    return np.cast['float32'](k)    

#%%
# graph
input_var = T.tensor4('in')
target_var = T.matrix('out')
learning_rate = theano.shared(float32(0.03))
momentum = theano.shared(float32(0.9))

if len(cols)==0:
    num_outputs = 30;
else:
    num_outputs = len(cols)

print("Building model and compiling functions...")
network = net_cnn(input_var, num_outputs=num_outputs)
prediction = lg.layers.get_output(network)
loss = lg.objectives.squared_error(prediction, target_var)
loss = loss.mean()
params = lg.layers.get_all_params(network, trainable=True)
updates = lg.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

valid_prediction = lg.layers.get_output(network, deterministic=True)
valid_loss = lg.objectives.squared_error(valid_prediction, target_var)
valid_loss = valid_loss.mean()

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [valid_loss, valid_prediction])

#%%

def fit(X, y,
        num_epochs=3000, 
        batch_size=50, 
        start_rate=0.03,
        stop_rate=0.0001,
        start_momentum=0.9,
        stop_momentum = 0.999,
        plot_steps=10,
        save_steps=50,
        para_file='para.pickle',
        train_loss_file='train_loss.pickle',
        valid_loss_file='valid_loss.pickle',):
            
    rates = np.linspace(start_rate, stop_rate, num_epochs)
    momentums = np.linspace(start_momentum, stop_momentum, num_epochs)
    train_size = int(X.shape[0]*0.8)
    valid_size = X.shape[0] - train_size
    train_X = X[:train_size]
    train_y = y[:train_size]
    valid_X = X[train_size:]
    valid_y = y[train_size:]
    read_model_data(network,para_file)
    train_loss = read_data(train_loss_file)
    valid_loss = read_data(valid_loss_file)
    min_loss = read_data('min_loss.pickle')    
    
    if train_loss is None:
        train_loss = [] 
    if valid_loss is None:
        valid_loss = [] 
    if min_loss is None:
        min_loss = {'index': -1, 'loss': 1e3}
    
    best_para = None    
    
    start_epoch = len(train_loss)
    print("Starting training from epoch {}...".format(start_epoch+1))
    print("epoch \t| train_loss \t| valid_loss \t| train/val \t | time \t\t|")
    print("-------------------------------------------------------------------------")
    for epoch in range(start_epoch, num_epochs):
        
        learning_rate.set_value(float32(rates[epoch]))
        momentum.set_value(float32(momentums[epoch]))
        train_err = 0
        start_time = time.time()
        for batch in iter_batch(train_X, train_y, batch_size, shuffle=True):
            batch_X, batch_y = batch
            batch_X, batch_y = rnd_flip(batch_X, batch_y)
            train_err += train_fn(batch_X, batch_y)*batch_X.shape[0]

        valid_err = 0
        for batch in iter_batch(valid_X, valid_y, batch_size, shuffle=False):
            batch_X, batch_y = batch
            dloss, _ = val_fn(batch_X, batch_y)
            valid_err += dloss*batch_X.shape[0]
        
        cur_train_loss = train_err/train_X.shape[0]
        cur_valid_loss = valid_err/valid_X.shape[0]
        if (epoch+1)%plot_steps==0:
            valid_print="{:.6f}".format(cur_valid_loss)
            if cur_valid_loss<min_loss['loss']:
                valid_print=colors.GREEN+valid_print+colors.ENDC
            else:
                valid_print=colors.RED+valid_print+colors.ENDC
            
            print("{} \t| {:.6f} \t| {:12} \t| {:.3f} \t| {:.3f} s \t|".format(
                epoch+1, 
                cur_train_loss,
                valid_print,
                cur_train_loss/cur_valid_loss,
                time.time() - start_time))
            #print("  learning rate:\t\t{}".format(learning_rate.get_value()))
        
        if min_loss['loss']>cur_valid_loss:
            min_loss['loss']=cur_valid_loss
            min_loss['index']=epoch
            best_para=lg.layers.get_all_param_values(network)
            
        if epoch-min_loss['index']>200:
            break
        train_loss.append(cur_train_loss)
        valid_loss.append(cur_valid_loss)
        if (epoch+1)%save_steps==0:
            write_model_data(network, para_file)
            write_data(train_loss, train_loss_file)
            write_data(valid_loss, valid_loss_file)
            write_data(min_loss, 'min_loss.pickle')
    write_data(best_para, para_file)
    write_data(train_loss, train_loss_file)
    write_data(valid_loss, valid_loss_file)

#%%

print("Loading data...")
X ,y =load2d(cols=cols)

print("X.shape == {}; X.min == {}; X.max == {}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {}; y.max == {}".format(
    y.shape, y.min(), y.max()))
#%%

if len(cols)>0:
    read_model_data(network,'para.pickle')

fit(X,y,
    num_epochs=10000, 
    plot_steps=1, 
    batch_size=128, 
    para_file=para_file,
    save_steps=50)

#%%
train_loss = read_data('train_loss.pickle')
valid_loss = read_data('valid_loss.pickle')
read_model_data(network,para_file)

#%%
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

test_X, _ = load2d(test=True,cols=cols)

#%%
test_batch_X = test_X[0:50]
_, test_y = val_fn(test_batch_X, y[0:50])

#%%
fig = plt.figure(figsize=(8, 24))
fig.subplots_adjust(
    left=0, right=1.5, bottom=0, top=1.5, hspace=0.05, wspace=0.05)

for i in range(48):
    ax = fig.add_subplot(12, 4, i + 1, xticks=[], yticks=[])
    plot_sample(test_batch_X[i], test_y[i], ax)
    #plot_sample(X[i], y[i], ax)

plt.show()
#%%

def predict(X):
    y = np.zeros((0, len(cols)))
    for batch in iter_batch(X, X, 50, shuffle=False):
        batch_X, _ = batch
        _, batch_y = val_fn(batch_X, y[0:50])
        y = np.vstack((y, batch_y))
    #y = np.c_(y, sub_y)
    return y
