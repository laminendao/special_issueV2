# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:08:08 2024

@author: mlndao
"""
from __future__ import print_function

import tensorflow as tf
# from keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


learning_rate_ = 0.001
# input_shape = (sequence_length, train_array.shape[2])
def create_lstm_1layer(dropout, activation, weights_file, input_shape):
    # history = tf.keras.callbacks.History()
    model = Sequential()
    model.add(LSTM(units=64, activation='tanh',input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse',metrics=['mse'], optimizer= 'adam') # A voir si varier ou pas s-score, rmse
    model.save_weights(weights_file)
    
    return model

def create_lstm_2layers(dropout, activation, weights_file, input_shape):
    # history = tf.keras.callbacks.History()
    model = Sequential()
    model.add(LSTM(units=64, activation='tanh',return_sequences=True,input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=32, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
#     model.save_weights(weights_file) 
    return model 

def create_lstm_3layers(dropout, activation, weights_file, input_shape):
#     history = tf.keras.callbacks.History()
    model = Sequential()
    model.add(LSTM(units=64, activation='tanh',return_sequences=True,input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=32, activation='tanh')) # A reduire et pourquoi
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.save_weights(weights_file)
    return model 

def create_lstm_4layers(dropout, activation, weights_file, input_shape):
#     history = tf.keras.callbacks.History()
    model = Sequential()
    model.add(LSTM(units=64, activation='tanh',return_sequences=True,input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=32, activation='tanh')) # A reduire et pourquoi
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.save_weights(weights_file)
    
    return model

def create_lstm_5layers(dropout, activation, weights_file, input_shape):
#     history = tf.keras.callbacks.History()
    model = Sequential()
    model.add(LSTM(units=64, activation='tanh',return_sequences=True,input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=32, activation='tanh')) # A reduire et pourquoi
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.save_weights(weights_file)
    
    return model

def model001(input_shape):

    nodes_per_layer = 256
    activation_value= 'tanh'
    dropout = 0.3
    bs = 64

    cb = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    model = Sequential()
    model.add(LSTM(nodes_per_layer, activation=activation_value, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=0.001))
    
    return model

def model002(input_shape):
    '''
    node = 64, activation = tanh, dropout = 0.2, bs = 128

    '''
    nodes_per_layer = 64
    activation_value= 'tanh'
    dropout = 0.2
    bs = 128

    cb = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    model = Sequential()
    model.add(LSTM(nodes_per_layer, activation=activation_value, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=0.001))
    
    return model

def model003(input_shape):
    '''
    node = 32, activation = tanh, dropout = 0.4, bs = 64
    '''
    nodes_per_layer = 32
    activation_value= 'tanh'
    dropout = 0.4
    bs = 64

    cb = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    model = Sequential()
    model.add(LSTM(nodes_per_layer, activation=activation_value, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=0.001))
    
    return model

def model004(input_shape):
    '''
    node = 256, activation = tanh, dropout = 0.3, bs = 64
    '''
    nodes_per_layer = 256
    activation_value= 'tanh'
    dropout = 0.3
    bs = 64

    cb = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    model = Sequential()
    model.add(LSTM(nodes_per_layer, activation=activation_value, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=0.001))
    
    return model

if __name__ == '__main__':
    1+1
    