# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:47:37 2023

@author: mlndao
"""
# package's importation

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random
import keras
import math
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import GroupKFold ,GroupShuffleSplit
from sklearn import preprocessing
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Masking
from scipy import optimize
import warnings
warnings.filterwarnings("ignore")

#%% Function making

# read the train and test data
def prepare_data(file_name):
    dir_path = r'data/' 
    dependent_var = ['RUL']
    index_names = ['Unit', 'Cycle']
    setting_names = ['Altitude', 'Mach', 'TRA']
    sensor_names = ['T20','T24','T30','T50','P20','P15','P30','Nf','Nc','epr','Ps30','phi',
                    'NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
    col_names = index_names + setting_names + sensor_names
    
    df_train = pd.read_csv(dir_path+'train_'+str(file_name),delim_whitespace=True,names=col_names)

    rul_train = pd.DataFrame(df_train.groupby('Unit')['Cycle'].max()).reset_index()
    rul_train.columns = ['Unit', 'max']
    df_train = df_train.merge(rul_train, on=['Unit'], how='left')
    df_train['RUL'] = df_train['max'] - df_train['Cycle']
    df_train.drop('max', axis=1, inplace=True)

    df_test = pd.read_csv(dir_path+'test_'+str(file_name), delim_whitespace=True, names=col_names)
    
    y_test = pd.read_csv(dir_path+'RUL_'+(file_name), delim_whitespace=True,names=["RUL"])
    #y_true["Unit"] = y_true.index + 1
    return df_train, df_test, y_test


# add operational condition to then normalize the data based on these operational conditions
def add_operating_condition(df):
    df_op_cond = df.copy()
    
    df_op_cond['Altitude'] = df_op_cond['Altitude'].round()
    df_op_cond['Mach'] = df_op_cond['Mach'].round(decimals=2)
    df_op_cond['TRA'] = df_op_cond['TRA'].round()
    
    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['Altitude'].astype(str) + '_' + \
                        df_op_cond['Mach'].astype(str) + '_' + \
                        df_op_cond['TRA'].astype(str)
    
    return df_op_cond

# normalize the data based on the operational condition
def condition_scaler(df_train, df_test, sensor_names):
  # apply operating condition specific scaling
  #scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range = (0, 1))
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_train, df_test


#to plot each sensors with respect to the RUL
def plot_signal(df, signal_name, unit=None):
    plt.figure(figsize=(13,5))
    if unit:
        plt.plot('RUL', signal_name, 
                data=df[df['Unit']==unit])
    else:
        for i in train['Unit'].unique():
            if (i % 10 == 0):  # only ploting every 10th unit_nr
                plt.plot('RUL', signal_name, 
                         data=df[df['Unit']==i])
    plt.xlim(350, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 375, 25))
    plt.ylabel(signal_name)
    plt.xlabel('Remaining Use fulLife')
    #plt.savefig(signal_name+'.jpeg')
    plt.show()

# denoise the signal using the exponential signal wih an alpha equals to 0.3
def exponential_smoothing(df, sensors, n_samples, alpha=0.3):
    df = df.copy()
    # first, calculate the exponential weighted mean of desired sensors
    df[sensors] = df.groupby('Unit')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean())
    
    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result
    
    mask = df.groupby('Unit')['Unit'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]
    
    return df

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

#the score defined in the paper
def compute_s_score(rul_true, rul_pred):
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))

#evaluate the model with R² and RMSE
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
    
def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]

def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['Unit'].unique()
        
    data_gen = (list(gen_train_data(df[df['Unit']==unit_nr], sequence_length, columns))
               for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array

def create_model(TW , remaining_):
#     history = History()
    model = Sequential()
    model.add(LSTM(units=128, activation='tanh',input_shape=(TW, len(remaining_))))
    model.add(Dense(units=128, activation='relu'))
    #model.add(GlobalAveragePooling1D(name = 'feature_layer'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    
    return model

def compute_MAPE(y_true, y_hat):
    mape = np.mean(np.abs((y_true - y_hat)/y_true))*100
    return mape

def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]  

def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['Unit'].unique()
        
    label_gen = [gen_labels(df[df['Unit']==unit_nr], sequence_length, label) 
                for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array
def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:,:] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values
        
    # specifically yield the last possible sequence
    stop = num_elements = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :] 
def plot_loss(fit_history):
    plt.figure(figsize=(13,5))
    plt.plot(range(1, len(fit_history.history['loss'])+1), fit_history.history['loss'], label='train')
    plt.plot(range(1, len(fit_history.history['val_loss'])+1), fit_history.history['val_loss'], label='validate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def new_column (df, column):
    #df = df.sort_values(by=column, ascending=False)
    df[column] = range(1, len(df) + 1)
    return df

#%% Data loading

train, test, y_test = prepare_data('FD004.txt')
# train = train.drop(['T20','P20','Nf_dmd','PCNfR_dmd','farB'],axis=1)
# test = test.drop(['T20','P20','Nf_dmd','PCNfR_dmd','farB'],axis=1)
train.shape, test.shape, y_test.shape

#%% Sensor selection
sensor_names = ['T20','T24','T30','T50','P20','P15','P30','Nf','Nc','epr','Ps30','phi',
                'NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']

# sensor_names = train

X_train_interim = add_operating_condition(train)
X_test_interim = add_operating_condition(test)
X_train_condition_scaled, X_test_condition_scaled = condition_scaler(X_train_interim, X_test_interim, sensor_names)

for sensor in sensor_names:
    plot_signal(X_train_condition_scaled, sensor)
    
# sensor_names = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi',
#                 'NRf','NRc','BPR','htBleed','W31','W32']
    
#%%
# train.columns
remaining_sensors = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi',
                'NRf','NRc','BPR','htBleed','W31','W32']

# sensor_names = train.columns

drop_sensors = [elm for elm in sensor_names if elm not in remaining_sensors]

#%%  Data preparation

X_train_interim = exponential_smoothing(X_train_interim, remaining_sensors, 0, 0.3)
X_test_interim = exponential_smoothing(X_test_interim, remaining_sensors, 0, 0.3)

# train = X_train_interim
# test = X_test_interim

# Data preparation

sequence_length = 30
gkf_cv = GroupKFold(n_splits=2)

for train_unit, val_unit in gkf_cv.split(train['Unit'].unique(), groups=train['Unit'].unique()):
    train_unit = train['Unit'].unique()[train_unit]  # gss returns indexes and index starts at 1
    val_unit = train['Unit'].unique()[val_unit]

    X_train_split = gen_data_wrapper(train, sequence_length,remaining_sensors,  train_unit)
    y_train_split = gen_label_wrapper(train, sequence_length, ['RUL'], train_unit)
    
    X_val_split = gen_data_wrapper(train, sequence_length, remaining_sensors, val_unit)
    y_val_split = gen_label_wrapper(train, sequence_length, ['RUL'], val_unit)

# create sequences train, test 
X_train = gen_data_wrapper(train, sequence_length,remaining_sensors)
y_train = gen_label_wrapper(train, sequence_length, ['RUL'])


test_gen = (list(gen_test_data(test[test['Unit']==unit_nr], sequence_length,remaining_sensors, -99.))
           for unit_nr in test['Unit'].unique())
X_test = np.concatenate(list(test_gen)).astype(np.float32)
print(X_train.shape, y_train.shape, X_test.shape)

#%% Modeling 

# Model avec une couche caché

model_1c = Sequential()
model_1c.add(LSTM(units=128, activation='tanh',input_shape=(sequence_length, len(remaining_sensors))))
model_1c.add(Dense(units=128, activation='relu'))
model_1c.add(Dropout(0.2))
model_1c.add(Dense(1,activation='relu'))
model_1c.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# model_1c.compile(loss='mean_squared_error', optimizer='adam')
# model_1c.save_weights('simple_lstm_weights.h5')


#%%
with tf.device('/device:GPU:0'):
    history = model_1c.fit(X_train_split, y_train_split,
                        validation_data=(X_val_split, y_val_split),
                        epochs=5,
                       batch_size=200)
plot_loss(history)

print(" Model evaluation for FD004 : ")
y_hat_train = model_1c.predict(X_train)
evaluate(y_train, y_hat_train, 'train')

y_hat_test = model_1c.predict(X_test)
evaluate(y_test, y_hat_test)
print(' ')
print(" Model evaluation for FD004 : ")

y_hat_test = model_1c.predict(X_test)
evaluate(y_test['RUL'].clip(upper=150), y_hat_test)
print(' ')
print("S-score for test : ",compute_s_score(y_test, y_hat_test))
print(' ')

#%% Hyper_parameter tunning M1

# I know lower alpha's perform better, so we can ditch a few high ones to reduce the search space
alpha_list = [0.01, 0.05] + list(np.arange(10,60+1,10)/100)

sequence_list = list(np.arange(10,40+1,5))
epoch_list = list(np.arange(5,20+1,5))
# nodes_list = [[32], [64], [128], [256], [32, 64], [64, 128], [128, 256]]

# lowest dropout=0.1, because I know zero dropout will yield better training results but worse generalization
dropouts = list(np.arange(1,5)/10)  

# again, earlier testing revealed relu performed significantly worse, so I removed it from the options
activation_functions = ['tanh', 'sigmoid']
batch_size_list = [32, 64, 128, 256]

tuning_options = np.prod([len(alpha_list),
                          len(sequence_list),
                          len(epoch_list),
                          # len(nodes_list),
                          len(dropouts),
                          len(activation_functions),
                          len(batch_size_list)])
tuning_options


train['RUL'].clip(upper=125, inplace=True)

def prep_data(train, test, drop_sensors, remaining_sensors, alpha):
    X_train_interim = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_interim = add_operating_condition(test.drop(drop_sensors, axis=1))

    X_train_interim, X_test_interim = condition_scaler(X_train_interim, X_test_interim, remaining_sensors)

    X_train_interim = exponential_smoothing(X_train_interim, remaining_sensors, 0, alpha)
    X_test_interim = exponential_smoothing(X_test_interim, remaining_sensors, 0, alpha)
    
    return X_train_interim, X_test_interim

# input_shape = (sequence_length, train_array.shape[2])
def create_model_c(dropout, activation, weights_file):
    
    model = Sequential()
    model.add(LSTM(units=128, activation='tanh',input_shape=input_shape))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.save_weights(weights_file)
    return model

# random grid search takes significant time, set iterations to a higher number if you truly want to tune parameters
ITERATIONS = 1

#%%
ITERATIONS = 5

results = pd.DataFrame(columns=['MSE', 'std_MSE', 'alpha', # bigger std means less robust
                                'epochs', 'dropout', 
                                'activation', 'batch_size', 
                                'sequence_length'])  

weights_file = 'lstm_hyper_parameter_weights.h5'

from tqdm import tqdm
for i in tqdm(range(ITERATIONS)):
    if ITERATIONS < 10:
        print('iteration ', i+1)
    elif ((i+1) % 10 == 0):
        print('iteration ', i+1)
    
    mse = []
    
    # init parameters
    alpha = random.sample(alpha_list, 1)[0]
    sequence_length = random.sample(sequence_list, 1)[0]
    epochs = random.sample(epoch_list, 1)[0]
    # nodes_per_layer = random.sample(nodes_list, 1)[0]
    dropout = random.sample(dropouts, 1)[0]
    activation = random.sample(activation_functions, 1)[0]
    batch_size = random.sample(batch_size_list, 1)[0]
    # remaining_sensors = random.sample(sensor_list, 1)[0]
    # drop_sensors = [element for element in sensor_names if element not in remaining_sensors]
    
    # create model
    input_shape = (sequence_length, len(remaining_sensors))
    model = create_model_c(dropout, activation, weights_file)
    
    # create train-val split
    X_train_interim, X_test_interim = prep_data(train, test, drop_sensors, remaining_sensors, alpha)
    gss = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
    for train_unit, val_unit in gss.split(X_train_interim['Unit'].unique(), groups=X_train_interim['Unit'].unique()):
        train_unit = X_train_interim['Unit'].unique()[train_unit]  # gss returns indexes and index starts at 1
        train_split_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, train_unit)
        train_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'], train_unit)
        
        val_unit = X_train_interim['Unit'].unique()[val_unit]
        val_split_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, val_unit)
        val_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'], val_unit)
        
        # train and evaluate model
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.load_weights(weights_file)  # reset optimizer and node weights before every training iteration
        
        with tf.device('/device:GPU:0'):
            history = model.fit(train_split_array, train_split_label,
                            validation_data=(val_split_array, val_split_label),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0)
        mse.append(history.history['val_loss'][-1])
    
    # append results
    d = {'MSE':np.mean(mse), 'std_MSE':np.std(mse), 'alpha':alpha, 
         'epochs':epochs, 'dropout':dropout, 
         'activation':activation, 'batch_size':batch_size, 'sequence_length':sequence_length}
    results = results.append(pd.DataFrame(d, index=[0]), ignore_index=True)
    plot_loss(history)
    
#%%
dict_results = {}
dict_results['1_layers'] = results
dict_results['1_layers'].sort_values('MSE')

#%% with optimal hyper_parameters

alpha = 0.5
sequence_length = 30
# nodes_per_layer = [256]
dropout = 0.1
activation = 'tanh'
weights_file = 'fd004_model_weights.m5'
epochs = 15  
batch_size = 128

# prep data
X_train_interim, X_test_interim = prep_data(train, test, drop_sensors, remaining_sensors, alpha)

train_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors)
label_array = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'])

test_gen = (list(gen_test_data(X_test_interim[X_test_interim['Unit']==unit_nr], sequence_length, remaining_sensors, -99.))
           for unit_nr in X_test_interim['Unit'].unique())
test_array = np.concatenate(list(test_gen)).astype(np.float32)

input_shape = (sequence_length, len(remaining_sensors))
final_model = create_model_c(dropout, activation, weights_file)

final_model.compile(loss='mean_squared_error', optimizer='adam')
final_model.load_weights(weights_file)

with tf.device('/device:GPU:0'):
    final_model.fit(train_array, label_array,
                epochs=epochs,
                batch_size=batch_size)
    
plot_loss(history)

#%%
print(" Model evaluation for FD004 : ")
y_hat_train = model.predict(train_array)
evaluate(y_train, y_hat_train, 'train')

#%%
y_hat_test = model.predict(test_array)
evaluate(y_test, y_hat_test)
print(' ')
print(" Model evaluation for FD004 : ")
#%%
y_hat_test = model.predict(X_test)
evaluate(y_test['RUL'].clip(upper=150), y_hat_test)
print(' ')
print("S-score for test : ",compute_s_score(y_test, y_hat_test))
print(' ')

#%%
# Model avec deux couches
model_2c = Sequential()
model_2c.add(LSTM(units=128, activation='tanh',input_shape=(sequence_length, len(remaining_sensors))))
model_2c.add(Dense(units=128, activation='relu'))
model_2c.add(Dense(units=32, activation='relu'))
model_2c.add(Dropout(0.2))
model_2c.add(Dense(1,activation='relu'))
model_2c.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# model_1c.compile(loss='mean_squared_error', optimizer='adam')
# model_1c.save_weights('simple_lstm_weights.h5')

# Modele avec 3 cuouches
model_3c = Sequential()
model_3c.add(LSTM(units=128, activation='tanh',input_shape=(sequence_length, len(remaining_sensors))))
model_3c.add(Dense(units=128, activation='relu'))
model_3c.add(Dense(units=32, activation='relu'))
model_3c.add(Dense(units=32, activation='relu'))
model_3c.add(Dropout(0.2))
model_3c.add(Dense(1,activation='relu'))
model_3c.compile(loss='mse',metrics=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))


#%%


model = create_model(sequence_length, remaining_sensors)
n_epochs = 20
batch_size = 384

with tf.device('/device:GPU:0'):
    history = model.fit(X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
                  epochs=n_epochs,
                  batch_size=batch_size)
    
#%%  Predict and evaluate 16 PC
print(" Model evaluation for FD004 : ")
y_hat_train = model_2c.predict(X_train)
evaluate(y_train, y_hat_train, 'train')

y_hat_test = model_2c.predict(X_test)
evaluate(y_test, y_hat_test)
print(' ')
print(" Model evaluation for FD004 : ")

y_hat_test = model_2c.predict(X_test)
evaluate(y_test['RUL'].clip(upper=150), y_hat_test)
print(' ')
print("S-score for test : ",compute_s_score(y_test, y_hat_test))
print(' ')
#%%
from lime import lime_tabular
lime_explainer = lime_tabular.RecurrentTabularExplainer(X_train, training_labels=y_train, 
                                                   feature_names=remaining_sensors,
                                                   mode = 'regression',
                                                   # discretize_continuous=True,
                                                   # class_names=['Falling', 'Rising'],
                                                   # discretizer='decile'
                                                   )
#%%
exp = lime_explainer.explain_instance(X_test[12], model_2c.predict,
                                 # num_features=21,
                                 # , labels=(1,)
                                 )

#%% Get Limes values

from tqdm import tqdm

def get_explainations(data) :
    
    # Iniatialisation
    df_expplanation = pd.DataFrame(columns=[str(i) for i in range(len(remaining_sensors)*sequence_length)])

    # Get explanations
    for row in tqdm(range(data.shape[0])) : 
        explanation = lime_explainer.explain_instance(data[row],
                                                 model_2c.predict,
                                                 num_features = len(remaining_sensors)*sequence_length)
        lime_values = explanation.local_exp[1]
        # Add explanation in df_explanation
        lime_dict = {}
        for tup in lime_values :
            lime_dict[str(tup[0])] = tup[1]
        df_expplanation.loc[len(df_expplanation)] = lime_dict
    
    return df_expplanation

lime_values = get_explainations(X_test)

#%% shape X_test


