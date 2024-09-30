#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:22:46 2024

@author: ndaolamine
"""

from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import random
import keras
import math
import cv2
import matplotlib as mpl
import itertools
import ruptures as rpt
import os
import tempfile
import re
import sys
import glob
import time
import copy
import json

from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold ,GroupShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from lime.lime_tabular import RecurrentTabularExplainer
from scipy import optimize
from itertools import combinations
from scipy.special import comb
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
from tensorflow.python.ops import gen_nn_ops
from tensorflow.keras import backend as K
from keras.models import load_model, Model, Sequential
# import cPickle as pkl
from collections import defaultdict
# from bs4 import BeautifulSoup
# import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
# from keras.layers.normalization import BatchNormalization
from keras_layer_normalization import LayerNormalization
from keras import regularizers
from keras import backend as K
# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import BatchNormalization
# from make_data import generate_data
from keras import optimizers

# read the train and test data
def prepare_data(file_name):
    dir_path = 'data/'
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
#     train = df
    plt.figure(figsize=(13,5))
    if unit:
        plt.plot('RUL', signal_name,
                data=df[df['Unit']==unit])
    else:
        for i in df['Unit'].unique():
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
    new_column = df.groupby('Unit')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean())
    df[sensors] = new_column.reset_index(level=0, drop=True)


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
#--------------------------------------------------- explain function

EPS = 1e-7

def reduce_channels(image, axis=-1, op='sum'):
    if op == 'sum':
        return image.sum(axis=axis)
    elif op == 'mean':
        return image.mean(axis=axis)
    elif op == 'absmax':
        pos_max = image.max(axis=axis)
        neg_max = -((-image).max(axis=axis))
    return np.select([pos_max >= neg_max, pos_max < neg_max], [pos_max, neg_max])


def gamma_correction(image, gamma=0.4, minamp=0, maxamp=None):
    c_image = np.zeros_like(image)
    image -= minamp
    if maxamp is None:
        maxamp = np.abs(image).max() + EPS
    image /= maxamp
    pos_mask = (image > 0)
    neg_mask = (image < 0)
    c_image[pos_mask] = np.power(image[pos_mask], gamma)
    c_image[neg_mask] = -np.power(-image[neg_mask], gamma)
    c_image = c_image * maxamp + minamp
    return c_image

def project_image(image, output_range=(0, 1), absmax=None, input_is_positive_only=False):
    if absmax is None:
        absmax = np.max(np.abs(image), axis=tuple(range(1, len(image.shape))))
    absmax = np.asarray(absmax)
    mask = (absmax != 0)
    if mask.sum() > 0:
        image[mask] = image[mask] / np.expand_dims(absmax[mask], axis=-1)
    if not input_is_positive_only:
        image = (image + 1) / 2 
    image = image.clip(0, 1)
    projection = output_range[0] + image * (output_range[1] - output_range[0])
    return projection

def get_model_params(model):
    names, activations, weights, layers = [], [], [], []
    for layer in model.layers:
        name = layer.name
        names.append(name)
        activations.append(layer.output)
        weights.append(layer.get_weights())
        layers.append(layer)

    return names, activations, weights, layers

def display(
    signal,
    true_chg_pts,
    computed_chg_pts=None,
    computed_chg_pts_color="k",
    computed_chg_pts_linewidth=3,
    computed_chg_pts_linestyle="--",
    computed_chg_pts_alpha=1.0,
    **kwargs
):
    """Display a signal and the change points provided in alternating colors.
    If another set of change point is provided, they are displayed with dashed
    vertical dashed lines. The following matplotlib subplots options is set by
    default, but can be changed when calling `display`):
    - figure size `figsize`, defaults to `(10, 2 * n_features)`.
    Args:
        signal (array): signal array, shape (n_samples,) or (n_samples, n_features).
        true_chg_pts (list): list of change point indexes.
        computed_chg_pts (list, optional): list of change point indexes.
        computed_chg_pts_color (str, optional): color of the lines indicating
            the computed_chg_pts. Defaults to "k".
        computed_chg_pts_linewidth (int, optional): linewidth of the lines
            indicating the computed_chg_pts. Defaults to 3.
        computed_chg_pts_linestyle (str, optional): linestyle of the lines
            indicating the computed_chg_pts. Defaults to "--".
        computed_chg_pts_alpha (float, optional): alpha of the lines indicating
            the computed_chg_pts. Defaults to "1.0".
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.
    Returns:
        tuple: (figure, axarr) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """
    try :
        import matplotlib.pyplot as plt
    except ImportError:
        raise plt.MatplotlibMissingError(
            "This feature requires the optional dependency matpotlib, you can install it using `pip install matplotlib`."
        )

    if type(signal) != np.ndarray:
        # Try to get array from Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    # let's set a sensible defaut size for the subplots
    matplotlib_options = {
        "figsize": (10, 2 * n_features),  # figure size
    }
    # add/update the options given by the user
    matplotlib_options.update(kwargs)

    # create plots
    fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
    if n_features == 1:
        axarr = [axarr]

    cmap = mpl.colormaps['RdYlGn']
    for i, (axe, sig) in enumerate(zip(axarr, signal.T)):
        # plot s
        axe.plot(range(n_samples), sig)

        # color each (true) regime
        bkps = [0] + sorted(true_chg_pts)
        alpha = 0.4  # transparency of the colored background

        for (_, start, end, imp) in true_chg_pts[i]:
            axe.axvspan(max(0, start - 0.5), end - 0.5, color=cmap(imp), alpha=alpha)


    fig.tight_layout()

    return fig, axarr

def usegment(signal, n_bkps=2):
    """
    Compute the segmentation of the signal uniform
    """
    # Calcula el número de períodos completos en la señal x
    period = len(signal) // n_bkps

    # Divide la señal x en tantos períodos completos como sea posible
    result = []
    for i in range(signal.shape[0]):
        result += [(i, p, p + period) for p in range(0, len(signal), period)]

    return result

def segment(signal, n_bkps=2):
    """
    Compute the segmentation of the signal
    """
    result = []
    for i in range(signal.shape[0]):
        algo = rpt.Dynp(model="l2").fit(signal[i].T)
        r = [0] + algo.predict(n_bkps)
        result += [(i, l, r) for (l,r) in zip(r, r[1:])]


    return result

def sampling(signal, segments, feature_faker, n=10, k=3):
    ranges = [(signal[i].min(), signal[i].max()) for i in range(signal.shape[0])]
    mean_std = [(signal[i].mean(), signal[i].std()) for i in range(signal.shape[0])]

    zprimes = []
    for kk in range(1, k):
        for seg in combinations(segments, kk):
            zprime = [1 if s in seg else 0 for s in segments]
            z = np.copy(signal)

            for zj, (i, start, end) in zip(zprime, segments):
                if zj == 0:
                   # z[i, start:end] = np.random.normal(mean_std[i][0],mean_std[i][1],end-start)
                  z[i, start:end] = feature_faker(*ranges[i], *mean_std[i], end-start)


            n -= 1
            zprimes.append((zprime, z))
            if n <= 0:
                break

        if n <= 0:
            break

    for _ in range(n):
        nsegments = random.randint(1, len(segments)-1)
        seg = random.sample(segments, nsegments)
        zprime = [1 if s in seg else 0 for s in segments]
        z = np.copy(signal)
        for zj, (i, start, end) in zip(zprime, segments):
            if zj == 0:
                #z[i, start:end] = np.random.normal(mean_std[i][0],mean_std[i][1],end-start)
                z[i, start:end] = feature_faker(*ranges[i], *mean_std[i], end-start)


        zprimes.append((zprime, z))

    return zprimes





def mean_sample(signal):
    """
    Creates a new sample

    a normal distribution with mean and std of each
    feature
    """
    mean_std = [(signal[i].mean(), signal[i].std()) for i in range(signal.shape[0])]

    result = np.zeros(signal.shape)
    for i in range(signal.shape[0]):
        #result[i] = np.random.normal(mean_std[i][0],mean_std[i][1], signal.shape[1])
        result[i] = mean_std[i][0]


    return result


def validate_acumen(explainer, samples, iterations=100, nstd=1.5, top_features=100, verbose=True):
    """
    Se eleccionan n puntos de los que tienen mayor score y se perturban creando una nueva
    serie (tp). También se crea otra serie (tr) metiendo ruido en otros n puntos aleatorios.
    La importancia qm de puntos debe cumplir las siguiente regla qm(t) >= qm(tr) > qm(tp)
    """

    ranking = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1]
        base_exp = explainer.explain(xi)

        if not np.isnan(base_exp).any():
            # compute the thresold using mean + n*std
            _mean, _std = base_exp.mean(), base_exp.std()
            theshold = _mean + nstd*_std
            nsamples = (base_exp > theshold).sum()
            nsamples = min(nsamples, top_features)
            aux1 = base_exp.flatten()

            top_mask = aux1.argsort()[-nsamples:]

            # tc
            tc = np.copy(xi).reshape(base_exp.shape)
            tc[base_exp >= theshold] = 0
            tc_exp = explainer.explain(tc.reshape(xi.shape))

            # obtengo las n muestra más imporantes de x'
            aux = tc_exp.flatten()
            aux_max = aux.argsort().max()

            # ranking
            ranking.append((1- (np.argsort(aux).argsort()[top_mask] / aux_max)).mean())

    return np.nanmean(ranking)

def validate_coherence(model, explainer, samples, targets, nstd=1.5, top_features=100, verbose=True):
    explains = []
    valid_idx = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1]
        exp = explainer.explain(xi)

        # compute the thresold using mean + n*std
        _mean, _std = exp.mean(), exp.std()
        theshold = _mean + nstd*_std
        nsamples = (exp > theshold).sum()
        nsamples = min(nsamples, top_features)
        aux = exp.flatten()
        theshold = aux[aux.argsort()][-nsamples]
        indexes = np.argwhere(exp.flatten() < theshold)

        # remove that features
        exp[exp < theshold] = 0
        xic = np.copy(xi).flatten()
        xic[indexes] = 0
        xic = xic.reshape(xi.shape)

        if not np.isnan(exp).any():
            valid_idx.append(i)
            explains.append(xic)

    samples = samples[valid_idx]
    targets = targets[valid_idx]

    tmax = targets.max()
    targets = targets / tmax

    pred = model.predict(samples) / tmax
    pred = pred.reshape(targets.shape)
    errors = 1 - (pred - targets) ** 2

    exp = np.array(explains).reshape(samples.shape)

    explains = np.array(explains).reshape(samples.shape)
    exp_pred = model.predict(explains) / tmax
    exp_errors = 1- (exp_pred - targets) ** 2

    coherence_i = np.abs(errors - exp_errors)
    coherence = np.mean(coherence_i)

    return {
            'coherence': coherence,
            'completeness':np.mean(exp_errors / errors),
            'congruency': np.sqrt(np.mean((coherence_i - coherence)**2))
           }


def validate_identity(model, explainer, samples, verbose=True):
    """
    The principle of identity states that identical objects should receive identical explanations. This is
    a measure of the level of intrinsic non-determinism in the method:

                                d(xa , xb ) = 0 => ∀a,b d(εa , εb ) = 0

    """
    errors = []
    for i, sample in tqdm.tqdm(enumerate(samples), total=samples.shape[0], disable=not verbose):
        exp_a = explainer.explain(samples[i:i+1])
        exp_b = explainer.explain(samples[i:i+1])

        if not np.isnan(exp_b).any() and not np.isnan(exp_b).any():
            errors.append(1 if np.all(exp_a == exp_b) else 0)

    return np.nanmean(errors)


def validate_separability(model, explainer, samples, verbose=True):
    """
     Non-identical objects can not have identical explanations:

                 ∀a,b; d(xa , xb ) ̸= 0 =>  d(εa , εb ) > 0

     This proxy is based on the assumption that every feature has
     a minimum level of importance, positive or negative, in the
     predictions. The idea is that if a feature is not actually
     needed for the prediction, then two samples that differ only
     in that feature will have the same prediction. In this
     scenario, the explanation method could provide the same
     explanation, even though the samples are different.

    """
    explains = []
    samples_aux = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1]
        exp = explainer.explain(xi)

        if not np.isnan(exp).any():
            samples_aux.append(xi)
            explains.append(exp)

    samples = np.array(samples_aux)

    errors = []
    for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):

        for j in range(i+1, len(samples)-1):

            if i == j:
                continue

            exp_a = explains[i] #explainer.explain(samples[i:i+1])
            exp_b = explains[j] #explainer.explain(samples[i+1:i+2])

            assert np.any(samples[i] != samples[j])
            #assert np.sum((exp_a - exp_b)**2) > 0

            errors.append(1 if np.sum((exp_a - exp_b)**2) > 0 else 0)

    return np.nanmean(errors)


def validate_stability(model, explainer, samples, verbose=True):
    """
    Similar objects must have similar explanations. This is built
    on the idea that an explanation method should only return
    similar explanations for slightly different objects. The
    spearman correlation ρ is used to define this

      ∀i;
      ρ({d(xi , x0 ), d(xi , x1 ), ..., d(xi , xn )},
        {d(εi , ε0 ), d(εi , ε1 ), ..., d(εi , εn )}) = ρi > 0

    """
    explains = []
    samples_aux = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1]
        exp = explainer.explain(xi)


        if not np.isnan(exp).any():
            samples_aux.append(xi)
            explains.append(exp)

    samples = np.array(samples_aux)

    errors = []
    for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):
        dxs, des = [], []
        xi = samples[i:i+1]
        for j in range(len(samples)):
            if i==j:
                continue

            xj = samples[j:j+1]
            #exp_i = explainer.explain(xi)
            #exp_j = explainer.explain(xj)
            exp_i = explains[i]
            exp_j = explains[j]

            if np.isnan(exp_i).any() or np.isnan(exp_j).any():
                continue

            dxs.append(euclidean(xi.flatten(), xj.flatten()))
            des.append(euclidean(exp_i.flatten(), exp_j.flatten()))


        errors.append(stats.spearmanr(dxs, des).correlation)


    return np.nanmean(errors)


def validate_selectivity(model, explainer, samples, samples_chunk=1, verbose=True):
    """
    The elimination of relevant variables must affect
    negatively to the prediction. To compute the selectivity
    the features are ordered from most to lest relevant.
    One by one the features are removed, set to zero for
    example, and the residual errors are obtained to get the
    area under the curve (AUC).
    """

    errors = []
    for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):
        dxs, des = [], []
        xi = samples[i:i+1]
        ei = explainer.explain(xi)
        if np.isnan(ei).any():
            continue

        idxs = ei.flatten().argsort()[::-1]
        xi = xi[0]
        xs = [xi]
        xprime = xi.flatten()
        l = idxs.shape[0]
        if samples_chunk >= 1:
            idxs = np.split(idxs, int(l/samples_chunk))

        for i in idxs:
            xprime[i] = 0
            xs.append(xprime.reshape(xi.shape))
            xprime = np.copy(xprime)

        preds = model.predict(np.array(xs), batch_size=32)[:,0]
        e = np.abs(preds[1:] - preds[:-1]) / (preds[0] + 1e-12)
        e = np.cumsum(e)
        e = 1 - (e / (e.max() + 1e-12))
        score = 1 - np.mean(e)

        errors.append(score)

    return np.nanmean(errors)



def apply_modifications(model, custom_objects=None):
    """
    Aplicamos las modificaciones realizadas en el modelo creando un nuevo grafo de
    computación. Para poder modificar el grafo tenemos que grabar y leer el modelo.
    """
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)
        
#--------------------------------- Function for L2X

BATCH_SIZE = 1000
# np.random.seed(0)
#tf.random.set_seed(0)
#tf.random.set_random_seed(0)
# random.seed(0)


def create_data(datatype, n=1000):
    """
    Create train and validation datasets.

    """
    x_train, y_train, _ = generate_data(n=n, datatype=datatype, seed=0)
    x_val, y_val, datatypes_val = generate_data(n=10 ** 5, datatype=datatype, seed=1)

    input_shape = x_train.shape[1]

    return x_train, y_train, x_val, y_val, datatypes_val, input_shape


def create_rank(scores, k):
    """
    Compute rank of each feature based on weight.

    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d)
        permutated_weights = score[idx]
        permutated_rank = (-permutated_weights).argsort().argsort() + 1
        rank = permutated_rank[np.argsort(idx)]

        ranks.append(rank)
    print("n: {}, d: {}, scores: {}, ranks: {}".format(n, d, scores[0:4], ranks[0:4]))
    return np.array(ranks)


def compute_median_rank(scores, k, datatype_val=None):
    ranks = create_rank(scores, k)
    if datatype_val is None:
        median_ranks = np.median(ranks[:, :k], axis=1)
        print(median_ranks)
    else:
        datatype_val = datatype_val[: len(scores)]
        median_ranks1 = np.median(
            ranks[datatype_val == "orange_skin", :][:, np.array([0, 1, 2, 3, 9])],
            axis=1,
        )
        median_ranks2 = np.median(
            ranks[datatype_val == "nonlinear_additive", :][
                :, np.array([4, 5, 6, 7, 9])
            ],
            axis=1,
        )
        median_ranks = np.concatenate((median_ranks1, median_ranks2), 0)
    return median_ranks


class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(
            shape=(batch_size, self.k, d),
            minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
            maxval=1.0,
        )

        gumbel = -K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        threshold = tf.expand_dims(
            tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1
        )
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'tau0': self.tau0,
            'k': self.k
        })
        return config

def get_filepath():
    currentLogs = glob.glob(f"results/saved_models/*-L2X.hdf5")
    numList = [0]
    for i in currentLogs:
        i = os.path.splitext(i)[0]
        try:
            num = re.findall("[0-9]+$", i)[0]
            numList.append(int(num))
        except IndexError:
            pass
    numList = sorted(numList)
    newNum = numList[-1] + 1
    return f"results/saved_models/{newNum}-L2X.hdf5"

def buildmodel(x, y, k, input_shape, n_class=1):
    # n_class = 1: regression task
    # n_class > 1: classfication task (not implemented yet)
    if n_class == 1:
        loss = "mse"
        monitor = "val_loss"
        final_activation = "linear"
        save_mode = 'min'
    elif n_class > 1:
        print('shape')
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1) 
        print(y.shape)
        if y.shape[1] == 1 and n_class == 2:
            y_new = np.zeros((y.shape[0],n_class))
            y_new[:,0] = copy.deepcopy(np.squeeze(y))
            y_new[:,1] = 1 - np.squeeze(y)
            y = copy.deepcopy(y_new)
            # for i in range(y_new.shape[0]):
            #     print(y_new[i,:])
        loss = "categorical_crossentropy"
        monitor = "val_acc"
        final_activation = "softmax"
        save_mode = 'max'
    st1 = time.time()
    st2 = st1
    activation = "selu"
    l2 = 1e-3  # default 1e-3
    # P(S|X)
    model_input = Input(shape=(input_shape,), dtype="float32")

    net = Dense(
        100,
        activation=activation,
        name="s/dense1",
        kernel_regularizer=regularizers.l2(l2),
    )(model_input)
    net = Dense(
        100,
        activation=activation,
        name="s/dense2",
        kernel_regularizer=regularizers.l2(l2),
    )(net)

    # A tensor of shape, [batch_size, max_sents, 100]
    logits = Dense(input_shape)(net)
    # [BATCH_SIZE, max_sents, 1]
    tau = 0.1
    samples = Sample_Concrete(tau, k, name="sample")(logits)

    # q(X_S)
    new_model_input = Multiply()([model_input, samples])
    net = Dense(
        200,
        activation=activation,
        name="dense1",
        kernel_regularizer=regularizers.l2(l2),
    )(new_model_input)
    net = BatchNormalization()(net)  # Add batchnorm for stability.
    net = Dense(
        200,
        activation=activation,
        name="dense2",
        kernel_regularizer=regularizers.l2(l2),
    )(net)
    net = BatchNormalization()(net)

    preds = Dense(
        n_class,
        activation=final_activation,
        name="dense4",
        kernel_regularizer=regularizers.l2(l2),
    )(net)
    model = Model(model_input, preds)
    pred_model = Model(model_input, samples)

    adam = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=loss, optimizer=adam, metrics=[ "mse"])

    # filepath = get_filepath()
    # filepath = "{}-L2X.hdf5".format(
    #     k
    # )  # Yang: hacky way to get the model to store with some name
    # checkpoint = ModelCheckpoint(
    #     filepath, monitor=monitor, verbose=1, save_best_only=True, mode=save_mode
    # )
    callbacks_list = []
    # print("start training, k: {}, final nonlinearity: {}".format(k, final_activation))
    
    model.fit(
        x,
        y,
        validation_data=(x, y),
        callbacks=callbacks_list,
        epochs=1,
        batch_size=BATCH_SIZE,
    )

    pred_model.compile(
        loss=None, optimizer="rmsprop", metrics=["mse"],
    )

    return model, pred_model

#************************** explain function

class KernelSHAP:

    def __init__(self, model, nsegments=2, nsamples=1000, feature_faker=lambda _min, _max, _mean, _std, _size: 0):
        self.model = model
        self.nsegments = nsegments
        self.nsamples = nsamples
        self.feature_faker = feature_faker


    # Weight is defined in the paper https://arxiv.org/pdf/1705.07874.pdf .
    def _get_weights(self, mask_row):
        """
        Compute the weight of a sample as an measuring of the number of 1's
        """
        M = no_of_features = len(mask_row)
        z = no_of_masked_feature = np.sum(mask_row)

        weight = (M-1) /  (  comb(M,z) * z * (M-z)   )
        return weight


    def explain(self, input_array):

        mean_pred = self.model.predict(input_array)[0][0]

        if len(input_array.shape) > 2:
            input_array = input_array.squeeze()

        # compute the mean prediction
        #mean_pred = self.model.predict(np.array([mean_sample(input_array)]))[0][0]


        # create the mask to train the linear model
        nsamples = self.nsamples
        mask = []
        predictions = []
        segments = segment(input_array, self.nsegments)
        segments = sorted(segments, key=(lambda x: x[1]))
        samples = sampling(input_array, segments, n=nsamples, feature_faker=self.feature_faker)
        zs = np.array([s[1] for s in samples])
        predictions = self.model.predict(zs, batch_size=128)
        for zprime, z in samples:
            mask.append(zprime)


        # weights  of the masks
        weights = np.apply_along_axis(self._get_weights, 1, mask)

        # compute the importance coeficients
        B0 = mean_pred
        W = np.diag(weights)


        X = np.array(mask)
        y = np.array(predictions).reshape(self.nsamples,1)

        y = np.abs(y - B0)
        y = 1 - (y / y.max())

        B = np.dot(np.linalg.inv(np.dot(np.dot(X.transpose(),W),X)), np.dot(np.dot(X.transpose(),W),y))
        coef = np.copy(B.squeeze())
        
        # Essayer d'autre type de kernel non linéaire ; limite de shap l'aspect lineaire


        heatmap = np.zeros(input_array.shape, np.float32)
        for (i, s, e), imp in zip(segments, coef):
            heatmap[i,s:e] = imp

        #heatmap = cv2.resize(heatmap.T, dsize=input_array.shape, interpolation=cv2.INTER_CUBIC).T

        return heatmap


    def display(self, input_array, explanation):

        s = input_array[0].squeeze()
        aux_hm = cv2.resize(explanation.T, dsize=s.shape, interpolation=cv2.INTER_CUBIC).T

        aux_hm =  (aux_hm - aux_hm.min()) / (aux_hm.max() - aux_hm.min())
        importances = [[(i, j, j+1, aux_hm[i,j]) for j in range(s.shape[1])] for i in range(s.shape[0])]

        display(s.T, importances)



def validate(explainer, sample, iterations=100, nstd=1.5, top_features=100):
    """
    N points are chosen from those with the highest score and are perturbed to create a new series (tp). 
    Another series (tr) is also created by adding noise to another n random points. 
    The importance qm of points must comply with the following rule qm(t) >= qm(tr) > qm(tp)
    """

    base_exp = explainer.explain(sample)
    if len(sample.shape) > 3:
        sample = sample[0,:,:,0]
    else:
        sample = sample[0]
    shape = sample.shape

    _mean, _std = base_exp.mean(), base_exp.std()
    theshold = _mean + nstd*_std
    nsamples = (base_exp > theshold).sum()
    nsamples = min(nsamples, top_features)
    aux = base_exp.flatten()
    theshold = aux[aux.argsort()][-nsamples]
    top_mask = aux >= theshold

    aux1 = aux


    # tc
    tc = np.copy(sample)
    assert (base_exp >= theshold).any()
    tc[base_exp >= theshold] = 0
    tc_exp = explainer.explain(tc.reshape((1, shape[0], shape[1], 1)))

    aux = tc_exp.flatten()
    theshold = aux[aux.argsort()][-nsamples]
    aux_max = aux.argsort().max()

    aux2 = aux

    # ranking
    top_tc_exp2 = (1- (aux.argsort()[top_mask] / aux_max)).mean()

    # number of noised samples outside top
    top_tc_exp1 = (aux < theshold)[top_mask].mean()


    # tr
    s = base_exp.shape[0] * base_exp.shape[1]
    t_mask = np.zeros(s, dtype=bool)
    t_mask[np.random.choice(s, nsamples, replace=False)] = True
    t_mask = t_mask.reshape(base_exp.shape)
    tr = np.copy(sample)
    tr[t_mask] = 0
    tr_exp = explainer.explain(tr.reshape((1, shape[0], shape[1], 1)))

    #
    base_exp = base_exp / base_exp.max()
    tc_exp = base_exp / tc_exp.max()


    return (nsamples, top_tc_exp1, top_tc_exp2)

********************************* Metrics

def identity(X_dist, E_dist) : 
    
    '''
        The principle of identity states that identical objects should receive identical explanations. This is 
        a measure of the level of intrinsic non-determinism in the method:
        
                                    d(xa , xb ) = 0 => ∀a,b d(εa , εb ) = 0
    ''' 

    # Defining a function to navigate around the distance matrix
    i_dm = X_dist.values #Computations on np.arrays are faster than on dataframes
    l_e_dm = E_dist.values
    errors = []
    
    # To run the loop with a progress bar
    for column in X_dist:
        for row in X_dist:
            if i_dm[row, column] == 0: #If two inputs have distance 0, then their explanations must too
                if l_e_dm[row, column] == 0:
                    errors.append(1)
                else:
                    errors.append(0)
        
        return np.nanmean(errors)*100

def separability(X_dist, E_dist):
    """
      Non-identical objects can not have identical explanations:
    
                  ∀a,b; d(xa , xb ) ̸= 0 =>  d(εa , εb ) > 0

      This proxy is based on the assumption that every feature has a minimum level of importance,
      positive or negative, in the predictions. The idea is that if a feature is not actually 
      needed for the prediction, then two samples that differ only in that feature will 
      have the same prediction. In this scenario, the explanation method could provide the same 
      explanation, even though the samples are different.

    """
    
    i_dm = X_dist.values #Computations on np.arrays are faster than on dataframes
    l_e_dm = E_dist.values
    errors = []
    
    # To run the loop with a progress bar
    for column in X_dist:
        for row in X_dist:
            if i_dm[row, column] > 0: #If two inputs have distance 0, then their explanations must too
                if l_e_dm[row, column] > 0:
                    errors.append(1)
                else:
                    errors.append(0)
        
        return np.nanmean(errors)*100
    

def stability(X_dist, E_dist):
    """
      Non-identical objects can not have identical explanations:
    
                  ∀a,b; d(xa , xb ) ̸= 0 =>  d(εa , εb ) > 0

      This proxy is based on the assumption that every feature has a minimum level of importance,
      positive or negative, in the predictions. The idea is that if a feature is not actually 
      needed for the prediction, then two samples that differ only in that feature will 
      have the same prediction. In this scenario, the explanation method could provide the same 
      explanation, even though the samples are different.

    """
    
    errors = []
    
    #Creating a list that contains all Spearman's Rhos rank correlation coefficients
    sp_rhos = list()
    
    #Creating a function that computes all Rhos
    for column in X_dist:
        sp_rho = spearmanr(X_dist.iloc[:,column], E_dist.iloc[:,column])
        sp_rhos.append(sp_rho[0])
    
    # Defining a function to navigate around the distance matrix

    for entry in sp_rhos:
        if entry >= 0:
            errors.append(1)
        else:
            errors.append(0)
    
    return np.nanmean(errors)*100



def coherence(model, explainer, samples, targets, e, nstd=1.5, top_features=5, verbose=True, L2X = False):
    '''
    model : model.predic
    explainer : get_explanation's function
    sample : data
    target : label
    e : e like KenrelSHAP
    '''
    explains = []
    valid_idx = []
    # for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
    for i in range(len(samples)):
        xi = samples[i:i+1] 
        exp = explainer(xi, e, L2X).values
        # exp = explainer(xi, lime_explainer, predict_with_xgboost).values

        # compute the thresold using mean + n*std 
        _mean, _std = exp.mean(), exp.std()
        theshold = _mean + nstd*_std
        nsamples = (exp > theshold).sum()
        nsamples = min(nsamples, top_features)
        aux = exp.flatten()
        theshold = aux[aux.argsort()][-nsamples]
        indexes = np.argwhere(exp.flatten() < theshold)

        # remove that features
        exp[exp < theshold] = 0
        xic = np.copy(xi).flatten()
        xic[indexes] = 0
        xic = xic.reshape(xi.shape)

        if not np.isnan(exp).any():
            valid_idx.append(i)
            explains.append(xic)

    samples = samples[valid_idx]
    targets = targets[valid_idx]
    # dtest = model.DMatrix(samples, targets)


    tmax = targets.max()
    targets = targets / tmax

    pred = model(samples) / tmax
    pred = pred.reshape(targets.shape)
    errors = 1 - (pred - targets) ** 2

    exp = np.array(explains).reshape(samples.shape)

    explains = np.array(explains).reshape(samples.shape)
    exp_pred = model(explains) / tmax
    exp_errors = 1- (exp_pred - targets) ** 2
    
    coherence_i = np.abs(errors - exp_errors)
    coherence = np.mean(coherence_i)
    
#     return {
#             'coherence': coherence, 
#             'completeness':np.mean(exp_errors / errors),
#             'congruency': np.sqrt(np.mean((coherence_i - coherence)**2))
#            }
    return coherence, np.mean(exp_errors / errors), np.sqrt(np.mean((coherence_i - coherence)**2))


def selectivity(model, explainer, samples, e_x, L2X = False, samples_chunk=1, verbose=True):
    """
    The elimination of relevant variables must affect 
    negatively to the prediction. To compute the selectivity 
    the features are ordered from most to lest relevant. 
    One by one the features are removed, set to zero for 
    example, and the residual errors are obtained to get the 
    area under the curve (AUC).
    
    - model : model.predic
    - explainer : get_explanation's function
    - sample : data
    - e_x : e like KenrelSHAP
    """

    errors = []
    # for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):
    for i in tqdm(range(len(samples))):
        dxs, des = [], []
        xi = samples[i:i+1]
        ei = explainer(xi, e_x, L2X).values
        if np.isnan(ei).any():
            continue
            
        idxs = ei.flatten().argsort()[::-1]
        xi = xi[0]
        xs = [xi]
        xprime = xi.flatten()
        l = idxs.shape[0]
        if samples_chunk >= 1:
            idxs = np.split(idxs, int(l/samples_chunk))
        
        for i in idxs:
            xprime[i] = 0
            xs.append(xprime.reshape(xi.shape))
            xprime = np.copy(xprime)
            
        preds = model(np.array(xs))   
        e = np.abs(preds[1:] - preds[:-1]) / np.abs(preds[0] + 1e-12)
        e = np.cumsum(e)
        e = 1 - (e / (e.max() + 1e-12))
        score = 1 - np.mean(e)
        
        errors.append(score)
         # print('ok')
        
    return np.nanmean(errors)



def acumen(explainer, samples, e, L2X=False, iterations=100, nstd=1.5, top_features=5, verbose=True):
    """
    Se eleccionan n puntos de los que tienen mayor score y se perturban creando una nueva
    serie (tp). También se crea otra serie (tr) metiendo ruido en otros n puntos aleatorios. 
    La importancia qm de puntos debe cumplir las siguiente regla qm(t) >= qm(tr) > qm(tp)
    
    - model : model.predic
    - explainer : get_explanation's function
    - sample : data
    - e : e like KenrelSHAP
    """
    
    ranking = []
    # for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
    for i in tqdm(range(len(samples))):

        xi = samples[i:i+1] 
        base_exp = explainer(xi, e, L2X).values

        if not np.isnan(base_exp).any():
            # compute the thresold using mean + n*std 
            _mean, _std = base_exp.mean(), base_exp.std()
            theshold = _mean + nstd*_std
            nsamples = (base_exp > theshold).sum()
            nsamples = min(nsamples, top_features)
            aux1 = base_exp.flatten()

            top_mask = aux1.argsort()[-nsamples:]

            # tc
            tc = np.copy(xi).reshape(base_exp.shape)
            tc[base_exp >= theshold] = 0
            tc_exp = explainer(tc.reshape(xi.shape), e, L2X).values

            # obtengo las n muestra más imporantes de x'
            aux = tc_exp.flatten()
            aux_max = aux.argsort().max()

            # ranking
            ranking.append((1- (np.argsort(aux).argsort()[top_mask] / aux_max)).mean())
        
    return np.nanmean(ranking)

##********************* Make data

"""
This script contains functions for generating synthetic data. 

Part of the code is based on https://github.com/Jianbo-Lab/CCM
""" 
# from __future__ import print_function
import numpy as np  
from scipy.stats import chi2

def generate_XOR_labels(X):
    y = np.exp(X[:,0]*X[:,1])

    prob_1 = np.expand_dims(1 / (1+y) ,1)
    prob_0 = np.expand_dims(y / (1+y) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_orange_labels(X):
    logit = np.exp(np.sum(X[:,:4]**2, axis = 1) - 4.0) 

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y

def generate_additive_labels(X):
    logit = np.exp(-100 * np.sin(0.2*X[:,0]) + abs(X[:,1]) + X[:,2] + np.exp(-X[:,3])  - 2.4) 

    prob_1 = np.expand_dims(1 / (1+logit) ,1)
    prob_0 = np.expand_dims(logit / (1+logit) ,1)

    y = np.concatenate((prob_0,prob_1), axis = 1)

    return y



def generate_data(n=100, datatype='', seed = 0, val = False):
    """
    Generate data (X,y)
    Args:
        n(int): number of samples 
        datatype(string): The type of data 
        choices: 'orange_skin', 'XOR', 'regression'.
        seed: random seed used
    Return: 
        X(float): [n,d].  
        y(float): n dimensional array. 
    """

    np.random.seed(seed)

    X = np.random.randn(n, 10)

    datatypes = None 

    if datatype == 'orange_skin': 
        y = generate_orange_labels(X) 

    elif datatype == 'XOR':
        y = generate_XOR_labels(X)    

    elif datatype == 'nonlinear_additive':  
        y = generate_additive_labels(X) 

    elif datatype == 'switch':

        # Construct X as a mixture of two Gaussians.
        X[:n//2,-1] += 3
        X[n//2:,-1] += -3
        X1 = X[:n//2]; X2 = X[n//2:]

        y1 = generate_orange_labels(X1)
        y2 = generate_additive_labels(X2)

        # Set the key features of X2 to be the 4-8th features.
        X2[:,4:8],X2[:,:4] = X2[:,:4],X2[:,4:8]

        X = np.concatenate([X1,X2], axis = 0)
        y = np.concatenate([y1,y2], axis = 0) 

        # Used for evaluation purposes.
        datatypes = np.array(['orange_skin'] * len(y1) + ['nonlinear_additive'] * len(y2)) 

        # Permute the instances randomly.
        perm_inds = np.random.permutation(n)
        X,y = X[perm_inds],y[perm_inds]
        datatypes = datatypes[perm_inds]


    return X, y, datatypes  