import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random
# import keras
import math

from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import GroupKFold
from sklearn import preprocessing
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from scipy import optimize
from methods import *
import warnings
from tensorflow.keras import optimizers
warnings.filterwarnings("ignore")



class KernelSHAP:
    
    def __init__(self, model, nsegments=1, nsamples=1000, feature_faker=lambda _min, _max, _mean, _std, _size: 0):
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



class L2X:
    """
    Rhe original l2x out put a vector of binary feature importance values for a
    chosen k, with k being the number of important features the user thinks is in
    the dataset. First of all user has to choose it, it's unclear how you would choose
    the best k. Second, since all important values have weights of 1, it is
    impossible to rank them.

    To overcome this, Yang proposes to run l2x for k = 1,2,...,M and add feature
    importance values. This way the most important feature will have a final importance
    value of M, because it will be 1 in each run.

    """

    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X #self.X = X.values
        x_reshaped = X.reshape((X.shape[0], -1)) #*****************
        self.x_reshaped = x_reshaped
        self.M = x_reshaped.shape[1]
        if 'batch_size' in kwargs:
            BATCH_SIZE = kwargs['batch_size']
        # set up models with k = 1,2,3,..., M
        self.models = []
        self.pred_models = []
        self.Y = self.f(X)
        # print("X shape", X.shape)
        for k in range(1, 100):
            model, pred_model = buildmodel(self.x_reshaped, self.Y, k, self.M)
            self.models.append(model)
            self.pred_models.append(pred_model)

    def explain(self, x):
        weights = np.zeros_like(x)
        #x = np.ones_like(x)
        self.expected_values = np.ones((x.shape[0], 1)) * np.mean(self.Y)
        for i in range(len(self.models)):
            # if i == 3:
                weights = weights + self.pred_models[i].predict(
                    x, verbose=False, batch_size=BATCH_SIZE
                )
        # normalize
        weights = weights / np.expand_dims(np.sum(weights, axis=1), 1)
                # print('k:', i+1)
        #print(weights[:10])
                # print(np.sum(weights,axis=0))
                # median_ranks = compute_median_rank(weights,4)
                # print(np.mean(median_ranks))
        return weights