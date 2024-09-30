# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:06:54 2024

@author: mlndao
"""
import numpy as np
import cv2

from scipy.special import comb
from methods import *

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
        M = len(mask_row) # no_of_features
        z  = np.sum(mask_row) # no_of_masked_feature

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

    #aux1 = aux


    # tc
    tc = np.copy(sample)
    assert (base_exp >= theshold).any()
    tc[base_exp >= theshold] = 0
    tc_exp = explainer.explain(tc.reshape((1, shape[0], shape[1], 1)))

    aux = tc_exp.flatten()
    theshold = aux[aux.argsort()][-nsamples]
    aux_max = aux.argsort().max()

    #aux2 = aux

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
    #tr_exp = explainer.explain(tr.reshape((1, shape[0], shape[1], 1)))

    #
    base_exp = base_exp / base_exp.max()
    tc_exp = base_exp / tc_exp.max()


    return (nsamples, top_tc_exp1, top_tc_exp2)