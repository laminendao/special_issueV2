# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:49:18 2024

@author: mlndao
"""

import numpy as np
from tqdm import tqdm
from methods import *

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