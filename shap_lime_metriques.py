# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:55:10 2023

@author: mlndao
"""

import numpy as np
import xgboost
import pandas as pd
import shap

from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# import matplotlib.pyplot as plt

#%% Exemple de données

N = 2_000
X = np.zeros((N, 5))

X[:1_000, 0] = 1

X[:500, 1] = 1
X[1_000:1_500, 1] = 1

X[:250, 2] = 1
X[500:750, 2] = 1
X[1_000:1_250, 2] = 1
X[1_500:1_750, 2] = 1

# mean-center the data
X[:, 0:3] -= 0.5

y = 2 * X[:, 0] - 3 * X[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Model

xgb_params = {
    'objective': 'reg:squarederror',
    'eta': 0.10,
    'eval_metric': 'rmse',
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample' : 0.8,
    'colsample_bytree': 0.8,
#     'silent': 1,
    'seed': 7
}

import xgboost as xgb
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

# dtrain.shape
# print(dvalid.shape)

model = xgb.train(xgb_params,
                          dtrain,
                          num_boost_round = 100, #corresponds to n_estimators
                         )

#%% Get Shap values

pred = model.predict(dtest, output_margin=True)

shap_explainer = shap.TreeExplainer(model)
explanation = shap_explainer(X_test)

shap_values = explanation.values

#%% Get Limes values

# Construct explanation's function
lime_explainer = LimeTabularExplainer(X_train, 
                                 mode="regression",
                                 training_labels = y_train)

# Define a custom prediction function that works with LIME
def predict_with_xgboost(X):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)

def get_explainations(data) :
    
    # Iniatialisation
    df_expplanation = pd.DataFrame(columns=[str(i) for i in range(data.shape[1])])

    # Get explanations
    for row in tqdm(range(data.shape[0])) : 
        explanation = lime_explainer.explain_instance(data[row],
                                                 predict_with_xgboost)
        lime_values = explanation.local_exp[1]
        # Add explanation in df_explanation
        lime_dict = {}
        for tup in lime_values :
            lime_dict[str(tup[0])] = tup[1]
        df_expplanation.loc[len(df_expplanation)] = lime_dict
    
    return df_expplanation

lime_values = get_explainations(X_test)

#%%
shap_explainer = shap_explainer
lime_explainers = get_explainations

#%% Metric's function

#Calculate distance matrix

# 1. Distance entre observations

X_dist = pd.DataFrame(squareform(pdist(X_test)))
Lime_dist = pd.DataFrame(squareform(pdist(lime_values))) # Lime values explanation matrix
shap_dist = pd.DataFrame(squareform(pdist(shap_values))) # Shape values explanation matrix

# Calculate metrics

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

#%% 
shap_explainer = shap_explainer
lime_explainers = get_explainations

def coherence(model, explainer, samples, targets, nstd=1.5, top_features=5, verbose=True):
    explains = []
    valid_idx = []
    # for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
    for i in range(len(samples)):
        xi = samples[i:i+1] 
        exp = explainer(xi).values
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
    
    return {
            'coherence': coherence, 
            'completeness':np.mean(exp_errors / errors),
            'congruency': np.sqrt(np.mean((coherence_i - coherence)**2))
           }

# coherence(model = predict_with_xgboost, explainer = explainer, samples = X_test, targets=y_test)

#%%

def selectivity(model, explainer, samples, samples_chunk=1, verbose=True):
    """
    The elimination of relevant variables must affect 
    negatively to the prediction. To compute the selectivity 
    the features are ordered from most to lest relevant. 
    One by one the features are removed, set to zero for 
    example, and the residual errors are obtained to get the 
    area under the curve (AUC).
    """

    errors = []
    # for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):
    for i in tqdm(range(len(samples))):
        dxs, des = [], []
        xi = samples[i:i+1]
        ei = explainer(xi).values
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
        
    return np.nanmean(errors)

#%%

def acumen(explainer, samples, iterations=100, nstd=1.5, top_features=5, verbose=True):
    """
    Se eleccionan n puntos de los que tienen mayor score y se perturban creando una nueva
    serie (tp). También se crea otra serie (tr) metiendo ruido en otros n puntos aleatorios. 
    La importancia qm de puntos debe cumplir las siguiente regla qm(t) >= qm(tr) > qm(tp)
    """
    
    ranking = []
    # for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
    for i in tqdm(range(len(samples))):

        xi = samples[i:i+1] 
        base_exp = explainer(xi).values

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
            tc_exp = explainer(tc.reshape(xi.shape)).values

            # obtengo las n muestra más imporantes de x'
            aux = tc_exp.flatten()
            aux_max = aux.argsort().max()

            # ranking
            ranking.append((1- (np.argsort(aux).argsort()[top_mask] / aux_max)).mean())
        
    return np.nanmean(ranking)
#%%

X_dist = pd.DataFrame(squareform(pdist(X_test)))
Lime_dist = pd.DataFrame(squareform(pdist(lime_values))) # Lime values explanation matrix
shap_dist = pd.DataFrame(squareform(pdist(shap_values))) # Shape values explanation matrix

#Lime's metrics
list_metrics_shap = {}

list_metrics_shap['identity'] = identity(X_dist, shap_dist)
list_metrics_shap['separability'] = separability(X_dist, shap_dist)
list_metrics_shap['stability'] = stability(X_dist, shap_dist)
list_metrics_shap['coherence'] = coherence(model=predict_with_xgboost, explainer = shap_explainer,
                                           samples=X_test, targets=y_test)
list_metrics_shap['selectivity'] = selectivity(model=predict_with_xgboost, explainer = shap_explainer,
                                           samples=X_test)
list_metrics_shap['accumen'] = acumen(shap_explainer, X_test)

#%%
#Lime's metrics
list_metrics_lime = {}

list_metrics_lime['identity'] = identity(X_dist, Lime_dist)
list_metrics_lime['separability'] = separability(X_dist, Lime_dist)
list_metrics_lime['stability'] = stability(X_dist, Lime_dist)
list_metrics_lime['coherence'] = coherence(model=predict_with_xgboost, explainer = lime_explainers,
                                           samples=X_test, targets=y_test)
list_metrics_lime['selectivity'] = selectivity(model=predict_with_xgboost, explainer = lime_explainers,
                                           samples=X_test)
list_metrics_lime['accumen'] = acumen(lime_explainers, X_test)
