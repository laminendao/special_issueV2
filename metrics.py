# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:54:34 2024

@author: mlndao
"""

from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
import stability_forlder as st
from methods import * 


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
        
        return np.nanmean(errors)

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
        
        return np.nanmean(errors)

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
    
    return np.nanmean(errors)

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
#     errors = 1 - (pred - targets) ** 2
    errors = (pred - targets) ** 2 # modified

    exp = np.array(explains).reshape(samples.shape)

    explains = np.array(explains).reshape(samples.shape)
    exp_pred = model(explains) / tmax
#     exp_errors = 1- (exp_pred - targets) ** 2
    exp_errors = (exp_pred - targets) ** 2 #modified
    
    coherence_i = np.abs(errors - exp_errors)
    coherence = np.mean(coherence_i)
    completeness = min(np.mean(exp_errors / errors), np.mean(errors / exp_errors))
    congruency = np.sqrt(np.mean((coherence_i - coherence)**2))
    
#     return {
#             'coherence': coherence, 
#             'completeness':np.mean(exp_errors / errors),
#             'congruency': np.sqrt(np.mean((coherence_i - coherence)**2))
#            }
    return coherence, completeness, congruency

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
    for i in range(len(samples)):
        # dxs, des = [], []
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
#         print('ok')
        
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
    for i in range(len(samples)):

        xi = samples[i:i+1] 
        #pred
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
            # pred_

            # I obtain the n most important samples of x'.
            aux = tc_exp.flatten()
            aux_max = aux.argsort().max()

            # ranking
            ranking.append((1- (np.argsort(aux).argsort()[top_mask] / aux_max)).mean())
        
    return np.nanmean(ranking)

def stability_Velmurugan(explainer, samples, e, top_features=5, verbose=True, L2X = False, nb_iter = 10):
    
    idx_ind = [i for i in range(samples.shape[0])]
    feat_list = [i for i in range(560)] # tw*nb_feature à rectifier
    list_stab_values = []
    
    for ind in idx_ind:
        list_iter = [samples[ind:ind+1] for i in range(nb_iter)]
        Z = []
        for d in list_iter : 
            ex = explainer(d, e = e, L2X=L2X) # Get explanation

            exp_abs = np.abs(ex) # get Zi
            each = exp_abs 
            weighted = list(each.values.flatten())
            nb_feature = 10
            weighted.sort()
            max_feat = weighted[-nb_feature]

            Zi = [0]*len(feat_list)
            for i, w in enumerate(each.values.flatten()) : 
                if w>=max_feat:
                    Zi[i] = 1
            print(sum(Zi))
            Z.append(Zi)
        stab_value = st.getStability(Z) # Calculate Stability
        list_stab_values.append(stab_value)
        
    return np.mean(list_stab_values)

def fidelity_1(model, explainer, samples, e, verbose=True, L2X = False, nb_iter = 10, nb_feature = 10):
    '''

    Parameters
    ----------
    explainer : FUNC
        DESCRIPTION. it return features importance
    samples : np or pd datatyp
        DESCRIPTION. it contains all indiviuds that we want to calculate the explanations's fidelity
    e : TYPE function
        DESCRIPTION. Local explainers like LIME or SHAP
    nb_iter : INT, optional
        DESCRIPTION. Nulber of repetition to compute MAPE. The default is 10.
    verbose : BOOLEEN, optional
        DESCRIPTION. The default is True.
    L2X : FUNC, optional
        DESCRIPTION. It check if the local explainer is L2X or not. The default is False.

    Returns FLOAT. fidelity
    -------

    '''
    idx_ind = [i for i in range(samples.shape[0])]
    feat_list = [i for i in range(560)] # tw*nb_feature à rectifier
    list_stab_values = []
    
    for ind in tqdm(idx_ind):
        list_iter = [samples[ind:ind+1] for i in range(nb_iter)]
        Z = []
        for xi in list_iter : 
            ex = explainer(xi, e = e, L2X=L2X) # Get explanation

            exp_abs = np.abs(ex) # get Zi
            each = exp_abs 
            weighted = list(each.values.flatten())
            weighted.sort()
            max_feat = weighted[-nb_feature]

            # Zi = [0]*len(feat_list)
            # for i, w in enumerate(each.values.flatten()) : 
            #     if w>=max_feat:
            #         Zi[i] = 1
            # Z.append(Zi)
            
            x_bar = xi_flatten = xi.flatten()
            for i, w in enumerate(each.values.flatten()):
                if w <= max_feat:
                    x_bar[i] = xi_flatten[i] + np.random.normal()
                    
            x_bar = np.copy(x_bar).reshape(xi.shape)
            
            y_pre = model(xi).values
            y_pre_bar = model(x_bar).values
            
            Y.append(y_pre)
            Y_bar.append(y_pre_bar)
        Mape_y =  compute_MAP(np.array(Y),np.array(Y_bar))
    return np.mean(Mape_y)


def fidelity(model, explainer, samples, e, nstd=1.5, top_features=10, verbose=True, L2X = False, rd=0, n_iter = 10):
    '''
    model : model.predic
    explainer : get_explanation's function
    sample : data
    target : label
    e : e like KenrelSHAP
    '''
    
    # for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
    # y_pred = []
    # X_bar = np.zeros(samples.shape)
    score_ = []
    for i in range(len(samples)):
        y = []
        y_pred = []
        for k in range(n_iter):
            np.random.seed(k)
            xi = samples[i:i+1] 
            exp = explainer(xi, e, L2X).values
            exp = np.abs(exp) #ajout : take absolute valeur of the feature importance
            _mean, _std = exp.mean(), exp.std()
            theshold = _mean + nstd*_std
            nsamples = (exp > theshold).sum()
            nsamples = min(nsamples, top_features)
            aux = exp.flatten()
            theshold = aux[aux.argsort()][-nsamples]
            
            x_bar = xi.flatten()
            x_bar[(aux <= theshold)] = x_bar[(aux <= theshold)] + np.random.normal()
            
            x_bar = x_bar.reshape(xi.shape)
            # X_bar[i:i+1] = x_bar
            y.append(model(xi))
            y_pred.append(model(x_bar))
        score_.append(compute_MAPE(np.array(y), np.array(y_pred)))
        # Y = model(xi)
        # y_pred.append(model(x_bar))
    # Y_pred = np.array(y_pred)
    # Y_pred = model(X_bar)
    # Y = model(samples)
    M_tot = samples.shape[1]*samples.shape[2]
    
    return 1-np.mean(score_)/100, nsamples/M_tot

def instability(model, explainer, samples, e, verbose=True, L2X = False, rd=0, n_iter = 10):
    '''
    model : model.predic
    explainer : get_explanation's function
    sample : data
    target : label
    e : e like KenrelSHAP
    '''
    
    # for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
    # y_pred = []
    # X_bar = np.zeros(samples.shape)
    score_ = []
    for i in range(len(samples)):
        instab = []
        for k in range(n_iter):
            np.random.seed(k)
            xi = samples[i:i+1] 
            x_bar = xi + np.random.normal()
            exp_xi = explainer(xi, e, L2X).values
            exp_xi_bar = explainer(x_bar, e, L2X).values           
            instab.append(np.linalg.norm(exp_xi-exp_xi_bar, ord=1))

        score_.append(np.mean(instab))
    
    return (np.mean(score_) - np.min(score_))/(np.max(score_)-np.min(score_))
#%%

def consistency(explainer, samples, e, top_features=5, verbose=True, L2X = False, nb_iter = 10):
    pass

# def sparsity(explainer, samples, e, top_features=5, verbose=True, L2X = False, nb_iter = 10):
#     pass

def faithfullness():
    pass

#%% test
# xi = np.array([2, 5, 44, 22, 6, 17, 74, 100])
# top_features = 3
# exp = np.array([1, 25, 40, 5, 78, 10, 74, 10])
# weighted =np.sort(each)

# thresodl = weighted[-nb_feature]

# x_bar = xi.astype(float)
# for i in range(len(xi)):
#     if xi[i]<=thresodl:
#         x_bar[i] = xi[i] + np.random.normal()

#%% Comp

# _mean, _std = exp.mean(), exp.std()
# theshold = _mean + nstd*_std
# nsamples = (exp > theshold).sum()
# nsamples = min(nsamples, top_features)
# aux = exp.flatten()
# theshold = aux[aux.argsort()][-nsamples]

# x_bar = xi_flatten = xi.flatten().astype(float)
# x_bar[(exp > theshold)] = x_bar[(exp > theshold)] + np.random.normal()     
       
#%%