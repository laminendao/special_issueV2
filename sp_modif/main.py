# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:00:18 2024

@author: mlndao
"""

#%% Package's importation
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_tabular import RecurrentTabularExplainer
from tqdm import tqdm

from model_function import *
from methods import *
from data_prep import *
from evaluator import *
from SHAP import *
from L2X import *
#%% Function to generate explanations

# Function explanation for lime
def get_lime_explanation(data, e, L2X=False) :
    # e  = fn = model.predict
    # Iniatialisation
    df_expplanation = pd.DataFrame(columns=[str(i) for i in range(data.shape[1]*data.shape[2])])

    # Get explanations
    for row in range(data.shape[0]) : 
        explanation = lime_explainer.explain_instance(data[row],
                                                      e,
                                                      num_features=data.shape[1]*data.shape[2]) 
        # fn = model.predict, initialize lime_explainer = Reccurent()
        lime_values = explanation.local_exp[1]
        # Add explanation in df_explanation
        lime_dict = {}
        for tup in lime_values :
            lime_dict[str(tup[0])] = tup[1]
        df_expplanation.loc[len(df_expplanation)] = lime_dict
    
    return df_expplanation

# # Function explanation for others
def get_explainations(data, e, L2X = False) :
    
    # df diemnsion
    if L2X==True :
        X_to_def_col = data[0:1]
        explanation_test = e.explain(X_to_def_col.reshape((X_to_def_col.shape[0], -1)))
        num_columns = explanation_test.flatten().shape[0]
        
    else : 
        explanation_test = e.explain(data[0:1])
        num_columns = explanation_test.flatten().shape[0]
    
    # Iniatialisation
    df_expplanation = pd.DataFrame(columns=[str(i) for i in range(num_columns)])

    # Get explanations
    for row in range(data.shape[0]) :
        if L2X==True:
            X_row = data[row:row+1]
            explanation = e.explain(X_row.reshape((X_row.shape[0], -1)))
        else :
            explanation = e.explain(data[row:row+1])
        # Add explanation in df_explanation
        explanation = explanation.flatten()
        feature_dict = {}
        for i in range(num_columns) :
            feature_dict[str(i)] = explanation[i]
        df_expplanation.loc[len(df_expplanation)] = feature_dict
    
    return df_expplanation

#%% Data importation

# Data preparation
train, test, y_test = prepare_data('FD004.txt')
print(train.shape, test.shape, y_test.shape)
sensor_names = ['T20','T24','T30','T50','P20','P15','P30','Nf','Nc','epr','Ps30','phi',
                'NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']

remaining_sensors = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi',
                'NRf','NRc','BPR','htBleed','W31','W32'] # selection based on main_notebook

drop_sensors = [element for element in sensor_names if element not in remaining_sensors]

#%% Model fit
# # Modeling
if True:
    
#     print(elm)
    print('...')
    sequence_length = 40
    alpha = 0.5
    upper = 100
    learning_rate_ = 0.001
    dropout = 0.2
    activation = 'tanh'
    epochs = 20
    batch_size = 128
    train_recrul = rul_piecewise_fct(train, upper)
    weights_file = str(alpha) + '1lstm_hyper_parameter_weights.h5'
    input_shape = (sequence_length, len(remaining_sensors))
    model = create_lstm_1layer(dropout, activation, weights_file, input_shape=input_shape)
    mse = []

    X_train_interim = add_operating_condition(train_recrul)
    X_test_interim = add_operating_condition(test)

    X_train_interim, X_test_interim = condition_scaler(X_train_interim, X_test_interim, remaining_sensors)

    X_train_interim = exponential_smoothing(X_train_interim, remaining_sensors, 0, alpha=alpha)
    X_test_interim = exponential_smoothing(X_test_interim, remaining_sensors, 0, alpha=alpha)

    # create sequences train, test
    train_array = gen_data_wrapper(X_train_interim, sequence_length,remaining_sensors)
    label_array = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'])

    test_gen = (list(gen_test_data(X_test_interim[X_test_interim['Unit']==unit_nr], sequence_length,remaining_sensors, -99.))
               for unit_nr in X_test_interim['Unit'].unique())
    test_array = np.concatenate(list(test_gen)).astype(np.float32)
    print(train_array.shape, label_array.shape, test_array.shape)

    # create train-val split
    X_train_interim, X_test_interim = prep_data(train, test, drop_sensors, remaining_sensors, alpha)
    gss = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
    
    mse_val = []
    R2_val = []
    RMSE = []
    score_val = []
    test_rul = rul_piecewise_fct(y_test,upper)
    
    with tf.device('/device:GPU:0'):
        for train_unit, val_unit in gss.split(X_train_interim['Unit'].unique(), groups=X_train_interim['Unit'].unique()):
            train_unit = X_train_interim['Unit'].unique()[train_unit]  # gss returns indexes and index starts at 1
            train_split_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, train_unit)
            train_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'], train_unit)

            val_unit = X_train_interim['Unit'].unique()[val_unit]
            val_split_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, val_unit)
            val_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'], val_unit)

            # train and evaluate model
            model.compile(loss='mse', optimizer='adam')
#             model.load_weights(weights_file)  # reset optimizer and node weights before every training iteration

#             with tf.device('/device:GPU:0'):
            history = model.fit(train_split_array, train_split_label,
                                validation_data=(val_split_array, val_split_label),
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0)
#             mse.append(history.history['val_loss'][-1])
            test_gen = (list(gen_test_data(X_test_interim[X_test_interim['Unit']==unit_nr], sequence_length,remaining_sensors, -99.))
                            for unit_nr in X_test_interim['Unit'].unique())
            test_array = np.concatenate(list(test_gen)).astype(np.float32)
            mse_val.append(history.history['val_loss'][-1])
            y_hat_val_split = model.predict(test_array)
            R2_val.append(r2_score(test_rul, y_hat_val_split))
            RMSE.append(np.sqrt(mean_squared_error(test_rul, y_hat_val_split)))
            score_val.append(compute_s_score(test_rul, y_hat_val_split))
    d = {'RMSE_val':np.sqrt(mse_val),'RMSE_test': RMSE,'R2_test':np.mean(R2_val), 'Score_test':np.mean(score_val),
                'alpha':alpha, 'rul_piecewise':upper, 'sequence_length':sequence_length}

#%% Model performance

train_array_1c = train_array
test_array_1c = test_array
label_array_1c = label_array

print(" Model evaluation for FD004 : ")
y_hat_train = model.predict(train_array)
evaluate(label_array, y_hat_train, 'train')

y_hat_test = model.predict(test_array)
evaluate(y_test, y_hat_test)
print(' ')
print(" Model evaluation for FD004 : ")

y_hat_test = model.predict(test_array)
evaluate(y_test['RUL'].clip(upper=upper), y_hat_test)
print(' ')
print("S-score for test : ",compute_s_score(y_test, y_hat_test))
print(' ')

#%% Explanation
from tqdm import tqdm
df_metrics5 = pd.DataFrame()
# model = model_2c
for RD in tqdm(range(50)):
    # Echantillonage
    n_individus = test_array_1c.shape[0]

    # # Choisir aléatoirement 10 indices d'individus
    np.random.seed(RD)
    indices_choisis = np.random.choice(n_individus, size=10, replace=False, )

    # # Sélectionner les données correspondant aux indices choisis
    test_array_sampling = test_array_1c[indices_choisis, :, :]
    label_array_sampling = y_test.values[indices_choisis, :]

    # # Afficher les dimensions des données sélectionnées
    print(test_array_sampling.shape, label_array_sampling.shape)
    # distance matrix XX'
    X_dist = pd.DataFrame(squareform(pdist(test_array_sampling.reshape((test_array_sampling.shape[0], -1)))))

    if True :
        # LIME
        lime_explainer = RecurrentTabularExplainer(test_array, training_labels=label_array,
                                                           feature_names=remaining_sensors,
                                                           mode = 'regression',
                                                           )
        lime_values = get_lime_explanation(test_array_sampling, e = model.predict)
        Lime_dist = pd.DataFrame(squareform(pdist(lime_values))) # Lime values explanation matrix

        #Lime's metrics
        list_metrics_lime = {}
        list_metrics_lime['identity'] = identity(X_dist, Lime_dist)
        list_metrics_lime['separability'] = separability(X_dist, Lime_dist)
        list_metrics_lime['stability'] = stability(X_dist, Lime_dist)
        list_metrics_lime['coherence'], list_metrics_lime['completness'], list_metrics_lime['congruence'] = coherence(model=model.predict, 
                                                    explainer = get_lime_explanation,
                                                   samples=test_array_sampling,
                                                    targets=label_array, e = model.predict)
        list_metrics_lime['selectivity'] = selectivity(model=model.predict, explainer = get_lime_explanation,
                                                   samples=test_array_sampling, e_x = model.predict)
        list_metrics_lime['accumen'] = acumen(get_lime_explanation, test_array_sampling, e=model.predict)
        list_metrics_lime['Verm_stability'] = stability_Velmurugan(get_lime_explanation, test_array_sampling,
                                                                   e=model.predict, top_features=200)
        list_metrics_lime['fidelity'], list_metrics_lime['sparsity'] = fidelity(model=model.predict, 
                                                    explainer = get_lime_explanation,
                                                    samples=test_array_sampling,
                                                    e = model.predict, L2X=True)
        list_metrics_lime['instability'] = instability(model=model.predict, 
                                                    explainer = get_lime_explanation,
                                                    samples=test_array_sampling,
                                                    e = model.predict, L2X=True)
        list_metrics_lime['explainer'] = 'lime_4c'
        df_metrics5 = pd.concat([df_metrics5, pd.DataFrame([list_metrics_lime])])
        print("Lime OK!!")

        # SHAP
        e = KernelSHAP(model)
        shapvalues = get_explainations(test_array_sampling, e)
        shapvalues.shape

        list_metrics_shap = {}
        shap_dist = pd.DataFrame(squareform(pdist(shapvalues))) # shap values explanation matrix

        list_metrics_shap['identity'] = identity(X_dist, shap_dist)
        list_metrics_shap['separability'] = separability(X_dist, shap_dist)
        list_metrics_shap['stability'] = stability(X_dist, shap_dist)
        list_metrics_shap['coherence'], list_metrics_shap['completness'], list_metrics_shap['congruence'] = coherence(model=model.predict, 
                                                    explainer = get_explainations,
                                                   samples=test_array_sampling,
                                                    targets=label_array, e = e)
        list_metrics_shap['selectivity'] = selectivity(model=model.predict, explainer = get_explainations,
                                       samples=test_array_sampling, e_x=e)
        list_metrics_shap['accumen'] = acumen(get_explainations, test_array_sampling, e=e)
        list_metrics_shap['Verm_stability'] = stability_Velmurugan(get_explainations, test_array_sampling,
                                                                   e=e, top_features=200)
        list_metrics_shap['fidelity'], list_metrics_shap['sparsity']= fidelity(model=model.predict, 
                                                    explainer = get_explainations,
                                                    samples=test_array_sampling,
                                                    e = e)
        list_metrics_shap['instability']= instability(model=model.predict, 
                                                    explainer = get_explainations,
                                                    samples=test_array_sampling,
                                                    e = e)

        list_metrics_shap['explainer'] = 'shap_4c'
        df_metrics5 = pd.concat([df_metrics5, pd.DataFrame([list_metrics_shap])])
        print("shap OK !!")

        # L2X
        e = L2X(model.predict, test_array_sampling)
        l2xvalues = get_explainations(test_array_sampling, e, L2X=True)
        l2xvalues.shape

        # l2x's metrics
        list_metrics_l2x = {}
        l2x_dist = pd.DataFrame(squareform(pdist(l2xvalues))) # Lime values explanation matrix

        list_metrics_l2x['identity'] = identity(X_dist, l2x_dist)
        list_metrics_l2x['separability'] = separability(X_dist, l2x_dist)
        list_metrics_l2x['stability'] = stability(X_dist, l2x_dist)
        list_metrics_l2x['coherence'], list_metrics_l2x['completness'], list_metrics_l2x['congruence'] = coherence(model=model.predict, explainer = get_explainations,
                                                   samples=test_array_sampling, targets=label_array_sampling, e = e, L2X=True)
        list_metrics_l2x['selectivity'] = selectivity(model=model.predict, explainer = get_explainations,
                                       samples=test_array_sampling, e_x=e, L2X=True)
        list_metrics_l2x['accumen'] = acumen(get_explainations, test_array_sampling, e=e, L2X=True)
        list_metrics_l2x['Verm_stability'] = stability_Velmurugan(get_explainations, test_array_sampling,
                                                                   e=e, top_features=200, L2X=True)
        list_metrics_l2x['fidelity'], list_metrics_l2x['sparsity']= fidelity(model=model.predict, 
                                                    explainer = get_explainations,
                                                    samples=test_array_sampling,
                                                    e = e, L2X=True)
        list_metrics_l2x['instability'] = instability(model=model.predict, 
                                                    explainer = get_explainations,
                                                    samples=test_array_sampling,
                                                    e = e, L2X=True)
        list_metrics_l2x['explainer'] = 'l2x_4c'
        print("L2X OK !!")

        df_metrics5 = pd.concat([df_metrics5, pd.DataFrame([list_metrics_l2x])])
df_metrics5     


#%% 
e = KernelSHAP(model)
# l2xvalues = get_explainations(test_array_sampling, e, L2X=True)
# l2xvalues.shape        
list_metrics_shap['fidelity'], list_metrics_shap['sparsity']= fidelity(model=model.predict, 
                                            explainer = get_explainations,
                                            samples=test_array_sampling,
                                            e = e)
#%
#%%
a = np.array([1, 5, 20, 25])
b = np.array([1, 5, 5, 25])

np.linalg.norm(a-b, ord=1)