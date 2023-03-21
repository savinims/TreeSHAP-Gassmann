 # -*- coding: utf-8 -*-
"""
@author: Savini Samarasinghe
"""

import data_helpers as data_helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmasher as cmr
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
# import time
# import os
# from sklearn.inspection import permutation_importance
# from xgboost.sklearn import XGBRegressor
input_folder_path = '../Inputs'
output_folder_path = '../Outputs_no_file'

# =============================================================================
# Load and preprocess data
# =============================================================================

rho_fluid = 1
k_fluid = 2.2

k_quartz = 38  # Carmichael 1990
k_feldspar = 65  # Blangy's thesis
k_mica = 52  # Blangy's thesis
k_clay = 25  # Gluf clays (Han) Mavko's rock physics handbook

blangy_df = data_helpers.get_blangy_dataframe(
    input_folder_path, rho_fluid, k_fluid, k_quartz, k_clay, k_mica, k_feldspar)
strandenes_df = data_helpers.get_strandenes_df(
    input_folder_path, rho_fluid, k_fluid, k_quartz, k_clay, k_mica, k_feldspar)
han_df = data_helpers.get_han_df(
    input_folder_path, rho_fluid, k_quartz, k_clay, k_mica, k_feldspar, k_fluid)
mapeli_df = data_helpers.get_mapeli_df(
    input_folder_path, rho_fluid, k_quartz, k_clay, k_mica, k_feldspar, k_fluid)
yin_df = data_helpers.get_yin_df(input_folder_path, rho_fluid, k_quartz, k_clay,
                                  k_mica, k_feldspar, k_fluid)  # samples are brine saturated, but
# we are using rho_fluid = rho_water for the calculations

dataframes = [blangy_df, strandenes_df, han_df, mapeli_df, yin_df]
df_descriptions = ['Blangy, 1992', 'Strandenes, 1991',
                    'Han, 1986', 'Mapeli, 2018', 'Yin, 1993']
dataset_names =  ['Blangy', 'Strandenes',
                    'Han', 'Mapeli', 'Yin']
# dataframes = [blangy_df, strandenes_df, han_df,  yin_df]
# df_descriptions = ['Blangy, 1992', 'Strandenes, 1991',
#                     'Han, 1986', 'Yin, 1993']
# dataset_names =  ['Blangy', 'Strandenes',
#                     'Han',  'Yin']
all_data_df = pd.DataFrame()

for idx, df in enumerate(dataframes):

    print(df_descriptions[idx])
    data_helpers.print_k_sat_error(df)
    data_helpers.draw_cross_plot(x_data=df['k_sat'], y_data=df['k_sat_gassmann'], x_label='Measured K sat', y_label='Gassmann estimate',
                                  colorbar_label='|Measured - Gassmann|', show=False, color_by_var=np.abs(df['k_sat']-df['k_sat_gassmann']))
    plt.title(df_descriptions[idx])
    plt.show()

    # concatenate all data into a single dataframe
    df['file_info'] = idx+1
    all_data_df = pd.concat([all_data_df, df])

## Some EDA
error = all_data_df['k_sat']- all_data_df['k_sat_gassmann']
file_info =all_data_df['file_info']
plot_vars = [all_data_df['pressure'], all_data_df['porosity'], all_data_df['k_dry'], all_data_df['grain_density'], all_data_df['k_mineral'], all_data_df['g_dry']/all_data_df['g_sat'], all_data_df['clay']]
plot_vars_txt = ['pressure', 'porosity', 'K dry', 'grain density', 'K mineral', 'G dry/G sat', 'caly content']


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
file_names = [dataset_names[int(x-1)] for x in list(file_info)]
   
for i, qty in enumerate(plot_vars):
    plt.figure()
    sns.scatterplot(x = error, y = qty, hue = file_names, alpha = 0.7)
    plt.grid()
    plt.xlabel('K sat - K sat Gassmann')
    plt.ylabel(plot_vars_txt[i])
    plt.show()


# =============================================================================
# Get the data ready for machine learning
# =============================================================================


file_as_feature = True # to investigate experiment specific biases
num_repeats = 10
test_size = 0.1
nonlinear_method = 'random_forest'
assert(nonlinear_method in ['svr','kernel_ridge','random_forest','xgboost', 'decision_tree'])
input_features = ['pressure', 'porosity','k dry', 'k mineral','grain density', 'g ratio']
num_measurement_features = len(input_features)
standardize_inputs = True
decorrelate_inputs = False # use PCA to decorrelate measurement data


if file_as_feature:
    [input_features.append(name) for name in dataset_names]
                    

all_shap_values = []
all_X_test = []
all_y_test = []
all_shap_interactions = []

for random_state in range(num_repeats):
    input_data = all_data_df[['pressure', 'porosity',
                              'k_dry', 'k_mineral', 'grain_density']].copy()
    input_data['g_ratio'] = all_data_df['g_dry']/all_data_df['g_sat']
    output_data = all_data_df['k_sat'] - all_data_df['k_sat_gassmann']
    #output_data = all_data_df['vp_sat'] - all_data_df['vp_sat_gassmann']
    
    # Standardize the inputs
    if standardize_inputs:
        scaler = StandardScaler(copy=True)
        scaler.fit(input_data.values)
        scaled_input_data = scaler.transform(input_data.values)
    else:
        scaled_input_data = input_data
    if decorrelate_inputs :
        pca = PCA()
        pca.fit(scaled_input_data )
        
        pc_names = ["PC" + str(x + 1) for x in range(pca.components_.shape[0])]
        x_pca = pca.transform(scaled_input_data)
        loadings = pd.DataFrame(pca.components_.T, columns=pc_names, index = input_features[:num_measurement_features])
        X_dummy = np.concatenate(
            (x_pca, all_data_df['file_info'].values[:, np.newaxis]), axis=1)
    else:
        X_dummy = np.concatenate(
            (scaled_input_data, all_data_df['file_info'].values[:, np.newaxis]), axis=1)
    
    
    # A % of data from each dataset used for validation
    output_data_binary = (output_data >=0)*1
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, output_data_binary, random_state=random_state,test_size=test_size, stratify=output_data_binary)
    
    
    
    # # becuase the different experiments can have biases and limitations that are not 
    # # necessarily associated to the Gassman's model, we can also include the data source as an additional input feature. The file number will be one hot encoded as a categorical variable.
    # here I will use onehotencoding instead of binary because it will be easier for the explainability methods 
    if file_as_feature:
        onehot_encoder = OneHotEncoder(sparse=False)
        train_file_info = onehot_encoder.fit_transform(np.expand_dims(X_train[:,-1], axis = 1))
        test_file_info = onehot_encoder.fit_transform(np.expand_dims(X_test[:,-1], axis = 1))
        X_train = np.hstack((X_train[:, :-1],train_file_info))
        X_test = np.hstack((X_test[:, :-1],test_file_info))
    else:        
        X_train = X_train[:, :-1]
        X_test = X_test[:, :-1]


