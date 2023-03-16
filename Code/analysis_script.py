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
output_folder_path = '../Outputs'

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

dataset_names =  ['Blangy', 'Strandenes',
                    'Han', 'Mapeli', 'Yin']
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

positive_error_only = False # to discard negative error samples
file_as_feature = True # to investigate experiment specific biases
num_repeats = 25
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
    
    
    if positive_error_only:
        data_id = (output_data>=0)       
        X_train, X_test, y_train, y_test = train_test_split(X_dummy[data_id,:], output_data[data_id], random_state=random_state,test_size=test_size, stratify=all_data_df['file_info'][data_id])
        # A % of data from each dataset used for validation
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_dummy, output_data, random_state=random_state,test_size=test_size, stratify=all_data_df['file_info'])
    
    
    
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

        
       
    # =============================================================================
    # Linear Regression
    # =============================================================================
    
    # if file is used the R2 squared is about 0.5, if not about 0.3 for the linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    coeff = model.coef_
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    # print the coefficients
    for i,feature in enumerate(input_features):
        print("Coefficient for {} = {}".format(feature, coeff[i]))
        
    
    data_helpers.draw_cross_plot(x_data=y_train, y_data=train_prediction, x_label='Ground truth', y_label='Prediction',
                                  colorbar_label='|GT - Prediction|', show=False, color_by_var=np.abs(y_train-train_prediction))
    plt.title('Training R squared = {}'.format(model.score(X_train, y_train)))
    
    data_helpers.draw_cross_plot(x_data=y_test, y_data=test_prediction, x_label='Ground truth', y_label='Prediction',
                                  colorbar_label='|GT - Prediction|', show=False, color_by_var=np.abs(y_test-test_prediction))
    plt.title('Testing R squared = {}'.format(model.score(X_test, y_test)))

    
       
    # =============================================================================
    # Non linear regression    
    # =============================================================================
    
    # most authors advocate against the use of the R2 in nonlinear regression 
    #analysis and recommend alternative measures, such as the Mean Squared Error 
    #(MSE; see Ratkowsky, 1990) or the AIC and BIC    
    if nonlinear_method == 'svr':
        model = SVR
        search = GridSearchCV(model(),
                    param_grid={"C": [1e0, 1e1, 1e2, 1e3], "epsilon": np.logspace(-2, 2, 5),
                                "kernel":["rbf"]}, 
                    scoring =  'neg_mean_squared_error', 
                    n_jobs = 12, cv = 10)

    elif nonlinear_method == 'kernel_ridge':
        model = KernelRidge
        search = GridSearchCV(model(),
        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5),
                    "kernel":["rbf"]},
        scoring =  'neg_mean_squared_error',
        n_jobs = 12, cv = 10 )
    elif nonlinear_method == 'decision_tree':
        model = DecisionTreeRegressor
        search= GridSearchCV( model(),
         param_grid={'max_depth':np.arange(2,21,2),
                     'min_samples_split':np.arange(2,10),
                     'min_samples_leaf':np.arange(1,8)}, 
         scoring =  'neg_mean_squared_error',
         n_jobs = 12, cv = 10)
    elif nonlinear_method == 'random_forest':    
         model = RandomForestRegressor
         search = GridSearchCV(model(), 
         param_grid={'max_depth':np.arange(5,21), 
                     'n_estimators':[25,50,75,100,200],
                     'min_samples_leaf':np.arange(1,5)}, 
         n_jobs = 12, cv = 5,
         scoring = 'neg_mean_squared_error')
                

    search.fit(X_train, y_train)
    print(f"Best model params: {search.best_params_} and MSE score: {search.best_score_:.3f}")

    optimal_model = model(**search.best_params_)
    optimal_model.fit(X_train,y_train)
    print('Pseudo R squared on training data= {}'.format(optimal_model.score(X_train, y_train)))
    print('Pseudo R squared on testing data= {}'.format(optimal_model.score(X_test, y_test)))
    
    
    if((nonlinear_method == 'decision_tree') | (nonlinear_method == 'random_forest')):    
        explainer = shap.Explainer(optimal_model)
        
        shap_values_test = explainer.shap_values(X_test)
        
        
        if decorrelate_inputs:
            # aggregate shap values using the pca loadings to investigate the attributions of the original 
            # measurement variables
            pca_shap_values = shap_values_test[:,:num_measurement_features]
            shap_values = shap_values_test
            for i in range(num_measurement_features):
                feature_loadings = loadings.loc[input_features[i]].values
                shap_values[:,i] = pca_shap_values@feature_loadings

        else:
            shap_values = shap_values_test
        
        my_cmap = plt.get_cmap('viridis')
        plt.figure()
        shap.summary_plot(shap_values,X_test,
                          feature_names=input_features, show=False)
        plt.title('Shap Values on test data')
        # Change the colormap of the artists
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if hasattr(fcc, "set_cmap"):
                    fcc.set_cmap(my_cmap)
        plt.show()
                    
     
        all_shap_values.append(shap_values)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    else:
        raise NotImplementedError('Kernel SHAP required for models that are not based on trees')

    
plot_data = np.array(all_X_test).reshape(-1,len(input_features))
inverse_transformed_data = scaler.inverse_transform(plot_data[:,:6])
if file_as_feature:
    rescaled_X_test = np.hstack((inverse_transformed_data, plot_data[:,6:]))
else:
    rescaled_X_test = inverse_transformed_data
    
y_data = np.array(all_y_test).reshape(-1,1)
shap = np.array(all_shap_values).reshape(-1,len(input_features))    
    
    
for i in range(num_measurement_features):
    plt.figure()
    plt.grid()
    if (input_features[i] == 'g ratio'):
        c_delta = np.max([np.max(rescaled_X_test[:,i])-1, 1-np.min(rescaled_X_test[:,i])])
        c_delta = np.round(c_delta,1)
        plt.scatter(shap[:,i], y_data, c = rescaled_X_test[:,i], cmap = cmr.fusion_r, alpha = 0.8, vmin = 1-c_delta, vmax = 1+c_delta)
    else:
        plt.scatter(shap[:,i], y_data, c = rescaled_X_test[:,i], cmap = cmr.dusk, alpha = 0.8)
    
    plt.xlabel('SHAP values')
    plt.ylabel('K sat - K sat Gassmann')
    plt.colorbar(label =input_features[i] )
    plt.savefig(output_folder_path+'/Shap_'+input_features[i]+'.png', dpi =300)
    plt.show()
    
    
    
for i in range(num_measurement_features, len(input_features) ):
    plt.figure()
    plt.grid()

    plt.scatter(shap[:,i], y_data, c = plot_data[:,i], cmap = 'RdGy', alpha = 0.8)
    
    plt.xlabel('SHAP values')
    plt.ylabel('K sat - K sat Gassmann')
    plt.colorbar(label =input_features[i] )
    plt.savefig(output_folder_path+'/Shap_'+input_features[i]+'.png', dpi =300)
    plt.show()
    
    
# =============================================================================
#     Mean absolute SHAP plot
# =============================================================================

mean_abs_shap_values = np.mean(np.abs(shap), axis = 0)
sort_idx = np.argsort(mean_abs_shap_values)
std_abs_shap_values = np.std(np.abs(shap), axis = 0)

plt.figure()
plt.barh(np.array(input_features)[sort_idx],mean_abs_shap_values[sort_idx], xerr = std_abs_shap_values[sort_idx], color = 'grey', edgecolor = 'grey', label = 'mean abs SHAP value')
plt.legend()
plt.savefig(output_folder_path+'/Shap_mean_abs.png', dpi =300)
plt.show()