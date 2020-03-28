import pickle
%load_ext autoreload
%autoreload 2
import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns
import argparse, os
import gpflow
from gpflow import kernels
from likelihoods import *
from data import *
from models import MLSVGP, BASESVGP
from meta_loop import meta_loop
from main import MAIN
from run_main import *
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
################################################
# Data is removed due to the confidentiality

################################################

# observation_train = pickle.load(open('Inventory_norm_train70.pickle', 'rb'))
# observation_test = pickle.load(open('Inventory_norm_test70.pickle', 'rb'))
# observation_train = np.log(observation_train)
# observation_test = np.log(observation_test)
# prices = pickle.load(open('ActualPrices.pickle', 'rb'))
# prices_random_train = prices[0 : observation_train.shape[1]].reshape(-1,1)
# prices_random_test = prices[observation_train.shape[1]+1 :].reshape(-1,1)
# C = np.array(pickle.load(open('LeasedSpace.pickle', 'rb')))
# ids_train = np.zeros((observation_train.shape[0], observation_train.shape[1]))
# ids_test = np.zeros((observation_test.shape[0], observation_test.shape[1]))


# capacity_train = np.zeros((observation_train.shape[0], observation_train.shape[1]))
# capacity_test = np.zeros((observation_test.shape[0], observation_test.shape[1]))

# actual_price_train = np.zeros((observation_train.shape[0], observation_train.shape[1]))
# actual_price_test = np.zeros((observation_test.shape[0], observation_test.shape[1]))

# for i in range(observation_train.shape[0]):
#     for j in range(observation_train.shape[1]):       
#         ids_train[i, j]= i
#         capacity_train[i, j] = C[i]
#         actual_price_train[i,j] = prices_random_train[j]
        
# for i in range(observation_test.shape[0]):
#     for j in range(observation_test.shape[1]):       
#         ids_test[i, j]= i
#         capacity_test[i, j] = C[i]
#         actual_price_test[i,j] = prices_random_test[j]
        
# prior_observation_train = observation_train[:, :-1].reshape(-1,1)
# prior_observation_test = np.concatenate((observation_train[:, -1:], observation_test[:, :-1]), axis =1).reshape(-1,1)


# actual_observation_train = observation_train[:, 1:].reshape(-1,1)
# actual_observation_test =  observation_test[:, :].reshape(-1,1)

# User_capacity_train = capacity_train[:, :-1].reshape(-1,1)
# User_capacity_test = np.concatenate((capacity_train[:, -1:], capacity_test[:, :-1]), axis =1).reshape(-1,1)

# P_train = actual_price_train[:, 1:].reshape(-1,1)
# P_test = actual_price_test[:, :].reshape(-1,1)

# actual_ids_train =  ids_train[:, :-1].reshape(-1,1)
# actual_ids_test =  ids_test.reshape(-1,1)
# print(prior_observation_test.shape, User_capacity_test.shape, P_test.shape , actual_observation_test.shape , actual_ids_test.shape)

# dataset_train= np.hstack((prior_observation_train, User_capacity_train, P_train , actual_observation_train , actual_ids_train))

# dataset_test= np.hstack((prior_observation_test, User_capacity_test, P_test , actual_observation_test , actual_ids_test))


# structured learning 

def basestocklevel(dataset):
    inj_data = []
    withd_data = []
    for i in range(1, len(dataset)):
        if dataset[i, 3] > dataset[i, 0]:
            inj_data.append(dataset[i, :])
        elif dataset[i, 3] < dataset[i, 0]:
            withd_data.append(dataset[i, :])
    return inj_data, withd_data


Injection_Data,  Withdrawal_Data= basestocklevel(dataset_train)
Injection_Data_arr = np.vstack(Injection_Data)
Withdrawal_Data_arr = np.vstack(Withdrawal_Data)



RMSerror_SML = []

RMSerror = 0

def create_model(**kwargs):
    if kwargs["model_name"] == "MLSVGP": 
        kernel = kernels.RBF(kwargs["dim_in"] + kwargs["dim_h"], ARD=True)
        Z= np.random.randn(kwargs["n_inducing"], kwargs["dim_in"]+kwargs["dim_h"])
        mean_func = None
        likelihood = MultiGaussian(dim=kwargs["dim_out"])
        model = MLSVGP(
              dim_in=kwargs["dim_in"], dim_out=kwargs["dim_out"],
                dim_h=kwargs["dim_h"], num_h=kwargs["n_env"],
                kern=kernel, likelihood=likelihood, mean_function=mean_func,
                Z=Z, name=kwargs["model_name"])
    return model

#number of inducing points {10, 15, 20, 25, 30, 35}
num = 10
kwargs= {"n_active_tasks": 4, "learning_rate": 0.01, "n_inducing" : num,  "batch_size": 10,
        "train_steps":400, "infer_steps": 100, "dim_in": 3, "dim_out":1, "dim_h": 1, "n_env":4, "meta_test_iters":0,
        "model_name": "MLSVGP","seed":1, "meta_train_iters": 20, "n_train_tasks":4}



model_SML = create_model(**kwargs)

Inference_inj_SML = MAIN(model_SML, Injection_Data_arr, **kwargs) 
Inference_wid_SML = MAIN(model_SML, Withdrawal_Data_arr, **kwargs) 

run_main(Inference_inj_SML, Injection_Data_arr, Injection_Data_arr, **kwargs) 
run_main(Inference_wid_SML, Withdrawal_Data_arr, Withdrawal_Data_arr, **kwargs) 

N = observation_test.shape[0]
M = observation_test.shape[1]
xx_test= np.zeros((N,M,3))
for j in range(N):
    for dim in range(M):
        xx_test[j,dim,0] = observation_test[j, dim]
        xx_test[j,dim,1] = capacity_test[j, dim]
        xx_test[j,dim,2] = actual_price_test[j, dim]

def inventory_pred(xx, ids):
    count = 0
    n_Trials = 10000
    y_pred = np.zeros((n_Trials, N, M))
    base_w_mean, base_w_var, h_w_mean, hw_var  = Inference_wid_SML.predict(xx, ids, True)
    base_inj_mean, base_inj_var, h_inj_mean, h_inj_var  = Inference_inj_SML.predict(xx, ids, True)
    print("base_w_mean", base_w_mean)
    print("base_inj_mean", base_inj_mean)
    count = 0
    for j in range(N):
        for i in range(M):
            for k in range(n_Trials):
                base_inj = np.random.normal(base_inj_mean[j, i, 0], np.sqrt(base_inj_var[j, i, 0]), 1)
                base_w = np.random.normal(base_w_mean[j, i, 0], np.sqrt(base_w_var[j, i, 0]), 1)
                if base_w > base_inj:
                    if xx_test[j, i, 0] <= base_inj:  
                        y_pred[k, j, i] =  base_inj
                    elif xx_test[j, i, 0] >= base_w:
                        y_pred[k, j, i] =  base_w
                    else:
                        y_pred[k, j, i] = xx[j, i, 0]
                else:
                    if xx_test[j, i, 0] <= base_w:  
                        y_pred[k, j, i] =  base_w
                    elif xx_test[j, i, 0] >= base_inj:
                        y_pred[k, j, i] =  base_inj
                    else:
                        y_pred[k, j, i] = xx[j, i, 0]   
    return y_pred
y_pred_SML = inventory_pred(xx_test, actual_ids_test.reshape(-1))

experiment_path = "Proportional inventory-experiments-SML/70/"
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
if not os.path.exists(experiment_path + "/experiment with {} inducing points/".format(num)):
    os.makedirs(experiment_path + "/experiment with {} inducing points/".format(num))
config_path = experiment_path + "/experiment with {} inducing points/ config.json".format(num)
with open(config_path, "w") as f:
    json.dump(kwargs, f)

mean_y_pred_SML = np.mean(y_pred_SML, axis = 0)
std_y_pred_SML = np.std(y_pred_SML, axis = 0)

img_path = experiment_path + "/experiment with {} inducing points/".format(num)
if not os.path.exists(img_path):
    os.makedirs(img_path)
config_path = experiment_path + "/experiment with {} inducing points/ RMSerror.json".format(num)
n_Trials = 10000

#########################################
max_obs = 0
min_obs = float('inf')

for i in range(N):
    if max_obs < max(observation_test[i, :]):
        max_obs =  max(observation_test[i, :])
    if min_obs > min(observation_test[i, :]):
        min_obs =  min(observation_test[i, :])
#############################################################

for k in range(n_Trials):
    RMSerror += sum(sum((observation_test[:, :] - y_pred_SML[k, :, :])**2))

RMSerror_SML.append(np.sqrt(RMSerror/ (n_Trials*N*M))/(max_obs-min_obs))

mean_y_pred_SML = np.mean(y_pred_SML, axis = 0)
std_y_pred_SML = np.std(y_pred_SML, axis = 0)
for i in range(N):
    plt.figure()
    img_path = experiment_path + "/experiment with {} inducing points/".format(num)
    img_path += "/img_user:{}.png".format(i)
    plt.plot(actual_price_test[0, :], observation_test[i, :], 'o', label= 'actual inventory')   
    plt.plot(actual_price_test[0, :], mean_y_pred_SML[i, :] + 2* std_y_pred_SML[i, :], 'x' ,label= 'upper confidence') 
    plt.plot(actual_price_test[0, :], mean_y_pred_SML[i, :] - 2* std_y_pred_SML[i, :], 'x' ,label= 'lower confidence')  
    plt.legend()
    plt.xlabel('$Price$', fontsize=13)
    plt.ylabel('$y = f(x,h)$', fontsize=13)
    plt.title(('Inventory of user:{}'. format(i+1)))
    plt.savefig(img_path)
    
RMS = {}
RMS['RMSerror'] = RMSerror_SML
with open(config_path, "w") as f:
    json.dump(RMS, f)