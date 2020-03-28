

####Nonparametric meta learning
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

# Data is deleted due to the confidentiality

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


# dataset_train= np.hstack((prior_observation_train, User_capacity_train, P_train , actual_observation_train , actual_ids_train))

# dataset_test= np.hstack((prior_observation_test, User_capacity_test, P_test , actual_observation_test , actual_ids_test))



def create_model(**kwargs):
    if kwargs["model_name"] == "MLSVGP": 
        #kernel.clear() 
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


num = 10

RMSerror= 0
kwargs= {"n_active_tasks": 4, "learning_rate": 0.1, "n_inducing" : num,  "batch_size": 10,
        "train_steps":400, "infer_steps": 100, "dim_in": 3, "dim_out":1, "dim_h": 1, "n_env":4, "meta_test_iters":0,
        "model_name": "MLSVGP","seed":1, "meta_train_iters": 20, "n_train_tasks":4}



model_NonPML = create_model(**kwargs)
Inference_NonPML = MAIN(model_NonPML, dataset_train, **kwargs) 
run_main(Inference_NonPML, dataset_train, dataset_train, **kwargs) 

N = observation_test.shape[0]
M = observation_test.shape[1]
xx_test= np.zeros((N,M,3))
experiment_path = "Proportional inventory-experiments-Non_parametric learning/70/"
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
if not os.path.exists(experiment_path + "/experiment with {} inducing points/".format(num)):
    os.makedirs(experiment_path + "/experiment with {} inducing points/".format(num))
config_path = experiment_path + "/experiment with {} inducing points/ config.json".format(num)
with open(config_path, "w") as f:
    json.dump(kwargs, f)


for j in range(N):
    for dim in range(M):
        xx_test[j,dim,0] = observation_test[j, dim]
        xx_test[j,dim,1] = capacity_test[j, dim]
        xx_test[j,dim,2] = actual_price_test[j, dim]

def plot(xx, ids):
    ymu, yvar, hmu, hvar = Inference_NonPML.predict(xx, ids, True)
    img_path = experiment_path + "/experiment with {} inducing points/".format(num)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in range(len(np.unique(ids))):
        img_path = experiment_path + "/experiment with {} inducing points/".format(num)
        img_path += "/img_user:{}.png".format(i)
        plt.figure()
        plt.plot(actual_price_test[0, :], observation_test[i, :].reshape(-1,1), 'kx', mew=0.5)
        plt.plot(actual_price_test[0, :], ymu[i,:,0] - 2*np.sqrt(yvar[i,:,0]), 'b*', alpha=0.3)
        plt.plot(actual_price_test[0, :], ymu[i,:,0] + 2*np.sqrt(yvar[i,:,0]), 'r*', alpha=0.3)    
        plt.xlabel('$Price$', fontsize=13)
        plt.ylabel('$y = f(x,h)$', fontsize=13)
        plt.title(('Non-parametric meta learning\n' 
                   'User:{}'. format(i+1)))
        plt.savefig(img_path)
    return ymu, yvar


y_mu, y_var = plot(xx_test, actual_ids_test.reshape(-1))

config_path = experiment_path + "/experiment with {} inducing points/RMSerror.json".format(num)
n_Trials = 10000
y_pred_NonPML = np.zeros((n_Trials, N, M))
for j in range(N):
    for i in range(M):
        for k in range(n_Trials):
            y_pred_NonPML[k, j, i] = np.random.normal(y_mu[j, i, 0], np.sqrt(y_var[j, i, 0]), 1)
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
    RMSerror += sum(sum((observation_test[:, :] - y_pred_NonPML[k, :, :])**2))
RMSerror_NonPML = []
RMSerror_NonPML.append(np.sqrt(RMSerror/ (n_Trials*N*M))/(max_obs -min_obs))
RMS_NonPML = {}
RMS_NonPML['RMSerror'] = RMSerror_NonPML
with open(config_path, "w") as f:
    json.dump(RMSerror_NonPML, f)

