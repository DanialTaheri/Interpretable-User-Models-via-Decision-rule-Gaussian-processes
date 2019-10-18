import tensorflow as tf
import numpy as np
import gpflow
import sys, argparse, os
from tqdm import tqdm
import data

from gpflow import settings
from gpflow import kernels
from likelihoods import *
from gpflow.params import Parameter
from gpflow.decors import params_as_tensors_for
from training import initialize_model
from func_utils import mu_std
from models import *

from gpflow.logdensities import gaussian
import pylab
#from dataset import get_seq_batch

class PILCO(object):

    def __init__(self, model, dataset, **kwargs):

        self.kwargs = kwargs
        self.rng = np.random.RandomState(kwargs["seed"])
        self.session = tf.Session()

        self.model = model
        self.dataset = dataset

        self._build_model_graph()
        self.data={}

        self.n_iters = 0
        self.n_active_tasks = kwargs["n_active_tasks"]
        self.mean = 0
        self.var = 0

    def _build_model_graph(self):

        self.model_objective = -self.model._build_likelihood()
        self.model_train_step, self.model_infer_step, self.model_optimizer =\
            initialize_model(self.model, self.model_objective,
            self.session, self.kwargs["learning_rate"])
        
##################################################################        


    def prepare_data(self):
        
        dataset=self.dataset
        num=dataset.shape[0]
        all_input_traj = []
        all_output_traj = []
        all_ids_traj = []
        n_episodes=0
        for eid in range(num):
            inputs, outputs, ids = get_inputs_outputs(dataset[eid, :])

            all_input_traj.append(inputs)
            all_output_traj.append(outputs)
            all_ids_traj.append(ids)
            n_episodes += 1

        all_inputs = np.vstack(all_input_traj)
        all_outputs = np.vstack(all_output_traj)
        all_ids_traj= np.vstack(all_ids_traj)
        
        inp_mu, inp_std = mu_std(all_inputs)
        out_mu, out_std = mu_std(all_outputs)
        self.mean = out_mu
        self.var = out_std
        norm = lambda inp, mu, std: (inp - mu) / std
        inp_norm = [norm(inp, inp_mu, inp_std) for inp in all_input_traj]
        out_norm = [norm(out, out_mu, out_std) for out in all_output_traj]

        #norm = lambda inp, mu, std: (inp - mu) / std
#         inp_norm = [inp for inp in all_input_traj]
#         out_norm = [out for out in all_output_traj]
        

        n_data = all_inputs.shape[0]
        
        self.data["n_data"] = n_data
        self.data["n_episodes"] = n_episodes
        self.data["inputs"] = inp_norm
        self.data["outputs"] = out_norm
        self.data["ids"] = all_ids_traj
        self.data["inp_mu"] = inp_mu
        self.data["inp_std"] = inp_std
        self.data["out_mu"] = out_mu
        self.data["out_std"] = out_std

        

    def get_seq_batch(self, seq, si, ei):

        inp_seq = self.data["inputs"]
        out_seq = self.data["outputs"]
        ids_seq = self.data["ids"]
        inp_seq = [inp_seq[i] for i in seq]
        out_seq = [out_seq[i] for i in seq]
        ids_seq = [ids_seq[i] for i in seq]
        #print("D,E:",inp_seq[0], inp_seq[0].shape)
        D = 3
        E = 1
        
        X_b = np.vstack(inp_seq[si:ei]).reshape(-1, D)
        Y_b = np.vstack(out_seq[si:ei]).reshape(-1, E)
        ids_b = np.vstack(ids_seq[si:ei]).reshape(-1)
        ids_unique = np.unique(ids_b)

        return X_b, Y_b, ids_b, ids_unique  
    
#############################################################################    
    def _set_inducing(self):
        n_data = self.data["n_data"]
        n_inducing = self.kwargs["n_inducing"]
        dim_in=self.kwargs["dim_in"]
        dim_h= self.kwargs["dim_h"]
        X=  np.vstack(self.data["inputs"])
        Z= self.model.feature.Z.read_value(session=self.session)
        diff = n_inducing - n_data
        if diff >= 0:
            Z[:n_data, :X.shape[1]] = X
        else:
            seq = np.arange(n_data)
            self.rng.shuffle(seq)
            Z[:, :X.shape[1]] = X[seq[:n_inducing]]
        self.model.feature.Z = Z

    def reset(self):

        self.session.close()
        self.session = tf.Session()
        self.model_train_step, self.model_infer_step, self.model_optimizer =\
            initialize_model(self.model, self.model_objective,
            self.session, self.kwargs["learning_rate"])
        self.dataset.__init__()

    def train_model(self):

        dataset = self.dataset
        self.prepare_data()
        
        kwargs = self.kwargs
        n_data = self.data["n_data"]

        n_inducing = kwargs["n_inducing"]
        n_episodes = self.data["n_episodes"]

        batch_size = kwargs["batch_size"]
        num_batches = max(int((n_episodes / batch_size)), 1)
        seq = np.arange(n_episodes)

        if (self.n_iters == 1) or (n_data <= n_inducing):
            self._set_inducing()

        for step in tqdm(range(kwargs["train_steps"])):
            all_obj = []
            self.rng.shuffle(seq)
            for b in range(int(num_batches)):
                
                si = b * batch_size
                ei = si + batch_size

                
                X_b, Y_b, ids_b, ids_unique = self.get_seq_batch(seq, si, ei)

                data_scale = n_data / X_b.shape[0]
                H_scale = self.n_active_tasks / ids_unique.shape[0]     
           
                feed_dict = {
                    self.model.X_mu_ph: X_b,
                    self.model.Y_ph: Y_b,
                    self.model.data_scale: data_scale
                }
                
                if self.model.name == "MLSVGP":
                    feed_dict[self.model.H_ids_ph] = ids_b
                    feed_dict[self.model.H_unique_ph] = ids_unique
                    feed_dict[self.model.H_scale] = H_scale            
   
                _, obj = self.session.run(
                    [self.model_train_step, self.model_objective],
                    feed_dict=feed_dict)

                all_obj.append(obj)
                    
                
            mobj = np.mean(all_obj)
        print(mobj)


    def infer_task_variable(self,env_id):

        dataset = self.dataset
        kwargs = self.kwargs
        n_data = dataset.shape["n_data"]

        norm = lambda inp, mu, std: (inp - mu) / std
        inp_norm = norm(inputs, dataset.data["inp_mu"], dataset.data["inp_std"])
        out_norm = norm(outputs, dataset.data["out_mu"], dataset.data["out_std"])
        batch_size = inputs.shape[0]
        ids = np.int32(batch_size * [env_id])
        data_scale = n_data / batch_size

        for step in range(kwargs["infer_steps"]):

            feed_dict = {
                self.model.X_mu_ph: inp_norm,
                self.model.Y_ph: out_norm,
                self.model.data_scale: data_scale,
                self.model.H_ids_ph: ids,
                self.model.H_unique_ph: 1,
                self.model.H_scale: self.n_active_tasks
            }

            _, obj = self.session.run(
                [self.model_infer_step, self.model_objective],
                feed_dict=feed_dict)

#             print("Step {}/{} :: {:.2f}".format(
#                 step+1, kwargs["infer_steps"], obj))


    def predict(self, xx, ids, use_var=True):
#         print(xx.shape)
        n_tasks = xx.shape[0]
        n_data = xx.shape[1]
        dim_in = xx.shape[2]
        
        all_input_traj = []
        
        for i in range(n_tasks):
            for eid in range(n_data):
                all_input_traj.append(xx[i, eid, :])
            
        all_inputs = np.vstack(all_input_traj)

        inp_mu, inp_std = mu_std(all_inputs)

        norm = lambda inp, mu, std: (inp - mu) / std
        inp_norm = [norm(inp, inp_mu, inp_std) for inp in all_input_traj]
        inp_norm = np.array(inp_norm)
        print(inp_norm.shape)

        if self.model.name == "MLSVGP":
            XX= inp_norm
            hmu,hvar = self.model.get_H_space(session=self.session)
            ymu_pred, yvar_pred = self.model.predict_y_ML(hmu, hvar, use_var=use_var) 


            fd = {
                self.model.X_mu_ph: XX,
                self.model.H_ids_ph: ids,
                self.model.num_steps: n_data}
            ymu, yvar = self.session.run([ymu_pred, yvar_pred], feed_dict=fd) #, hmu1, hvar1, hmu, hvar

            ymu = ymu.reshape(n_tasks, n_data, ymu.shape[1])
            yvar = yvar.reshape(n_tasks, n_data, yvar.shape[1])
            ymu = ymu*self.var + self.mean
            yvar = yvar*self.var + self.mean
            print("ML:",self.mean, self.var)
            
            return ymu, yvar, hmu, hvar
        elif self.model.name == "BASESVGP":
            XX= inp_norm
            ymu_pred, yvar_pred = self.model.predict_y_B( use_var=use_var) 


            fd = {
                self.model.X_mu_ph: XX,
                self.model.num_steps: n_data}
            ymu, yvar = self.session.run([ymu_pred, yvar_pred], feed_dict=fd) 
            ymu = ymu.reshape(n_tasks, n_data, ymu.shape[1])
            yvar = yvar.reshape(n_tasks, n_data, yvar.shape[1])
            ymu = ymu*self.var + self.mean
            yvar = yvar*self.var + self.mean
            print("B:", self.mean, self.var)        
        

        return ymu, yvar

   

    
    def plot_H_space(self, xmax=3, ymax=3):

        H_mu, H_var = self.model.get_H_space(session=self.session)
        H_err = 2*np.sqrt(H_var)
#         pylab.figure()
#         for h in range(self.n_active_tasks):
#             pylab.errorbar(
#                 H_mu[h, 0], H_mu[h, 1],
#                 xerr=H_err[h, 0], yerr=H_err[h, 1], fmt="o",
#                 label="Task {}".format(h))
        
        print("H_mu:", H_mu)
        print("H_var:", H_var)
        print("H_error:", H_err)
#         pylab.xlim(-xmax, xmax)
#         pylab.ylim(-ymax, ymax)
#         pylab.legend()

    def f_values(self):

        lik_noise, kern_var, kern_ls = self.model.get_model_param(session=self.session)
        
        print("lik_noise:", lik_noise)
        print("kern_var:", kern_var)
        print("kern_ls:",kern_ls)
        

def get_inputs_outputs(X):
    inputs = X[0:3]
    outputs = X[3:4]
    ids= X[-1]
    return inputs, outputs, ids       

