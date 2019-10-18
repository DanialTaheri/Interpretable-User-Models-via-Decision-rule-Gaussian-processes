# Imports
#%matplotlib notebook

import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns
from func_utils import mu_std

sns.set_style('darkgrid')
np.random.seed(42)
#

# Define the exponentiated quadratic 
def exponentiated_quadratic(xa, xb, sigma):
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return sigma**2 * np.exp(sq_norm)



# class MultiEnvData:

#     def __init__(self):

#         self.trajectories = {}
#         self.data = {}
#         self.num_envs = 0


#     def get_inputs_outputs(self, dataset):
#         inputs = np.hstack([states_transf[:-1]])
#         outputs = states[1:] - states[:-1]
#         return inputs, outputs

#     def prepare_data(self):

#         all_input_traj = []
#         all_output_traj = []
#         all_ids_traj = []
#         n_episodes = 0
#         for eid in self.trajectories:
#             episodes = self.trajectories[eid]
#             for ep, ep_data in enumerate(episodes):
#                 states = ep_data["states"]
#                 inputs, outputs = self.get_inputs_outputs(states)
#                 ids = np.int32(inputs.shape[0] * [eid])

#                 all_input_traj.append(inputs)
#                 all_output_traj.append(outputs)
#                 all_ids_traj.append(ids.reshape(-1, 1))
#                 n_episodes += 1

#         all_inputs = np.vstack(all_input_traj)
#         all_outputs = np.vstack(all_output_traj)

#         inp_mu, inp_std = mu_std(all_inputs)
#         out_mu, out_std = mu_std(all_outputs)

#         norm = lambda inp, mu, std: (inp - mu) / std
#         inp_norm = [norm(inp, inp_mu, inp_std) for inp in all_input_traj]
#         out_norm = [norm(out, out_mu, out_std) for out in all_output_traj]

#         n_data = all_inputs.shape[0]

#         self.data["n_data"] = n_data
#         self.data["n_episodes"] = n_episodes
#         self.data["inputs"] = inp_norm
#         self.data["outputs"] = out_norm
#         self.data["ids"] = all_ids_traj
#         self.data["inp_mu"] = inp_mu
#         self.data["inp_std"] = inp_std
#         self.data["out_mu"] = out_mu
#         self.data["out_std"] = out_std

#     def get_seq_batch(self, seq, si, ei):

#         inp_seq = self.data["inputs"]
#         out_seq = self.data["outputs"]
#         ids_seq = self.data["ids"]
#         inp_seq = [inp_seq[i] for i in seq]
#         out_seq = [out_seq[i] for i in seq]
#         ids_seq = [ids_seq[i] for i in seq]
#         #D = inp_seq[0].shape[1]
#         #E = out_seq[0].shape[1]
#         D=1
#         E=1

#         X_b = np.vstack(inp_seq[si:ei]).reshape(-1, D)
#         Y_b = np.vstack(out_seq[si:ei]).reshape(-1, E)
#         ids_b = np.vstack(ids_seq[si:ei]).reshape(-1)
#         ids_unique = np.unique(ids_b)

#         return X_b, Y_b, ids_b, ids_unique

#     def state_transform(self, states):
#         return states





