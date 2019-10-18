from meta_loop import *
import numpy as np
import argparse, os
import json
from main import MAIN

def run_main(main, training_envs, test_envs, **kwargs):

    if kwargs["model_name"] == "BASESVGP":

        all_iters = independent_loop(
            main=main,
            envs=training_envs,
            n_iters=kwargs["meta_train_iters"],
            **kwargs)
        envs_l=[]
        envs_l= np.unique(training_envs[:, 2])
        n_train = len(training_envs)
        training_iters = all_iters[:n_train]
        test_iters = all_iters[n_train:]

    else:

        training_iters= meta_loop(
            main=main,
            n_iters=kwargs["meta_train_iters"],
            envs=training_envs,
            train=True,
            **kwargs)

        test_iters = meta_loop(
            main=main,
            n_iters=kwargs["meta_test_iters"],
            envs=test_envs,
            train=False,
            **kwargs)
        
        
