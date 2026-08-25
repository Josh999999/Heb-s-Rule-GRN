import os
import numpy as np
import Global

from Experiment1 import experiment1
from GRN import tanh_sigma, diag_mask
from Global import N, S1, TARGET, TARGETS, reseed



if __name__ == "__main__":

    reseed()   # reproducible: reset the shared global RNG

    Global.N = 8
    N = Global.N
    TARGET = S1
    Global.FIGURES_OUTPUT = "Experiment2"
    
    # Initialise interaction matrix
    mask = diag_mask()
    B = np.zeros((N, N)) * mask


    # Experiment 1
    def experiment2_():
        experiment1(
            alpha = B.copy(),
            n_generations = 800_000,  
            sigmoid = tanh_sigma,
            t1_coef = 1.0,
            t2_coef = 0.2,
            dev_time_steps = 10,
            mutation_u1 = 0.1,
            mutation_u2 = 0.0067,
            prob_mutB = 0.067,
            switch_every = 2000,
            R = 30,
            K = 4,
            targets = TARGETS,   # varying environment: S1 and S2
        ) 

    
    # Run the experiment
    experiment2_()