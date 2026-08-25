import numpy as np
import Global
import os

from GRN import sswm_evolve, hebbian_interactions, tanh_sigma, diag_mask
from plot_experiments import plot_interaction_trajectories, show_interaction_heatmap, plot_development_trajectories, show_phenotypes
from Global import N, S1, reseed




def experiment1(
        alpha: np.ndarray = None,
        n_generations: int = 200_000,     
        sigmoid = tanh_sigma,
        t1_coef: float = 1.0,
        t2_coef: float = 0.2,
        dev_time_steps: int = 10,
        mutation_u1: float = 0.1,
        mutation_u2: float = 0.0067,
        prob_mutB: float = 0.067,
        R: int = 30,                        
        K: int = 4,
        switch_every = 2000,
        targets: np.ndarray = None,  
    ) -> tuple[np.array]:



    
    """!----------- Run Experiment -----------!"""
    S_set = TARGET if targets is None else np.asarray(targets, dtype=float)
    folder = Global.FIGURES_OUTPUT or "Experiment1"
    os.makedirs(folder, exist_ok=True)


    # Create the interaction matrix through evolution with natural selection
    _, B, _, _, trajectories, accepted_gens = sswm_evolve(
        B = alpha,
        S = S_set,
        n_generations = n_generations,
        sigmoid = sigmoid, 
        t1_coef = t1_coef, 
        t2_coef = t2_coef, 
        dev_time_steps = dev_time_steps,
        mutation_u1 = mutation_u1, 
        mutation_u2 = mutation_u2, 
        prob_mutB = prob_mutB,
        record_trajectories = True, 
        switch_every = switch_every,
    )


    # Create the Hebbian Interaction matrix
    r = np.mean(np.abs(B))
    B_heb = hebbian_interactions(S = S_set, bin_mask = None, r = r)






    """!----------- Plot Results -----------!"""
    # (A) Regulatory interaction coefficients evolve into positive and negative classes.
    plot_interaction_trajectories(
        accepted_gens = accepted_gens, 
        interaction_trajectories = trajectories, 
        title = "Interaction coefficients over evolutionary time", 
        saveloc = f"{folder}/Figure1A.png"
    )




    # (B) The matrix of evolved regulatory interactions.
    show_interaction_heatmap(
        B = B, 
        title = "Evolved interaction matrix B", 
        saveloc = f"{folder}/Figure1B.png"
    )




    # (C) Interaction matrix derived from Hebb's rule.
    show_interaction_heatmap(
        B = B_heb, 
        title = "Hebbian interaction matrix", 
        saveloc = f"{folder}/Figure1C.png"
    )




    # (D) Gene expression levels over developmental time, from random G.
    plot_development_trajectories(
        B = B, 
        K = K, 
        dev_time_steps = dev_time_steps, 
        t1_coef = t1_coef, 
        t2_coef = t2_coef, 
        sigmoid = sigmoid, 
        title = "Development trajectories of individual genes", 
        saveloc = f"{folder}/Figure1D.png"
    )




    # (E) 30 independent adult phenotypes from random G -> target pattern or its complement.
    show_phenotypes(
        B = B, 
        R = R, 
        sigmoid = sigmoid, 
        t1_coef = t1_coef, 
        t2_coef = t2_coef, 
        dev_time_steps = dev_time_steps, 
        title=  "Adult phenotypes from random G", 
        saveloc = f"{folder}/Figure1E.png"
    )




if __name__ == "__main__":

    reseed()   # reproducible: reset the shared global RNG

    Global.N = 8
    N = Global.N
    TARGET = S1
    Global.FIGURES_OUTPUT = "Experiment1"

    # Initialise interaction matrix
    mask = diag_mask()
    B = np.zeros((N, N)) * mask


    # Experiment 1
    def experiment1_():
        experiment1(
            alpha = B.copy(),
            n_generations = 200_000,     
            sigmoid = tanh_sigma,
            t1_coef = 1.0,
            t2_coef = 0.2,
            dev_time_steps = 10,
            mutation_u1 = 0.1,
            mutation_u2 = 0.0067,
            prob_mutB = 0.067,
            R = 30,                       
            K = 4,                        
        ) 

    
    # Run the experiment
    experiment1_()