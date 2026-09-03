import numpy as np
import Global
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GRN import sswm_evolve, develop, hamming_dist, tanh_sigma, linear_sigma, random_profiles, sparse_topology
from plot_experiments import plot_2d_projection
from Global import SIDE_4, SELECTED_PATTERNS, S1, TARGET_PATTERNS, reseed, FOUR_LOOP_PATTERN, ZERO_LOOP_PATTERN




def experiment4(
        alpha: np.ndarray = None,
        mask: np.ndarray = None,
        n_generations: int = 500_000,
        switch_every: int = 2000,    
        sigmoid = tanh_sigma, 
        dev_time_steps: int = 10,
        t1_coef: float = 1.0,
        t2_coef: float = 0.2,
        speedup: int = 20,          
        mutation_u1: float = 0.1,
        mutation_u2: float = 0.0067,
        R: int = 100,
        n_mut_G: int = 1, 
        n_mut_B: int = 1,
        prob_mutB: float = 1.0
    ):

    # Set the Targets for the experiment
    global TARGETS
    TARGETS = TARGET_PATTERNS

    folder = FIGURES_OUTPUT or "Experiment4"
    os.makedirs(folder, exist_ok=True)

    u1 = mutation_u1 * speedup
    u2 = mutation_u2 * speedup

    _, B, _, _, _, _ = sswm_evolve(
        B = alpha,
        S = TARGETS,
        N = Global.N,
        n_generations = n_generations,
        sigmoid = sigmoid, 
        t1_coef = t1_coef, 
        t2_coef = t2_coef, 
        dev_time_steps = dev_time_steps,
        mutation_u1 = u1, 
        mutation_u2 = u2,
        prob_mutB = prob_mutB, 
        n_mut_G = n_mut_G, 
        n_mut_B = n_mut_B,
        mask = mask,
        switch_every = switch_every,
        record_trajectories = False, 
    )




    # --- (A) Display the Images ---------------    
    fig, axes = plt.subplots(1, 8, figsize=(12, 1.9))

    for ax, pattern in zip(axes, SELECTED_PATTERNS):
        ax.imshow(pattern, cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")

    fig.suptitle("Eight target phenotypes (modularly varying environment)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{folder}/Figure4A.png", dpi=150)
    plt.close(fig)




    # --- (B) 100 adult phenotypes from random G ---------------    
    def grow(G_vec, sigma=tanh_sigma):
        
        return develop(G_vec, B, sigmoid = sigma, t1 = t1_coef, t2 = t2_coef, T = dev_time_steps)
    
    Gs = random_profiles(R)
    P_nl = [grow(G_vec = g, sigma = tanh_sigma) for g in Gs]

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))

    for ax, p in zip(axes.ravel(), P_nl):
        ax.imshow(np.sign(p).reshape(SIDE_4, SIDE_4), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")

    fig.suptitle("100 adult phenotypes produced by the evolved network from random G")
    fig.tight_layout()
    fig.savefig(f"{folder}/Figure4B.png", dpi=150)
    plt.close(fig)




    # --- (C) 10 adult phenotypes under LINEAR development -----------------
    P_lin = [grow(g, linear_sigma) for g in Gs]

    fig, axes = plt.subplots(1, 10, figsize=(12, 1.6))

    for ax, p in zip(axes, P_lin[:10]):
        ax.imshow(np.sign(p).reshape(SIDE_4, SIDE_4), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")

    fig.suptitle("Adult phenotypes produced by a LINEAR developmental process", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{folder}/Figure4C.png", dpi=150)
    plt.close(fig)




    # --- (D) 2D projection ------------------------------------------------
    def proj(vectors):
        return ([hamming_dist(v, FOUR_LOOP_PATTERN) for v in vectors], [hamming_dist(v, ZERO_LOOP_PATTERN) for v in vectors])

    d1_t, d2_t = proj(list(TARGETS))
    d1_e, d2_e = proj(P_nl)
    d1_l, d2_l = proj(P_lin)

    plot_2d_projection(
        groups = [("target phenotypes", d1_t, d2_t), ("evolved phenotypes", d1_e, d2_e), ("linear mapping", d1_l, d2_l)], 
        saveloc = f"{folder}/Figure4D.png", 
        title = "D1 = Hamming distance from the 4-loop phenotype, D2 = Hamming distance from the 0-loop phenotype"
    )




if __name__ == "__main__":

    reseed()   # reproducible: reset the shared global RNG

    Global.N = SIDE_4 * SIDE_4
    N = Global.N
    TARGET = S1
    FIGURES_OUTPUT = "Experiment4"

    # Initialise interaction matrix
    mask = sparse_topology(k = 10, self_interaction = False)
    B = np.zeros((N, N)) * mask


    # Experiment 1
    def experiment4_():
        experiment4(
            alpha = B.copy(),
            mask = mask,
            n_generations = 500_000,          
            sigmoid = tanh_sigma, 
            switch_every = 2000,    
            dev_time_steps = 10,
            t1_coef = 1.0,
            t2_coef = 0.2,
            speedup = 20,       
            mutation_u1 = 0.1,
            mutation_u2 = 0.0067,
            R = 100,
            n_mut_G = 1, 
            n_mut_B = 1,
            prob_mutB = 1.0,
        )
            
    # Run the experiment
    experiment4_()