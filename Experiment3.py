import numpy as np
import Global
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GRN import hamming_dist, tanh_sigma, linear_sigma, random_profiles, sparse_topology, sswm_evolve, develop
from phenotype_noise import corrupt_progressively, shrinking_patch, interpolate
from plot_experiments import show_image_strip, plot_2d_projection
from Global import N, IMAGE_DARWIN_FLAT, TARGET, reseed, SIDE_3, IMAGE_DARWIN, IMAGE_HEB, IMAGE_DARWIN_FLAT, IMAGE_HEB_FLAT, FIGURES_OUTPUT




def experiment3(
        alpha: np.ndarray = None,
        mask: np.ndarray = None,
        sigmoid = tanh_sigma,               
        n_generations: int = 400_000,   
        switch_every: int = 2000,      
        speedup: int = 40,              
        dev_time_steps: int = 10,
        t1_coef: float = 1.0,
        t2_coef: float = 0.2,   
        mutation_u1: float = 0.1,
        mutation_u2: float = 0.0067,
        V: int = 20,      
        prob_mutB: float = 1.0
    ):

    u1 = mutation_u1 * speedup
    u2 = mutation_u2 * speedup

    folder = FIGURES_OUTPUT or "Experiment3"
    os.makedirs(folder, exist_ok=True)

    _, W, _, _, _, _ = sswm_evolve(
        B = alpha,
        S = TARGETS, 
        N = N, 
        n_generations = n_generations, 
        switch_every = switch_every,
        mutation_u1 = u1, 
        mutation_u2 = u2, 
        n_mut_G = 1, 
        n_mut_B = 1, 
        prob_mutB = prob_mutB,
        sigmoid = sigmoid, 
        t1_coef = t1_coef, 
        t2_coef = t2_coef, 
        dev_time_steps = dev_time_steps, 
        mask = mask
    )




    # --- (A) Display target as Image ----------
    fig, axes = plt.subplots(figsize=(5, 2.6))

    axes.imshow(IMAGE_DARWIN, cmap="gray", vmin=-1, vmax=1)
    axes.set_title("IMAGE_DARWIN_FLAT (Darwin stand-in)", fontsize=8)
    axes.axis("off")

    fig.tight_layout()
    fig.savefig(f"{folder}/Figure3A.png", dpi=150)
    plt.close(fig)




    # --- (B) Display target as Image ----------
    fig, axes = plt.subplots(figsize=(5, 2.6))

    axes.imshow(IMAGE_HEB, cmap="gray", vmin=-1, vmax=1)
    axes.set_title("IMAGE_HEB_FLAT (Hebb stand-in)", fontsize=8)
    axes.axis("off")

    fig.tight_layout()
    fig.savefig(f"{folder}/Figure3B.png", dpi=150)
    plt.close(fig)




    # --- (C) increasing random corruption of G, starting from IMAGE_DARWIN_FLAT ----------    
    def grow(G_img, weights = W, sigma = tanh_sigma):

        return develop(G = G_img.ravel(), B = weights, sigmoid = sigma, t1 = t1_coef, t2 = t2_coef, T = dev_time_steps).reshape(SIDE_3, SIDE_3)


    Gs = corrupt_progressively(P = IMAGE_DARWIN, steps = V)
    Ps = [grow(g) for g in Gs]
    show_image_strip(G_images = Gs, P_images = Ps, saveloc = "Figure3C_imgstrip.png", title = "Exp 3(C): increasing noise/mutations on G (top) and resulting adult phenotypes (bottom)")

    match = [np.mean(np.sign(p.ravel()) == np.sign(IMAGE_DARWIN_FLAT)) for p in Ps]
    levels = np.linspace(0, 100, V + 1)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(levels, match, "o-")
    ax.set_xlabel("% of G replaced with random alleles")
    ax.set_ylabel("fraction of adult phenotype matching IMAGE_DARWIN_FLAT")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness of recall to corruption of G")

    fig.tight_layout()
    fig.savefig(f"{folder}/Figure3C.png", dpi=150)
    plt.close(fig)




    # --- (D) fully random G ----------------------------------------------
    Gs = random_profiles(V)
    Ps = [grow(g) for g in Gs]
    show_image_strip(G_images = Gs, P_images = Ps, saveloc = "Figure3D_imgstrip.png", title = "Exp 3(D): random G (top) and resulting adult phenotypes (bottom)")




    # --- (E) recall from a shrinking partial patch of IMAGE_HEB_FLAT ------------------
    Gs = shrinking_patch(IMAGE_HEB, steps=V)
    Ps = [grow(g) for g in Gs]
    show_image_strip(G_images = Gs, P_images = Ps, saveloc = "Figure3E_imgstrip.png", title = "Exp 3(E): partial G resembling IMAGE_HEB_FLAT (top) and resulting adult phenotypes (bottom)")

    match = [np.mean(np.sign(p.ravel()) == np.sign(IMAGE_HEB_FLAT)) for p in Ps]
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(np.linspace(100, 0, V + 1), match, "o-")
    ax.set_xlabel("% of image area retained in the patch")
    ax.set_ylabel("fraction of adult phenotype matching IMAGE_HEB_FLAT")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall from a partial stimulus")

    fig.tight_layout()
    fig.savefig(f"{folder}/Figure3E.png", dpi=150)
    plt.close(fig)



    # --- (F) G interpolated IMAGE_DARWIN_FLAT -> IMAGE_HEB_FLAT, NONLINEAR development ---------------    
    levels = np.linspace(0, 100, V + 1)

    Gs = interpolate(IMAGE_DARWIN, IMAGE_HEB, steps=V)
    Ps_nl = [grow(g, sigma = tanh_sigma) for g in Gs]
    show_image_strip(G_images = Gs, P_images = Ps_nl, saveloc = "Figure3F_imgstrip.png", title = "Exp 3(F): G varies systematically from IMAGE_DARWIN_FLAT to IMAGE_HEB_FLAT (top); adult phenotypes (bottom)")

    d1_nl = [hamming_dist(p, IMAGE_DARWIN_FLAT) for p in Ps_nl]
    d2_nl = [hamming_dist(p, IMAGE_HEB_FLAT) for p in Ps_nl]

    fig, axes = plt.subplots(figsize=(11, 4), sharey=True)

    axes.plot(levels, d1_nl, "o-", label = "Hamming distance to IMAGE_DARWIN_FLAT")
    axes.plot(levels, d2_nl, "s-", label = "Hamming distance to IMAGE_HEB_FLAT")
    axes.set_xlabel("% of G taken from IMAGE_HEB_FLAT rather than IMAGE_DARWIN_FLAT")

    axes.set_ylabel("Hamming distance of adult phenotype")

    fig.suptitle("Abrupt ('categorical') switching under nonlinear development")
    fig.tight_layout()
    fig.savefig(f"{folder}/Figure3F.png", dpi=150)
    plt.close(fig)




    # --- (G) same G sequence, LINEAR development, same evolved B ---------
    Ps_lin = [grow(g, sigma=linear_sigma) for g in Gs]
    show_image_strip(G_images = Gs, P_images = Ps_lin, saveloc = "Figure3G_imgstrip.png", title = "Exp 3(G): as (F) but with a LINEAR developmental process")

    d1_lin = [hamming_dist(p, IMAGE_DARWIN_FLAT) for p in Ps_lin]
    d2_lin = [hamming_dist(p, IMAGE_HEB_FLAT) for p in Ps_lin]

    fig, axes = plt.subplots(figsize=(11, 4), sharey=True)

    axes.plot(levels, d1_lin, "o-", label="Hamming distance to IMAGE_DARWIN_FLAT")
    axes.plot(levels, d2_lin, "s-", label="Hamming distance to IMAGE_HEB_FLAT")
    axes.set_xlabel("% of G taken from IMAGE_HEB_FLAT rather than IMAGE_DARWIN_FLAT")
    axes.legend()

    axes.set_ylabel("Hamming distance of adult phenotype")

    fig.suptitle("Abrupt ('categorical') switching under linear development")
    fig.tight_layout()
    fig.savefig(f"{folder}/Figure3FG.png", dpi=150)
    plt.close(fig)




    # --- (H) 2D projection: TARGETS / evolved (nonlinear) / linear mapping -
    Gs = random_profiles(100)
    P_nl = [grow(g, sigma=tanh_sigma) for g in Gs]
    P_lin = [grow(g, sigma=linear_sigma) for g in Gs]

    groups = [
        ("target phenotypes",
         [hamming_dist(t, IMAGE_DARWIN_FLAT) for t in [IMAGE_DARWIN_FLAT, IMAGE_HEB_FLAT]],
         [hamming_dist(t, IMAGE_HEB_FLAT) for t in [IMAGE_DARWIN_FLAT, IMAGE_HEB_FLAT]]),
        ("evolved phenotypes",
         [hamming_dist(p, IMAGE_DARWIN_FLAT) for p in P_nl],
         [hamming_dist(p, IMAGE_HEB_FLAT) for p in P_nl]),
        ("linear mapping",
         [hamming_dist(p, IMAGE_DARWIN_FLAT) for p in P_lin],
         [hamming_dist(p, IMAGE_HEB_FLAT) for p in P_lin]),
    ]
    plot_2d_projection(groups = groups, saveloc = f"{folder}/Figure3H.png", title = "Phenotype distribution is bimodal under nonlinear development")








if __name__ == "__main__":

    reseed()   # reproducible: reset the shared global RNG

    Global.N = SIDE_3 * SIDE_3
    N = Global.N
    TARGETS = [IMAGE_DARWIN_FLAT, IMAGE_HEB_FLAT] # Needs to be put in with the Images (Darwin and Heb)
    FIGURES_OUTPUT = "Experiment3"

    # Initialise interaction matrix
    mask = sparse_topology(k = 10, self_interaction = False)
    B = np.zeros((N, N)) * mask


    # Experiment 1
    def experiment3_():
        experiment3(
            alpha = B.copy(),
            mask = mask,
            sigmoid = tanh_sigma,       
            n_generations = 400_000,   
            switch_every = 2000,      
            speedup = 40,              
            dev_time_steps = 10,
            t1_coef = 1.0,
            t2_coef = 0.2,     
            mutation_u1 = 0.1,
            mutation_u2 = 0.0067,
            V = 20, 
            prob_mutB = 1.0       
        )

    # Run the experiment
    experiment3_()