import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from GRN import random_profiles, develop, tanh_sigma
from Global import N




def plot_interaction_trajectories(accepted_gens: np.array, interaction_trajectories: np.array, saveloc: str, title = None):
    fig, ax = plt.subplots(figsize=(6, 4))

    gens = np.asarray(accepted_gens) / 1e5

    for trajectory in interaction_trajectories:
        ax.plot(gens, trajectory, lw=0.8)

    ax.set_title(title)
    ax.set_xlabel(r"Generations ($\times 10^5$)")
    ax.set_ylabel("Regulation coefficient")

    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)



def show_interaction_heatmap(B: np.array, saveloc: str, title = None):
    fig, ax = plt.subplots(figsize=(5, 4))

    vmax = np.max(np.abs(B)) or 1.0
    im = ax.imshow(B, cmap="bone", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Gene i")
    ax.set_ylabel("Gene j")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)



def plot_development_trajectories(
        B: np.array, K: int, saveloc: str, dev_time_steps: int = 10, t1_coef: float = 1.0, t2_coef: float = 0.2, sigmoid = tanh_sigma, title = None
    ):
    trajectories = []
    initial_profiles = random_profiles(K)

    for P in initial_profiles:
        history = [P]


        for _ in range(dev_time_steps):
            P = develop(P, B, sigmoid = sigmoid, t1 = t1_coef, t2 = t2_coef, T = 1)
            history.append(P)


        trajectories.append(np.array(history).T) # Transpose so rows track genes
    

    fig, axes = plt.subplots(ncols = K, figsize=(3 * K, 3.5), sharey=True)

    for k in range(K):
        for gene in trajectories[k]:
            axes[k].plot(gene, lw=1)
        axes[k].set_xlabel("Developmental time steps")
        axes[k].set_ylabel("Gene expression levels")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)



def show_phenotypes(
        B: np.array, 
        R: int, 
        saveloc: str, 
        dev_time_steps: int = 10, 
        t1_coef: float = 1.0, 
        t2_coef: float = 0.2, 
        sigmoid = tanh_sigma,
        title = None
    ):
    phenotypes = []
    initial_profiles = random_profiles(R)

    for G in initial_profiles:
        P = develop(G, B, sigmoid = sigmoid, t1 = t1_coef, t2 = t2_coef, T = dev_time_steps)
        phenotypes.append(P)


    fig, ax = plt.subplots(figsize=(5, 5))

    vmax = np.max(np.abs(phenotypes)) or 1.0
    im = ax.imshow(phenotypes, cmap="bone", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Genes")
    ax.set_ylabel("Phenotype samples")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)




def show_image_strip(G_images: np.array, P_images: np.array, saveloc: str, title = None):
    K = len(G_images)
    def _sq(a):
        a = np.asarray(a)
        if a.ndim == 1:
            s = int(round(len(a) ** 0.5)); a = a.reshape(s, s)
        return a
    fig, axes = plt.subplots(2, K, figsize=(0.7 * K, 2.0))

    for k in range(K):
        axes[0][k].imshow(np.sign(_sq(G_images[k])), cmap="gray", vmin=-1, vmax=1)
        axes[1][k].imshow(np.sign(_sq(P_images[k])), cmap="gray", vmin=-1, vmax=1)
        axes[0][k].axis("off")
        axes[1][k].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)




def plot_2d_projection(groups: np.array, saveloc: str, title = None):
    fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 4))

    for ax, (group_title, D1, D2) in zip(axes, groups):
        ax.scatter(D2, D1, facecolors="none", edgecolors="C0", alpha=0.8)
        ax.set_title(group_title)
        ax.set_xlabel("D2")
        ax.set_ylabel("D1")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(saveloc, dpi=150)
    plt.close(fig)
