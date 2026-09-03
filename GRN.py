"""External Imports (Libraries and APIs)"""
import numpy as np


"""Local Imports"""
# All files import all from Global (contains all Global / Environmental variables)
import Global




def tanh_sigma(x: np.ndarray) -> np.ndarray:
    r = np.tanh(x)


    return r




def linear_sigma(x: np.ndarray) -> np.ndarray:


    return x



def develop(G: np.ndarray, B: np.ndarray, sigmoid = tanh_sigma, t1: float = 1.0, t2: float = 0.2, T: int = 10) -> np.ndarray:    
    P = np.asarray(G, dtype = float).copy()


    for _ in range(T):
        P += t1 * sigmoid(B @ P) - t2 * P


    return P




def fitness(P: np.ndarray, S: np.ndarray) -> float | np.ndarray:
    P = np.asarray(P, dtype = float)

    W = 1.0 + P @ S
    

    return W




def signed_fitness(P: np.ndarray, S: np.ndarray) -> float | np.ndarray:
    P = np.asarray(P, dtype = float).copy()

    P_sign = np.sign(P)

    W = 1.0 + P_sign @ S
    

    return W




def aligned_fitness(P: np.ndarray, S: np.ndarray) -> float | np.ndarray:
    P = np.asarray(P, dtype = float).copy()

    P = np.clip(P, a_min = -1, a_max = 1)

    W = 1.0 + np.absolute(P @ S)
    

    return W




def random_profiles(R: int, N: int = None) -> np.ndarray:
    if N is None: N = Global.N
    G = Global.RNG.uniform(low = -1.0, high = 1.0, size = (R, N))


    return G




def mutate_profile(G: np.ndarray, u1: float = 0.1, n_mut: int = 1) -> np.ndarray:
    G_mut = np.asarray(G, dtype = float).copy()
    N = G_mut.shape[-1]

    idx = Global.RNG.integers(low = 0, high = Global.N, size = n_mut)
    h = G_mut[idx] + Global.RNG.uniform(low = -u1, high = u1, size = n_mut)
    G_mut[idx] = np.clip(h, -1.0, 1.0)


    return G_mut




def mutate_interactions(B: np.ndarray, u2: float = 0.0067, prob_mut: float = 0.067, bin_mask: np.ndarray = None, n_mut: int = 1) -> np.ndarray:

    B_mut = np.asarray(B, dtype = float).copy()
    N = B_mut.shape[0]


    if Global.RNG.random() >= prob_mut:

        return B_mut


    if bin_mask is None:
        rows = Global.RNG.integers(low = 0, high = Global.N, size = n_mut)
        cols = Global.RNG.integers(low = 0, high = Global.N, size = n_mut)

    else:
        allowed = np.flatnonzero(bin_mask.ravel())


        if allowed.size <= 0:

            return B_mut
        

        flat = Global.RNG.choice(allowed, size = n_mut, replace = True)
        rows, cols = np.unravel_index(flat, (Global.N, Global.N))


    for i, j in zip(rows, cols):
        B_mut[i, j] += Global.RNG.uniform(low = -u2, high = u2)


    return B_mut




def hebbian_interactions(S: np.ndarray, lr: float = 1.0, bin_mask: np.ndarray = None, r: float = 1.0) -> np.ndarray:
    S = np.atleast_2d(np.asarray(S, dtype = float))


    # Check and apply mask
    if bin_mask is None:
        bin_mask = 1

    H = lr * (S.T @ S) * bin_mask


    # Scale Hebbian interactions
    r = np.abs(r)
    H = H / r


    return H



def sparse_topology(
        N: int = None,
        k: int = 10,
        self_interaction: bool = False,
        symmetric: bool = True,
        combine: str = "union",
    ) -> np.ndarray:

    if N is None:
        N = Global.N
 
    bin_mask = np.zeros((N, N), dtype=bool)
    all_genes = np.arange(N)
 
    for i in range(N):
        others = Global.RNG.choice(all_genes[all_genes != i], size=k, replace=False)
        bin_mask[i, others] = True
 
    if symmetric:
        if combine == "union":
            bin_mask = bin_mask | bin_mask.T
        elif combine == "intersection":
            bin_mask = bin_mask & bin_mask.T
        else:
            raise ValueError("combine must be 'union' or 'intersection'")
 
    # Set last so symmetrisation cannot clear it.
    np.fill_diagonal(bin_mask, bool(self_interaction))
 
    return bin_mask
 


 
def diag_mask(N: int = None) -> np.ndarray:
    
    if N is None:
        N = Global.N

    m = np.ones((N, N), dtype=bool)

    np.fill_diagonal(m, False)

    
    return m
 


 
def mask_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask).ravel())
 


 
def masked_matrix(mask: np.ndarray, values: np.ndarray = None) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    B = np.zeros(mask.shape, dtype=float) if values is None \
        else np.asarray(values, dtype=float).copy()
    return np.where(mask, B, 0.0)
 
 


def hamming_dist(a: np.ndarray, b: np.ndarray) -> int:
    D = int(np.sum(np.sign(np.asarray(a).ravel()) != np.sign(np.asarray(b).ravel())))


    return D




def sswm_evolve(
        B: np.ndarray = None,
        S: np.ndarray = None,
        N: int = None,
        n_generations: int = 200_000,
        sigmoid=tanh_sigma,
        t1_coef: float = 1.0,
        t2_coef: float = 0.2,
        dev_time_steps: int = 10,
        mutation_u1: float = 0.1,
        mutation_u2: float = 0.0067,
        prob_mutB: float = 0.067,
        n_mut_G: int = 1,
        n_mut_B: int = 1,
        mask: np.ndarray = None,
        switch_every: int = 2000,
        record_trajectories: bool = True,
        record_every: int = 1000,
        symmetric_B: bool = False,
    ) -> tuple:
 
    S = Global.TARGET if S is None else S
    TARGET_ = np.atleast_2d(np.asarray(S, dtype=float))
    M, dimN = TARGET_.shape[:2]
    N_ = int(N) if N is not None else dimN
 
    if mask is None:
        mask = diag_mask(N_)
    mask = np.asarray(mask).astype(bool)
 
    B = masked_matrix(mask) if B is None else masked_matrix(mask, B)
    B_flat = B.reshape(-1)                       # view, mutated in place
    G = np.zeros(N_)
 
    allowed = mask_indices(mask)
    n_allowed = allowed.size
 
    if symmetric_B and not np.array_equal(mask, mask.T):
        raise ValueError("symmetric_B=True requires a symmetric mask")
 
    # Precompute the transposed partner of each allowed entry once.
    if symmetric_B:
        rows, cols = np.unravel_index(allowed, (N_, N_))
        partner = cols * N_ + rows               # flat index of (j, i)
 
    def _develop(g):
        P = np.asarray(g, dtype=float).copy()
        for _ in range(dev_time_steps):
            P += t1_coef * sigmoid(B @ P) - t2_coef * P
        return P
 
    interaction_developments, recorded_gens = [], []
 
    ei = int(Global.RNG.integers(M))
    P = _develop(G)
    w = fitness(P, TARGET_[ei])
 
    for gen in range(n_generations):
 
        if M > 1 and gen > 0 and gen % switch_every == 0:
            ei = int(Global.RNG.integers(M))
            w = fitness(P, TARGET_[ei])
 
        G_mut = mutate_profile(G, u1=mutation_u1, n_mut=n_mut_G)
 
        # ---- mutate B in place, within the mask, remembering the delta ----
        undo = None
        if n_allowed and Global.RNG.random() < prob_mutB:
            t = Global.RNG.integers(0, n_allowed, size=n_mut_B)
            deltas = Global.RNG.uniform(-mutation_u2, mutation_u2, size=n_mut_B)
            idx = allowed[t]
            np.add.at(B_flat, idx, deltas)
            if symmetric_B:
                mirror = partner[t]
                off = mirror != idx              # skip any diagonal entry
                np.add.at(B_flat, mirror[off], deltas[off])
                undo = (idx, mirror[off], deltas, deltas[off])
            else:
                undo = (idx, None, deltas, None)
 
        P_mut = _develop(G_mut)
        w_mut = fitness(P_mut, TARGET_[ei])
 
        if w_mut >= w:
            G, P, w = G_mut, P_mut, w_mut
        elif undo is not None:
            idx, mirror, deltas, mdeltas = undo
            np.add.at(B_flat, idx, -deltas)
            if mirror is not None:
                np.add.at(B_flat, mirror, -mdeltas)
 
        # ---- subsampled recording of masked entries only ----
        if record_trajectories and (gen % record_every == 0):
            interaction_developments.append(B_flat[allowed].copy())
            recorded_gens.append(gen)
 
    if interaction_developments:
        interaction_trajectories = np.array(interaction_developments).T
        recorded_gens = np.array(recorded_gens)
    else:
        interaction_trajectories = np.empty((n_allowed, 0))
        recorded_gens = np.array([], dtype=int)
 
    return G, B, P, w, interaction_trajectories, recorded_gens