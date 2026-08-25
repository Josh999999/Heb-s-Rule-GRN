import numpy as np
import Global


def corrupt_progressively(P: np.ndarray, steps: int = 20) -> np.ndarray:
    P = np.asarray(P, dtype=float).copy()

    n = P.size
    order = Global.RNG.permutation(n)
    cuts = np.linspace(0, n, steps + 1).astype(int)

    P_mut = P.copy().ravel()
    images = [P_mut.copy().reshape(P.shape)]

    for s in range(steps):
        idx = order[cuts[s]:cuts[s + 1]]
        P_mut[idx] = Global.RNG.choice([-1.0, 1.0], size=idx.size)
        images.append(P_mut.copy().reshape(P.shape))

    return images


def shrinking_patch(P: np.ndarray, steps: int = 20) -> np.ndarray:
    P = np.asarray(P, dtype=float).copy()

    if P.ndim == 1:
        side = int(round(P.size ** 0.5))
        P = P.reshape(side, side)
    H, W = P.shape
    y, x = np.ogrid[:H, :W]

    images = []

    for s in range(steps + 1):
        frac = 1.0 - s / steps

        if frac <= 0.0:
            images.append(np.zeros((H, W)))
            continue

        radius = int(np.sqrt(frac * (H * W) / np.pi))
        radius = max(1, min(radius, min(H, W) // 2))

        cy = int(Global.RNG.integers(radius, H - radius + 1)) if H - radius + 1 > radius else H // 2
        cx = int(Global.RNG.integers(radius, W - radius + 1)) if W - radius + 1 > radius else W // 2

        inside = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

        P_mut = np.zeros((H, W))
        P_mut[inside] = P[inside]
        images.append(P_mut)

    return images


def interpolate(P_from: np.ndarray, P_to: np.ndarray, steps: int = 20) -> np.ndarray:
    P_from = np.asarray(P_from, dtype=float).copy()
    P_to = np.asarray(P_to, dtype=float).copy()

    n = P_from.size
    order = Global.RNG.permutation(n)
    cuts = np.linspace(0, n, steps + 1).astype(int)

    P_mut = P_from.copy().ravel()
    flat_to = P_to.ravel()
    images = [P_mut.copy().reshape(P_from.shape)]

    for s in range(steps):
        idx = order[cuts[s]:cuts[s + 1]]
        P_mut[idx] = flat_to[idx]
        images.append(P_mut.copy().reshape(P_from.shape))

    return images
