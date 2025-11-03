# models/compression.py
import copy
import os
from typing import Callable, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.utils.extmath import randomized_svd
except Exception:
    randomized_svd = None


def svd_truncate_matrix(mat: np.ndarray, k: int) -> np.ndarray:
    """
    Full SVD truncation: compute U, S, Vt and return the rank-k approximation.
    mat: 2D numpy array shape (m, n)
    k: desired rank
    """
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    return (U_k * S_k) @ Vt_k


def randomized_svd_truncate_matrix(mat: np.ndarray,
                                   k: int,
                                   n_oversamples: int = 10,
                                   n_iter: int = 2,
                                   random_state: Optional[int] = None) -> np.ndarray:
    """
    Randomized SVD truncation (approximate) using sklearn.randomized_svd if available.
    Falls back to full SVD if sklearn is not present.
    """
    if randomized_svd is None:
        return svd_truncate_matrix(mat, k)
    U, S, Vt = randomized_svd(mat, n_components=k, n_oversamples=n_oversamples,
                              n_iter=n_iter, random_state=random_state)
    return (U * S) @ Vt



def adaptive_rank_from_entropy(singular_values: np.ndarray, tau: float) -> int:
    """
    Choose smallest k s.t. cumulative entropy >= tau * total_entropy.

    singular_values: 1D array of singular values (non-negative, typically descending)
    tau: threshold in (0,1), e.g., 0.90
    """
    s = np.array(singular_values, dtype=np.float64)
    total = s.sum()
    if total == 0:
        return 1
    p = s / total
    # avoid log(0)
    p_safe = np.where(p <= 0, 1e-12, p)
    ent = -p_safe * np.log(p_safe)
    cumsum_ent = np.cumsum(ent)
    total_ent = cumsum_ent[-1] if cumsum_ent.size > 0 else 0.0
    if total_ent == 0:
        return 1
    threshold = tau * total_ent
    k = int(np.searchsorted(cumsum_ent, threshold, side="left") + 1)
    return max(1, min(k, len(s)))



def truncate_conv2d_weight(weight: torch.Tensor,
                           method: str,
                           k: int,
                           n_oversamples: int = 10,
                           n_iter: int = 2,
                           random_state: Optional[int] = None) -> torch.Tensor:
    """
    Truncate a Conv2d weight tensor using SVD/randomized SVD.

    weight: torch.Tensor with shape (out_channels, in_channels, kh, kw)
    method: 'svd' or 'randomized'
    k: rank to keep
    Returns a torch.Tensor with same shape and dtype as input.
    """
    if weight.ndim != 4:
        raise ValueError("Expected a Conv2d weight tensor of shape (out_c, in_c, kh, kw)")

    out_c, in_c, kh, kw = weight.shape
    # reshape into 2D matrix: (out_c, in_c * kh * kw)
    mat = weight.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)

    # clamp k
    max_rank = min(mat.shape[0], mat.shape[1])
    k_use = max(1, min(int(k), max_rank))

    if method == "svd":
        approx = svd_truncate_matrix(mat, k_use)
    elif method == "randomized":
        approx = randomized_svd_truncate_matrix(mat, k_use, n_oversamples=n_oversamples,
                                                n_iter=n_iter, random_state=random_state)
    else:
        raise ValueError(f"Unknown method '{method}' (use 'svd' or 'randomized')")

    approx_tensor = torch.from_numpy(approx.reshape(out_c, in_c, kh, kw)).to(weight.device).type_as(weight)
    return approx_tensor



def default_layer_selector(module: nn.Module) -> bool:
    """
    Default: select all Conv2d layers.
    Customize this if you want to compress only encoder or certain named layers.
    """
    return isinstance(module, nn.Conv2d)



def compress_model_svd(model: nn.Module,
                       rank: Optional[int] = None,
                       energy: Optional[float] = None,
                       use_randomized: bool = False,
                       random_state: Optional[int] = None,
                       layer_selector: Optional[Callable[[nn.Module], bool]] = None,
                       svd_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Compress a model by applying SVD truncation to selected Conv2d layers.

    Provide either `rank` (fixed int) or `energy` (fraction 0..1 of singular energy to keep).
    - use_randomized: if True, use randomized SVD for reconstruction (faster for large mats if sklearn available)
    - random_state: seed passed to randomized SVD
    - layer_selector: function(module) -> bool to choose which layers to compress (default: Conv2d)
    - svd_kwargs: extra kwargs forwarded to truncate_conv2d_weight (n_oversamples, n_iter, ...)
    """
    if layer_selector is None:
        layer_selector = default_layer_selector
    if svd_kwargs is None:
        svd_kwargs = {}

    new_model = copy.deepcopy(model)

    for name, module in new_model.named_modules():
        if not layer_selector(module):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        w = module.weight.data  # tensor
        out_c, in_c, kh, kw = w.shape
        mat = w.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)

        # compute singular values (try full SVD; fallback to randomized for stability/perf)
        try:
            s = np.linalg.svd(mat, compute_uv=False)
        except Exception:
            if randomized_svd is not None:
                approx_rank = min(mat.shape[0], mat.shape[1], 128)
                U, S, Vt = randomized_svd(mat, n_components=approx_rank, n_oversamples=10,
                                          n_iter=2, random_state=random_state)
                s = S
            else:
                s = np.linalg.svd(mat, compute_uv=False)

        # determine k
        if energy is not None:
            energy_cumsum = np.cumsum(s ** 2) / np.sum(s ** 2)
            k = int(np.searchsorted(energy_cumsum, energy) + 1)
        elif rank is not None:
            k = int(rank)
        else:
            raise ValueError("Either 'rank' or 'energy' must be provided to compress_model_svd")

        method = "randomized" if use_randomized else "svd"
        new_w = truncate_conv2d_weight(w, method=method, k=k, random_state=random_state, **svd_kwargs)
        module.weight.data.copy_(new_w)
        # biases (if present) are left unchanged

    return new_model


def compress_model_arsvd(model: nn.Module,
                         tau: float = 0.95,
                         recon_method: str = "randomized",
                         n_oversamples: int = 10,
                         n_iter: int = 2,
                         random_state: Optional[int] = None,
                         layer_selector: Optional[Callable[[nn.Module], bool]] = None) -> nn.Module:
    """
    Adaptive-Rank SVD: for each selected layer compute singular values, choose k by entropy threshold tau,
    and reconstruct the weight with either full SVD or randomized reconstruction.

    recon_method: 'svd' or 'randomized'
    """
    if layer_selector is None:
        layer_selector = default_layer_selector

    new_model = copy.deepcopy(model)

    for name, module in new_model.named_modules():
        if not layer_selector(module):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        w = module.weight.data
        out_c, in_c, kh, kw = w.shape
        mat = w.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)

        # get singular values (full or approximate)
        try:
            s = np.linalg.svd(mat, compute_uv=False)
        except Exception:
            if randomized_svd is not None:
                approx_rank = min(mat.shape[0], mat.shape[1], 128)
                U, S, Vt = randomized_svd(mat, n_components=approx_rank, n_oversamples=n_oversamples,
                                          n_iter=max(1, n_iter), random_state=random_state)
                s = S
            else:
                s = np.linalg.svd(mat, compute_uv=False)

        k = adaptive_rank_from_entropy(s, tau)
        new_w = truncate_conv2d_weight(w, method=recon_method, k=k,
                                       n_oversamples=n_oversamples, n_iter=n_iter, random_state=random_state)
        module.weight.data.copy_(new_w)

    return new_model


# -----------------------
# Small utilities
# -----------------------
def model_size_bytes(model: nn.Module) -> int:
    """
    Return size in bytes of model.state_dict() when serialized. Useful to compute compression ratio.
    """
    tmp = "tmp_model_for_size.pth"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return size
