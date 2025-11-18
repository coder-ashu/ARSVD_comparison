# models/compression.py
import copy
import os
from typing import Callable, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.utils.extmath import randomized_svd
except Exception:
    randomized_svd = None


def svd_truncate_matrix(mat: np.ndarray, k: int) -> np.ndarray:
    """
    Exact SVD truncation: return U_k * S_k * Vt_k
    """
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    # ensure correct broadcasting: multiply columns of U_k by S_k
    return (U_k * S_k[np.newaxis, :]) @ Vt_k


def randomized_svd_truncate_matrix(mat: np.ndarray,
                                   k: int,
                                   n_oversamples: int = 10,
                                   n_iter: int = 2,
                                   random_state: Optional[int] = None) -> np.ndarray:
    """
    Randomized SVD truncation using sklearn.utils.extmath.randomized_svd if available.
    Falls back to exact SVD if randomized_svd isn't available.
    """
    if randomized_svd is None:
        return svd_truncate_matrix(mat, k)
    # sklearn's randomized_svd returns U, S, Vt with shapes (m, n_components), (n_components,), (n_components, n)
    U, S, Vt = randomized_svd(mat, n_components=k, n_oversamples=n_oversamples,
                              n_iter=n_iter, random_state=random_state)
    return (U * S[np.newaxis, :]) @ Vt


def adaptive_rank_from_entropy(singular_values: np.ndarray, tau: float) -> int:
    """
    Compute adaptive rank using the entropy rule from ARSVD papers.
    Implementation detail:
     - Use energy distribution p_i = s_i^2 / sum(s_j^2)
     - Compute per-component entropy contribution: -p_i * log(p_i) (with zeros handled safely)
     - Take cumulative entropy and find smallest k s.t. cumsum_ent >= tau * total_ent
    Returns at least 1 and at most len(singular_values).
    """
    s = np.array(singular_values, dtype=np.float64)
    if s.size == 0:
        return 1
    energy = s ** 2
    total_energy = energy.sum()
    if total_energy == 0:
        return 1
    p = energy / total_energy

    # safe entropy: only compute -p * log(p) where p > 0
    mask = p > 0
    ent = np.zeros_like(p)
    ent[mask] = -p[mask] * np.log(p[mask])

    cumsum_ent = np.cumsum(ent)
    total_ent = cumsum_ent[-1] if cumsum_ent.size > 0 else 0.0
    if total_ent == 0:
        return 1

    # threshold
    threshold = tau * total_ent
    # searchsorted returns insertion point, +1 to include that index (1-based count)
    idx = int(np.searchsorted(cumsum_ent, threshold, side="left"))
    k = idx + 1
    return max(1, min(k, len(s)))


def truncate_conv2d_weight(weight: torch.Tensor,
                           method: str,
                           k: int,
                           n_oversamples: int = 10,
                           n_iter: int = 2,
                           random_state: Optional[int] = None) -> torch.Tensor:
    """
    Truncate a Conv2d weight tensor (out_c, in_c, kh, kw) to a rank-k approximation of the
    unfolded matrix (out_c, in_c*kh*kw), then reshape back to conv shape.
    method: 'svd' or 'randomized'
    """
    if weight.ndim != 4:
        raise ValueError("Expected a Conv2d weight tensor of shape (out_c, in_c, kh, kw)")

    out_c, in_c, kh, kw = weight.shape
    mat = weight.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)

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
    return isinstance(module, nn.Conv2d)


# theoretical reduction
def compressed_storage_for_svd(out_c: int, in_c: int, kh: int, kw: int, r: int, dtype=np.float32) -> float:
    """
    Return MB required to store (U_r, S_r, Vt_r) for a conv weight shaped (out_c, in_c, kh, kw).
    Stored as:
      U: out_c x r
      S: r
      Vt: r x (in_c*kh*kw)
    """
    bytes_per = np.dtype(dtype).itemsize
    u_elems = out_c * r
    s_elems = r
    vt_elems = r * in_c * kh * kw
    total_bytes = (u_elems + s_elems + vt_elems) * bytes_per
    return total_bytes / (1024 ** 2)  # MB


def model_compressed_storage(model: nn.Module, rank_map: Dict[str, int], dtype=np.float32) -> float:
    total_mb = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            out_c, in_c, kh, kw = module.weight.shape
            k = rank_map.get(name, min(out_c, in_c * kh * kw))
            total_mb += compressed_storage_for_svd(out_c, in_c, kh, kw, k, dtype=dtype)
    return total_mb


def compress_model_svd(model: nn.Module,
                       rank: Optional[int] = None,
                       energy: Optional[float] = None,
                       use_randomized: bool = False,
                       random_state: Optional[int] = None,
                       layer_selector: Optional[Callable[[nn.Module], bool]] = None,
                       svd_kwargs: Optional[Dict[str, Any]] = None,
                       return_rank_map: bool = False,
                       return_weights: bool = False) -> Any:
    """
    Compress a model by truncating Conv2d layer weights (dense reconstruction).
    Backwards-compatible: by default returns new_model.
    If return_rank_map=True returns (new_model, rank_map).
    If return_rank_map & return_weights True returns (new_model, rank_map, weights_map).
    """
    if layer_selector is None:
        layer_selector = default_layer_selector
    if svd_kwargs is None:
        svd_kwargs = {}

    new_model = copy.deepcopy(model)
    rank_map: Dict[str, int] = {}
    weights_map: Dict[str, torch.Tensor] = {}

    for name, module in new_model.named_modules():
        if not layer_selector(module):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        w = module.weight.data
        out_c, in_c, kh, kw = w.shape
        mat = w.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)

        # compute singular values (randomized if requested)
        try:
            if use_randomized and randomized_svd is not None:
                max_possible = min(mat.shape[0], mat.shape[1])
                n_components = min(max_possible, rank or max_possible)
                n_oversamples = svd_kwargs.get("n_oversamples", 10)
                n_iter = svd_kwargs.get("n_iter", 2)
                U_tmp, S_tmp, Vt_tmp = randomized_svd(mat, n_components=n_components,
                                                      n_oversamples=n_oversamples,
                                                      n_iter=n_iter,
                                                      random_state=random_state)
                s = S_tmp
            else:
                s = np.linalg.svd(mat, compute_uv=False)
        except Exception:
            s = np.linalg.svd(mat, compute_uv=False)

        # determine k
        if energy is not None:
            energy_cumsum = np.cumsum(s ** 2) / np.sum(s ** 2)
            k = int(np.searchsorted(energy_cumsum, energy) + 1)
        elif rank is not None:
            k = int(rank)
        else:
            raise ValueError("Either 'rank' or 'energy' must be provided to compress_model_svd")

        max_rank = min(mat.shape[0], mat.shape[1])
        k_use = max(1, min(int(k), max_rank))

        method = "randomized" if use_randomized else "svd"
        new_w = truncate_conv2d_weight(w, method=method, k=k_use,
                                       n_oversamples=svd_kwargs.get("n_oversamples", 10),
                                       n_iter=svd_kwargs.get("n_iter", 2),
                                       random_state=random_state)

        module.weight.data.copy_(new_w)

        rank_map[name] = int(k_use)
        if return_weights:
            weights_map[name] = new_w.detach().cpu().clone()

    if return_rank_map and return_weights:
        return new_model, rank_map, weights_map
    if return_rank_map:
        return new_model, rank_map
    return new_model


def compress_model_arsvd(model: nn.Module,
                         tau: float = 0.95,
                         recon_method: str = "randomized",
                         n_oversamples: int = 10,
                         n_iter: int = 2,
                         random_state: Optional[int] = None,
                         layer_selector: Optional[Callable[[nn.Module], bool]] = None,
                         return_rank_map: bool = False,
                         return_weights: bool = False) -> Any:
    """
    ARSVD: compute per-layer rank via entropy rule and reconstruct using recon_method.
    Returns new_model by default; optionally (new_model, rank_map) or (new_model, rank_map, weights_map).
    Implementation notes / fixes:
      - Entropy is computed over energy distribution (s**2).
      - Randomized SVD approximate rank selection tuned to reasonable defaults.
      - Keeps function names & structure from original code.
    """
    if layer_selector is None:
        layer_selector = default_layer_selector

    new_model = copy.deepcopy(model)
    rank_map: Dict[str, int] = {}
    weights_map: Dict[str, torch.Tensor] = {}

    for name, module in new_model.named_modules():
        if not layer_selector(module):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        w = module.weight.data
        out_c, in_c, kh, kw = w.shape
        mat = w.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)

        # compute singular values (approx with randomized if recon_method is randomized)
        try:
            min_dim = min(mat.shape[0], mat.shape[1])
            if (recon_method == "randomized") and (randomized_svd is not None):
                # choose a reasonable approximation rank for randomized SVD
                # we avoid forcing extremely large approximate ranks; cap at min_dim
                approx_rank = min(min_dim, max(32, min(256, min_dim)))
                U_tmp, S_tmp, Vt_tmp = randomized_svd(mat, n_components=approx_rank,
                                                      n_oversamples=n_oversamples,
                                                      n_iter=max(1, n_iter),
                                                      random_state=random_state)
                s = S_tmp
            else:
                s = np.linalg.svd(mat, compute_uv=False)
        except Exception:
            s = np.linalg.svd(mat, compute_uv=False)

        # adaptive rank by entropy rule (energy-based)
        k = adaptive_rank_from_entropy(s, tau)
        k_use = max(1, min(int(k), min(mat.shape[0], mat.shape[1])))

        # reconstruct using requested method
        new_w = truncate_conv2d_weight(w, method=recon_method, k=k_use,
                                       n_oversamples=n_oversamples, n_iter=n_iter, random_state=random_state)
        module.weight.data.copy_(new_w)

        rank_map[name] = int(k_use)
        if return_weights:
            weights_map[name] = new_w.detach().cpu().clone()

    if return_rank_map and return_weights:
        return new_model, rank_map, weights_map
    if return_rank_map:
        return new_model, rank_map
    return new_model


# actual parameter reduction
def factorize_conv2d_module(module: nn.Conv2d, k: int, use_randomized: bool = False, random_state: Optional[int] = None,
                            n_oversamples: int = 10, n_iter: int = 2) -> nn.Sequential:
    """
    Create a factorized two-layer replacement for a Conv2d:
      conv1: in_c -> k, kernel (kh,kw)
      conv2: k -> out_c, kernel 1x1

    Uses SVD (or randomized SVD if requested) to create weights.
    """
    W = module.weight.detach().cpu().numpy()
    out_c, in_c, kh, kw = W.shape
    mat = W.reshape(out_c, in_c * kh * kw)

    if use_randomized and randomized_svd is not None:
        n_components = min(mat.shape[0], mat.shape[1], max(1, k))
        U, S, Vt = randomized_svd(mat, n_components=n_components, n_oversamples=n_oversamples,
                                  n_iter=n_iter, random_state=random_state)
        # ensure we only keep k components even if randomized returned more (numerical)
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
    else:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]

    # conv1: in_c -> k (kernel kh x kw)
    conv1 = nn.Conv2d(in_c, k, kernel_size=(kh, kw), stride=module.stride,
                      padding=module.padding, dilation=module.dilation, bias=False)
    # conv2: k -> out_c (1x1)
    conv2 = nn.Conv2d(k, out_c, kernel_size=1, bias=(module.bias is not None))

    Vt_reshaped = Vt.reshape(k, in_c, kh, kw)
    conv1.weight.data.copy_(torch.from_numpy(Vt_reshaped).to(conv1.weight.device).type_as(conv1.weight))

    US = (U * S[np.newaxis, :])  # (out_c, k)
    US_reshaped = US.reshape(out_c, k, 1, 1)
    conv2.weight.data.copy_(torch.from_numpy(US_reshaped).to(conv2.weight.device).type_as(conv2.weight))

    if module.bias is not None:
        conv2.bias.data.copy_(module.bias.data.clone())

    return nn.Sequential(conv1, conv2)


def compress_model_factorized_copy(model: nn.Module,
                                   rank_map: Optional[Dict[str, int]] = None,
                                   default_rank: int = 32,
                                   use_randomized: bool = False,
                                   random_state: Optional[int] = None,
                                   n_oversamples: int = 10,
                                   n_iter: int = 2) -> nn.Module:
    """
    Deep-copy model and replace Conv2d modules with factorized Sequential modules
    according to rank_map (or default_rank). Only replaces when factorization reduces params.
    """
    new_model = copy.deepcopy(model)

    for name, module in list(new_model.named_modules()):
        if isinstance(module, nn.Conv2d):
            k = default_rank
            if rank_map and name in rank_map:
                k = rank_map[name]
            out_c, in_c, kh, kw = module.weight.shape
            max_rank = min(out_c, in_c * kh * kw)
            k_use = max(1, min(int(k), max_rank - 1))
            orig_params = out_c * in_c * kh * kw + (module.bias.numel() if module.bias is not None else 0)
            fact_params = (k_use * in_c * kh * kw) + (out_c * k_use) + (module.bias.numel() if module.bias is not None else 0)
            if fact_params >= orig_params:
                continue
            # find parent object to set attribute
            parent = new_model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1], factorize_conv2d_module(module, int(k_use), use_randomized=use_randomized,
                                                                  random_state=random_state, n_oversamples=n_oversamples,
                                                                  n_iter=n_iter))
            except Exception:
                # if factorization fails, skip replacement
                continue

    return new_model


# -----------------------
# Small utilities
# -----------------------
def model_size_bytes(model: nn.Module) -> int:
    tmp = "tmp_model_for_size.pth"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return size
