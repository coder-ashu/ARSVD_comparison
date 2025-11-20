# models/compression.py
import copy
import os
from typing import Callable, Optional, Dict, Any, Tuple
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.utils.extmath import randomized_svd
except Exception:
    randomized_svd = None


# ------------------------
# Low-level SVD helpers
# ------------------------
def svd_truncate_matrix(mat: np.ndarray, k: int) -> np.ndarray:
    """
    Exact SVD truncation: return (U_k * S_k) @ Vt_k
    mat: 2D numpy array
    """
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    return (U_k * S_k[np.newaxis, :]) @ Vt_k


def randomized_svd_truncate_matrix(mat: np.ndarray,
                                   k: int,
                                   n_oversamples: int = 10,
                                   n_iter: int = 2,
                                   random_state: Optional[int] = None) -> np.ndarray:
    """
    Randomized SVD truncation using sklearn's randomized_svd if available.
    Falls back to exact SVD if not available.
    """
    if randomized_svd is None:
        return svd_truncate_matrix(mat, k)
    U, S, Vt = randomized_svd(mat, n_components=k, n_oversamples=n_oversamples,
                              n_iter=n_iter, random_state=random_state)
    return (U * S[np.newaxis, :]) @ Vt


# ------------------------
# ARSVD entropy rank selector
# ------------------------
def adaptive_rank_from_entropy(singular_values: np.ndarray, tau: float) -> int:
    """
    Adaptive rank computed from entropy of energy distribution.
    Uses energy p_i = s_i^2 / sum(s_j^2) then cumulative entropy rule.
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
    mask = p > 0
    ent = np.zeros_like(p)
    ent[mask] = -p[mask] * np.log(p[mask])
    cumsum_ent = np.cumsum(ent)
    total_ent = cumsum_ent[-1] if cumsum_ent.size > 0 else 0.0
    if total_ent == 0:
        return 1
    threshold = tau * total_ent
    idx = int(np.searchsorted(cumsum_ent, threshold, side="left"))
    k = idx + 1
    return max(1, min(k, len(s)))


# ------------------------
# Utilities to resolve nested module names & set modules robustly
# ------------------------
def _resolve_parent_and_attr(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """
    Given a root module and a dotted name e.g. "features.28.conv", return (parent_module, final_attr).
    Handles numeric parts used for nn.Sequential / ModuleList by indexing.
    """
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            idx = int(p)
            parent = parent[idx]
        else:
            parent = getattr(parent, p)
    final = parts[-1]
    return parent, final


def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    """
    Set a child module referenced by dotted name to new_module.
    Handles numeric indices for Sequential/ModuleList.
    """
    parent, final = _resolve_parent_and_attr(root, name)
    # attempt integer indexing if final looks like a digit and parent supports item assignment
    if final.isdigit():
        try:
            idx = int(final)
            # if parent is a Sequential or ModuleList, we can set by index
            if isinstance(parent, (nn.Sequential, nn.ModuleList)) or hasattr(parent, "__setitem__"):
                parent[idx] = new_module
                return
        except Exception:
            pass
    # else use setattr
    setattr(parent, final, new_module)


# ------------------------
# Truncation helpers for Conv2d and Linear
# ------------------------
def truncate_conv2d_weight(weight: torch.Tensor,
                           method: str,
                           k: int,
                           n_oversamples: int = 10,
                           n_iter: int = 2,
                           random_state: Optional[int] = None) -> torch.Tensor:
    """
    Truncate a Conv2d weight tensor (out_c, in_c, kh, kw) by computing approximate/exact
    low-rank reconstruction of the unfolded matrix (out_c, in_c*kh*kw). Returns tensor in same shape & dtype.
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


def truncate_linear_weight(weight: torch.Tensor,
                           method: str,
                           k: int,
                           n_oversamples: int = 10,
                           n_iter: int = 2,
                           random_state: Optional[int] = None) -> torch.Tensor:
    """
    Truncate a Linear weight (out, in) using SVD/randomized SVD and return tensor of same shape/dtype.
    """
    if weight.ndim != 2:
        raise ValueError("Expected a Linear weight tensor of shape (out, in)")
    mat = weight.detach().cpu().numpy()
    max_rank = min(mat.shape[0], mat.shape[1])
    k_use = max(1, min(int(k), max_rank))
    if method == "svd":
        approx = svd_truncate_matrix(mat, k_use)
    elif method == "randomized":
        approx = randomized_svd_truncate_matrix(mat, k_use, n_oversamples=n_oversamples, n_iter=n_iter,
                                                random_state=random_state)
    else:
        raise ValueError(f"Unknown method '{method}'")
    return torch.from_numpy(approx).to(weight.device).type_as(weight)


# ------------------------
# Default layer selector
# ------------------------
def default_layer_selector(module: nn.Module) -> bool:
    """
    By default compress both Conv2d and Linear layers. If you want conv-only behavior,
    pass a custom layer_selector to compress_model_svd / compress_model_arsvd.
    """
    return isinstance(module, (nn.Conv2d, nn.Linear))


# ------------------------
# Theoretical storage calculations
# ------------------------
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
    """
    Sum MB for conv and linear layers according to rank_map.
    rank_map keys are module names (as from model.named_modules()).
    """
    total_mb = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            out_c, in_c, kh, kw = module.weight.shape
            k = rank_map.get(name, min(out_c, in_c * kh * kw))
            total_mb += compressed_storage_for_svd(out_c, in_c, kh, kw, k, dtype=dtype)
        elif isinstance(module, nn.Linear):
            out, inp = module.weight.shape
            k = rank_map.get(name, min(out, inp))
            bytes_per = np.dtype(dtype).itemsize
            total_bytes = (out * k + k + k * inp) * bytes_per  # U + S + Vt
            total_mb += total_bytes / (1024 ** 2)
    return total_mb


# ------------------------
# Factorize conv2d and linear modules (actual param reduction)
# ------------------------
def factorize_conv2d_module(module: nn.Conv2d, k: int, use_randomized: bool = False, random_state: Optional[int] = None,
                            n_oversamples: int = 10, n_iter: int = 2) -> nn.Sequential:
    """
    Create a factorized two-layer replacement for a Conv2d:
      conv1: in_c -> k, kernel (kh,kw)  (no bias)
      conv2: k -> out_c, kernel 1x1 (bias copied if present)
    Weights are placed on the same device/dtype as the original module (best-effort).
    """
    W = module.weight.detach().cpu().numpy()
    out_c, in_c, kh, kw = W.shape
    mat = W.reshape(out_c, in_c * kh * kw)

    if use_randomized and randomized_svd is not None:
        n_components = min(mat.shape[0], mat.shape[1], max(1, k))
        U, S, Vt = randomized_svd(mat, n_components=n_components, n_oversamples=n_oversamples,
                                  n_iter=n_iter, random_state=random_state)
        U = U[:, :k]; S = S[:k]; Vt = Vt[:k, :]
    else:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        U = U[:, :k]; S = S[:k]; Vt = Vt[:k, :]

    # conv1: in_c -> k (kernel kh x kw)  ; conv1 has no bias
    conv1 = nn.Conv2d(in_c, k, kernel_size=(kh, kw), stride=module.stride,
                      padding=module.padding, dilation=module.dilation, bias=False)
    # conv2: k -> out_c (1x1), keep bias if original had one
    conv2 = nn.Conv2d(k, out_c, kernel_size=1, bias=(module.bias is not None))

    # set weights (Vt -> conv1, U*S -> conv2)
    Vt_reshaped = Vt.reshape(k, in_c, kh, kw)
    conv1.weight.data.copy_(torch.from_numpy(Vt_reshaped).to(conv1.weight.device).type_as(conv1.weight))

    US = (U * S[np.newaxis, :])  # (out_c, k)
    US_reshaped = US.reshape(out_c, k, 1, 1)
    conv2.weight.data.copy_(torch.from_numpy(US_reshaped).to(conv2.weight.device).type_as(conv2.weight))

    if module.bias is not None:
        conv2.bias.data.copy_(module.bias.data.clone())

    factor = nn.Sequential(conv1, conv2)

    # move factor to same device & dtype as original module (best-effort)
    try:
        factor = factor.to(module.weight.device)
        # ensure dtype matches
        for p in factor.parameters():
            p.data = p.data.to(module.weight.dtype)
    except Exception:
        pass

    return factor


def factorize_linear_module(module: nn.Linear, k: int, use_randomized: bool = False,
                            random_state: Optional[int] = None, n_oversamples: int = 10, n_iter: int = 2) -> nn.Sequential:
    """
    Factorize a Linear(out, in) into first: in->k (bias=False), second: k->out (bias=original_bias).
    """
    W = module.weight.detach().cpu().numpy()
    out, inp = W.shape
    if use_randomized and randomized_svd is not None:
        n_comp = min(out, inp, max(1, k))
        U, S, Vt = randomized_svd(W, n_components=n_comp, n_oversamples=n_oversamples, n_iter=n_iter, random_state=random_state)
        U = U[:, :k]; S = S[:k]; Vt = Vt[:k, :]
    else:
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        U = U[:, :k]; S = S[:k]; Vt = Vt[:k, :]

    A = (U * S[np.newaxis, :])  # (out, k)

    first = nn.Linear(in_features=Vt.shape[1], out_features=k, bias=False)
    second = nn.Linear(in_features=k, out_features=A.shape[0], bias=(module.bias is not None))

    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(Vt).to(first.weight.device).type_as(first.weight))
        second.weight.copy_(torch.from_numpy(A).to(second.weight.device).type_as(second.weight))
        if module.bias is not None:
            second.bias.copy_(module.bias.data.clone())

    # move to module device/dtype
    try:
        first = first.to(module.weight.device)
        second = second.to(module.weight.device)
        for p in first.parameters():
            p.data = p.data.to(module.weight.dtype)
        for p in second.parameters():
            p.data = p.data.to(module.weight.dtype)
    except Exception:
        pass

    return nn.Sequential(first, second)


# ------------------------
# High-level compression: SVD (fixed rank)
# ------------------------
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
    Compress a model by truncating Conv2d/Linear layer weights or replacing with factorized modules.
    Returns:
      - new_model (default)
      - (new_model, rank_map) if return_rank_map=True
      - (new_model, rank_map, weights_map) if return_weights=True
    """
    if layer_selector is None:
        layer_selector = default_layer_selector
    if svd_kwargs is None:
        svd_kwargs = {}

    new_model = copy.deepcopy(model)
    rank_map: Dict[str, int] = {}
    weights_map: Dict[str, Any] = {}

    for name, module in list(new_model.named_modules()):
        if not layer_selector(module):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        # handle Conv2d
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            out_c, in_c, kh, kw = w.shape
            mat = w.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)

            # singular values
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
                k_val = int(np.searchsorted(energy_cumsum, energy) + 1)
            elif rank is not None:
                k_val = int(rank)
            else:
                raise ValueError("Either 'rank' or 'energy' must be provided to compress_model_svd")

            max_rank = min(mat.shape[0], mat.shape[1])
            k_use = max(1, min(int(k_val), max_rank))

            # decide factorization vs dense truncation
            orig_params = out_c * in_c * kh * kw + (module.bias.numel() if module.bias is not None else 0)
            fact_params = (k_use * in_c * kh * kw) + (out_c * k_use) + (module.bias.numel() if module.bias is not None else 0)

            if fact_params < orig_params:
                try:
                    factor_mod = factorize_conv2d_module(module, int(k_use), use_randomized=use_randomized,
                                                         random_state=random_state,
                                                         n_oversamples=svd_kwargs.get("n_oversamples", 10),
                                                         n_iter=svd_kwargs.get("n_iter", 2))
                    _set_module_by_name(new_model, name, factor_mod)
                    rank_map[name] = int(k_use)
                    if return_weights:
                        weights_map[name] = factor_mod
                    continue
                except Exception:
                    pass

            # fallback dense truncated weight
            method = "randomized" if use_randomized else "svd"
            new_w = truncate_conv2d_weight(w, method=method, k=k_use,
                                           n_oversamples=svd_kwargs.get("n_oversamples", 10),
                                           n_iter=svd_kwargs.get("n_iter", 2),
                                           random_state=random_state)
            module.weight.data.copy_(new_w)
            rank_map[name] = int(k_use)
            if return_weights:
                weights_map[name] = new_w.detach().cpu().clone()

        # handle Linear
        elif isinstance(module, nn.Linear):
            w = module.weight.data
            mat = w.detach().cpu().numpy()
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

            if energy is not None:
                energy_cumsum = np.cumsum(s ** 2) / np.sum(s ** 2)
                k_val = int(np.searchsorted(energy_cumsum, energy) + 1)
            elif rank is not None:
                k_val = int(rank)
            else:
                raise ValueError("Either 'rank' or 'energy' must be provided to compress_model_svd")

            max_rank = min(mat.shape[0], mat.shape[1])
            k_use = max(1, min(int(k_val), max_rank))

            out, inp = mat.shape
            orig_params = out * inp + (module.bias.numel() if module.bias is not None else 0)
            fact_params = (k_use * inp) + (out * k_use) + (module.bias.numel() if module.bias is not None else 0)

            if fact_params < orig_params:
                try:
                    factor_mod = factorize_linear_module(module, int(k_use), use_randomized=use_randomized,
                                                         random_state=random_state,
                                                         n_oversamples=svd_kwargs.get("n_oversamples", 10),
                                                         n_iter=svd_kwargs.get("n_iter", 2))
                    _set_module_by_name(new_model, name, factor_mod)
                    rank_map[name] = int(k_use)
                    if return_weights:
                        weights_map[name] = factor_mod
                    continue
                except Exception:
                    pass

            # fallback dense truncation
            method = "randomized" if use_randomized else "svd"
            new_w = truncate_linear_weight(w, method=method, k=k_use,
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


# ------------------------
# ARSVD (adaptive rank by entropy)
# ------------------------
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
    """
    if layer_selector is None:
        layer_selector = default_layer_selector

    new_model = copy.deepcopy(model)
    rank_map: Dict[str, int] = {}
    weights_map: Dict[str, Any] = {}

    for name, module in list(new_model.named_modules()):
        if not layer_selector(module):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        # common preparation
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            out_c, in_c, kh, kw = w.shape
            mat = w.detach().cpu().numpy().reshape(out_c, in_c * kh * kw)
        elif isinstance(module, nn.Linear):
            w = module.weight.data
            mat = w.detach().cpu().numpy()
        else:
            continue

        # singular values (approx if randomized recon)
        try:
            if (recon_method == "randomized") and (randomized_svd is not None):
                min_dim = min(mat.shape[0], mat.shape[1])
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

        k = adaptive_rank_from_entropy(s, tau)
        k_use = max(1, min(int(k), min(mat.shape[0], mat.shape[1])))

        # decide factorization vs dense truncation (conv & linear separate)
        if isinstance(module, nn.Conv2d):
            out_c, in_c, kh, kw = module.weight.shape
            orig_params = out_c * in_c * kh * kw + (module.bias.numel() if module.bias is not None else 0)
            fact_params = (k_use * in_c * kh * kw) + (out_c * k_use) + (module.bias.numel() if module.bias is not None else 0)
            if fact_params < orig_params:
                try:
                    factor_mod = factorize_conv2d_module(module, int(k_use), use_randomized=(recon_method == "randomized"),
                                                         random_state=random_state, n_oversamples=n_oversamples, n_iter=n_iter)
                    _set_module_by_name(new_model, name, factor_mod)
                    rank_map[name] = int(k_use)
                    if return_weights:
                        weights_map[name] = factor_mod
                    continue
                except Exception:
                    pass
            new_w = truncate_conv2d_weight(w, method=recon_method, k=k_use,
                                           n_oversamples=n_oversamples, n_iter=n_iter, random_state=random_state)
            module.weight.data.copy_(new_w)
            rank_map[name] = int(k_use)
            if return_weights:
                weights_map[name] = new_w.detach().cpu().clone()

        elif isinstance(module, nn.Linear):
            out, inp = module.weight.shape
            orig_params = out * inp + (module.bias.numel() if module.bias is not None else 0)
            fact_params = (k_use * inp) + (out * k_use) + (module.bias.numel() if module.bias is not None else 0)
            if fact_params < orig_params:
                try:
                    factor_mod = factorize_linear_module(module, int(k_use), use_randomized=(recon_method == "randomized"),
                                                         random_state=random_state, n_oversamples=n_oversamples, n_iter=n_iter)
                    _set_module_by_name(new_model, name, factor_mod)
                    rank_map[name] = int(k_use)
                    if return_weights:
                        weights_map[name] = factor_mod
                    continue
                except Exception:
                    pass
            new_w = truncate_linear_weight(w, method=recon_method, k=k_use,
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


# ------------------------
# Build factorized copy (actual parameter reduction)
# ------------------------
def compress_model_factorized_copy(model: nn.Module,
                                   rank_map: Optional[Dict[str, int]] = None,
                                   default_rank: int = 32,
                                   use_randomized: bool = False,
                                   random_state: Optional[int] = None,
                                   n_oversamples: int = 10,
                                   n_iter: int = 2) -> nn.Module:
    """
    Deep-copy model and replace Conv2d/Linear modules with factorized Sequential modules
    according to rank_map (or default_rank). Only replaces when factorization reduces params.
    """
    new_model = copy.deepcopy(model)

    for name, module in list(new_model.named_modules()):
        if isinstance(module, nn.Conv2d):
            k = default_rank if (rank_map is None or name not in rank_map) else rank_map[name]
            out_c, in_c, kh, kw = module.weight.shape
            max_rank = min(out_c, in_c * kh * kw)
            k_use = max(1, min(int(k), max_rank - 1))  # ensure some reduction
            orig_params = out_c * in_c * kh * kw + (module.bias.numel() if module.bias is not None else 0)
            fact_params = (k_use * in_c * kh * kw) + (out_c * k_use) + (module.bias.numel() if module.bias is not None else 0)
            if fact_params >= orig_params:
                continue
            try:
                factor = factorize_conv2d_module(module, int(k_use), use_randomized=use_randomized,
                                                 random_state=random_state, n_oversamples=n_oversamples, n_iter=n_iter)
                _set_module_by_name(new_model, name, factor)
            except Exception:
                continue

        elif isinstance(module, nn.Linear):
            k = default_rank if (rank_map is None or name not in rank_map) else rank_map[name]
            out, inp = module.weight.shape
            max_rank = min(out, inp)
            k_use = max(1, min(int(k), max_rank - 1))
            orig_params = out * inp + (module.bias.numel() if module.bias is not None else 0)
            fact_params = (k_use * inp) + (out * k_use) + (module.bias.numel() if module.bias is not None else 0)
            if fact_params >= orig_params:
                continue
            try:
                factor = factorize_linear_module(module, int(k_use), use_randomized=use_randomized,
                                                 random_state=random_state, n_oversamples=n_oversamples, n_iter=n_iter)
                _set_module_by_name(new_model, name, factor)
            except Exception:
                continue

    return new_model


# ------------------------
# Model size helper
# ------------------------
def model_size_bytes(model: nn.Module) -> int:
    tmp = "tmp_model_for_size.pth"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return size
