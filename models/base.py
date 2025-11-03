# models/base.py
import copy
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import torch
import optuna

from .compression import compress_model_svd, compress_model_arsvd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------
# Abstract Base Class
# -------------------------
class Model(ABC):
    """
    Abstract base class for all model strategies.
    """

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train or fine-tune the model."""
        raise NotImplementedError

    @abstractmethod
    def optimize(self, trial: optuna.trial.Trial, *args, **kwargs):
        """Optuna hyperparameter optimization objective."""
        raise NotImplementedError


# -------------------------
# Baseline Model
# -------------------------
class BaselineModel(Model):
    """
    Wrapper around an uncompressed model (baseline training).
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_fn: Optional[Callable[..., torch.nn.Module]] = None,
                 evaluate_fn: Optional[Callable[..., float]] = None,
                 device: str = "cpu"):
        self.model = model
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.model.to(self.device)

    def train(self, *args, **kwargs):
        if self.train_fn is None:
            raise NotImplementedError("train_fn must be provided for BaselineModel.")
        logger.info("Training baseline model...")
        trained = self.train_fn(self.model, *args, **kwargs)
        trained.to(self.device)
        return trained

    def optimize(self, trial: optuna.trial.Trial, *args, **kwargs):
        if self.train_fn is None or self.evaluate_fn is None:
            raise NotImplementedError("train_fn and evaluate_fn required for BaselineModel.optimize.")
        logger.info("Optimizing baseline model via Optuna...")
        trained = self.train_fn(self.model, trial=trial, *args, **kwargs)
        metric = self.evaluate_fn(trained, *args, **kwargs)
        return float(metric)


# -------------------------
# SVD Truncation Strategy
# -------------------------
class SVDTruncation(Model):
    """
    Applies SVD truncation (fixed rank or energy) to model weights.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_fn: Optional[Callable[..., torch.nn.Module]] = None,
                 evaluate_fn: Optional[Callable[..., float]] = None,
                 device: str = "cpu"):
        self.original_model = model
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    def train(self, model_to_train: torch.nn.Module, *args, **kwargs):
        if self.train_fn is None:
            raise NotImplementedError("train_fn required for SVDTruncation fine-tuning.")
        model_to_train.to(self.device)
        return self.train_fn(model_to_train, *args, **kwargs)

    def optimize(self, trial: optuna.trial.Trial, *args, **kwargs):
        if self.evaluate_fn is None:
            raise NotImplementedError("evaluate_fn required for SVDTruncation.optimize.")

        use_energy = trial.suggest_categorical("use_energy", [False, True])
        use_randomized = trial.suggest_categorical("use_randomized", [False, True])
        method = "randomized" if use_randomized else "svd"

        if use_energy:
            energy = trial.suggest_float("energy", 0.7, 0.999)
            compressed = compress_model_svd(self.original_model, energy=energy, use_randomized=use_randomized)
        else:
            rank_k = trial.suggest_int("rank_k", 4, 128)
            compressed = compress_model_svd(self.original_model, rank=rank_k, use_randomized=use_randomized)

        fine_tune = trial.suggest_categorical("fine_tune", [False, True])
        if fine_tune and self.train_fn is not None:
            trained = self.train(compressed, trial=trial, *args, **kwargs)
        else:
            trained = compressed

        trained.to(self.device)
        metric = self.evaluate_fn(trained, *args, **kwargs)
        return float(metric)


# -------------------------
# ARSVD Strategy
# -------------------------
class ARSVD(Model):
    """
    Adaptive-Rank SVD compression per layer using entropy threshold tau.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 entropy_threshold: float = 0.95,
                 train_fn: Optional[Callable[..., torch.nn.Module]] = None,
                 evaluate_fn: Optional[Callable[..., float]] = None,
                 device: str = "cpu"):
        self.original_model = model
        self.tau = entropy_threshold
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    def compress_once(self,
                      recon_method: str = "randomized",
                      n_oversamples: int = 10,
                      n_iter: int = 2,
                      random_state: Optional[int] = None) -> torch.nn.Module:
        """
        Apply ARSVD compression once and return a new model.
        """
        return compress_model_arsvd(self.original_model,
                                    tau=self.tau,
                                    recon_method=recon_method,
                                    n_oversamples=n_oversamples,
                                    n_iter=n_iter,
                                    random_state=random_state)

    def train(self, *args, **kwargs):
        if self.train_fn is None:
            raise NotImplementedError("train_fn must be provided for ARSVD fine-tuning.")
        raise NotImplementedError("Use compress_once() -> train_fn(model, ...) pattern in pipeline.")

    def optimize(self, trial: optuna.trial.Trial, *args, **kwargs):
        if self.evaluate_fn is None:
            raise NotImplementedError("evaluate_fn required for ARSVD.optimize.")

        tau = trial.suggest_float("tau", 0.5, 0.995)
        recon_method = trial.suggest_categorical("recon_method", ["randomized", "svd"])
        n_oversamples = trial.suggest_int("n_oversamples", 5, 20)
        n_iter = trial.suggest_int("n_iter", 0, 4)
        random_state = trial.suggest_int("rs", 0, 9999)

        self.tau = tau
        compressed = self.compress_once(recon_method=recon_method,
                                        n_oversamples=n_oversamples,
                                        n_iter=n_iter,
                                        random_state=random_state)

        fine_tune = trial.suggest_categorical("fine_tune", [False, True])
        if fine_tune and self.train_fn is not None:
            trained = self.train_fn(compressed, trial=trial, *args, **kwargs)
        else:
            trained = compressed

        trained.to(self.device)
        metric = self.evaluate_fn(trained, *args, **kwargs)
        return float(metric)
