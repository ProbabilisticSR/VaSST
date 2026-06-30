"""
VaSST Python Implementation.
"""

# required imports
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any
from tqdm.auto import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
import os
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import time
import re

# ----------------------------
# Utilities
# ----------------------------
def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return a / (b.sign() * b.abs().clamp_min(eps))

def safe_pow(a: torch.Tensor, power: float, eps: float = 1e-6) -> torch.Tensor:
    return (a.abs().clamp_min(eps)) ** power

def safe_sin(a: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sin(a)

def safe_cos(a: torch.Tensor) -> torch.Tensor:
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cos(a)

def safe_inv(a: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    denom = a.sign() * a.abs().clamp_min(eps)
    return 1.0 / denom

def safe_exp(a: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    a = torch.nan_to_num(a, nan=0.0, posinf=clip, neginf=-clip)
    a = a.clamp(-clip, clip)
    return torch.exp(a)

def safe_log(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = torch.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
    return torch.log(a.abs().clamp_min(eps))


# ----------------------------
# Operator specification
# ----------------------------
@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def default_operator_set() -> List[OperatorSpec]:
    ops: List[OperatorSpec] = []
    
    # binary
    ops.append(OperatorSpec("add", 2, lambda a, b: a + b))
    ops.append(OperatorSpec("sub", 2, lambda a, b: a - b))
    ops.append(OperatorSpec("mul", 2, lambda a, b: a * b))
    ops.append(OperatorSpec("div", 2, lambda a, b: safe_div(a, b)))
    
    # unary
    ops.append(OperatorSpec("sq",  1, lambda a, b=None: safe_pow(a, 2.0)))
    ops.append(OperatorSpec("sin", 1, lambda a, b=None: safe_sin(a)))
    ops.append(OperatorSpec("cos", 1, lambda a, b=None: safe_cos(a)))
    ops.append(OperatorSpec("inv", 1, lambda a, b=None: safe_inv(a)))
    ops.append(OperatorSpec("exp", 1, lambda a, b=None: safe_exp(a)))
    ops.append(OperatorSpec("log", 1, lambda a, b=None: safe_log(a)))
    
    return ops

# making a custom operator set
def make_operator_set(names: List[str]) -> List[OperatorSpec]:
    name_to_op = {
        "add": OperatorSpec("add", 2, lambda a, b: a + b),
        "sub": OperatorSpec("sub", 2, lambda a, b: a - b),
        "mul": OperatorSpec("mul", 2, lambda a, b: a * b),
        "div": OperatorSpec("div", 2, lambda a, b: safe_div(a, b)),
        "sq":  OperatorSpec("sq",  1, lambda a, b=None: safe_pow(a, 2.0)),
        "sin": OperatorSpec("sin", 1, lambda a, b=None: safe_sin(a)),
        "cos": OperatorSpec("cos", 1, lambda a, b=None: safe_cos(a)),
        "inv": OperatorSpec("inv", 1, lambda a, b=None: safe_inv(a)),
        "exp": OperatorSpec("exp", 1, lambda a, b=None: safe_exp(a)),
        "log": OperatorSpec("log", 1, lambda a, b=None: safe_log(a)),
    }
    return [name_to_op[nm] for nm in names]


# ----------------------------
# Tree skeleton helpers
# ----------------------------
def num_nodes_full_binary(depth: int) -> int:
    return 2 ** (depth + 1) - 1

def node_depths(depth: int, device=None) -> torch.Tensor:
    N = num_nodes_full_binary(depth)
    d = torch.empty(N, dtype=torch.long, device=device)
    for i in range(N):
        d[i] = int(math.floor(math.log2(i + 1)))
    return d

def heap_children(i: int) -> tuple[int, int]:
    return 2 * i + 1, 2 * i + 2

def is_leaf(i: int, N: int) -> bool:
    return (2 * i + 2) >= N

def prune_by_expand(e_hat: torch.Tensor) -> torch.Tensor:
    N = e_hat.numel()
    e = e_hat.clone()
    for i in range(N):
        if e[i].item() == 0:
            stack = [i]
            while stack:
                k = stack.pop()
                l, r = heap_children(k)
                if r < N:
                    e[l] = 0
                    e[r] = 0
                    stack.append(l)
                    stack.append(r)
    return e


def tree_to_expression(
    e_hat: torch.Tensor,
    op_hat: torch.Tensor,
    ft_hat: torch.Tensor,
    operators: List[OperatorSpec],
    feature_names: Optional[List[str]] = None,
    i: int = 0,
) -> str:
    N = e_hat.numel()
    p = int(ft_hat.max().item()) + 1 if feature_names is None else len(feature_names)
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(p)]

    if i >= N:
        return feature_names[int(ft_hat[N - 1].item())]

    if is_leaf(i, N):
        return feature_names[int(ft_hat[i].item())]

    if int(e_hat[i].item()) == 0:
        return feature_names[int(ft_hat[i].item())]

    op = operators[int(op_hat[i].item())]
    l, r = heap_children(i)

    left_expr = tree_to_expression(e_hat, op_hat, ft_hat, operators, feature_names, l)

    if op.arity == 1:
        return f"{op.name}({left_expr})"

    right_expr = tree_to_expression(e_hat, op_hat, ft_hat, operators, feature_names, r)
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(op.name, op.name)
    return f"({left_expr} {sym} {right_expr})"


# ----------------------------
# Collapsed BLR
# ----------------------------
@dataclass
class BLRHyperparams:
    mu0: torch.Tensor
    Sigma0: torch.Tensor
    a0: float
    b0: float


def robust_cholesky(A: torch.Tensor, jitter: float = 1e-6, max_tries: int = 10) -> torch.Tensor:
    K = A.shape[-1]
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    jit = float(jitter)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(A + jit * I)
        except RuntimeError:
            jit *= 10.0
    raise RuntimeError("robust_cholesky failed after jitter escalation.")


def log_marginal_likelihood_blr(
    y: torch.Tensor,
    T: torch.Tensor,
    hyp: BLRHyperparams,
    jitter: float = 1e-6,
) -> torch.Tensor:
    n, K = T.shape
    device, dtype = T.device, T.dtype

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    T = torch.nan_to_num(T, nan=0.0, posinf=1e6, neginf=-1e6)

    mu0 = hyp.mu0.to(device=device, dtype=dtype)
    Sigma0 = hyp.Sigma0.to(device=device, dtype=dtype)
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)

    L0 = robust_cholesky(Sigma0, jitter=jitter)
    Sigma0_inv = torch.cholesky_solve(torch.eye(K, device=device, dtype=dtype), L0)
    Sigma0_inv_mu0 = torch.cholesky_solve(mu0.unsqueeze(-1), L0).squeeze(-1)
    logdet_Sigma0 = 2.0 * torch.sum(torch.log(torch.diagonal(L0)))

    Sigma_n_inv = Sigma0_inv + (T.T @ T)
    ridge = (1e-6 if dtype == torch.float64 else 1e-4)
    Sigma_n_inv = Sigma_n_inv + ridge * torch.eye(K, device=device, dtype=dtype)
    Sigma_n_inv = 0.5 * (Sigma_n_inv + Sigma_n_inv.T)

    Ln_inv = robust_cholesky(Sigma_n_inv, jitter=jitter)
    Sigma_n = torch.cholesky_solve(torch.eye(K, device=device, dtype=dtype), Ln_inv)
    logdet_Sigma_n = -2.0 * torch.sum(torch.log(torch.diagonal(Ln_inv)))

    rhs = Sigma0_inv_mu0 + (T.T @ y)
    mu_n = Sigma_n @ rhs

    a_n = hyp.a0 + 0.5 * n

    yTy = torch.dot(y, y)
    mu0_Sigma0inv_mu0 = torch.dot(mu0, Sigma0_inv_mu0)
    mu_n_Sigmaninv_mu_n = torch.dot(mu_n, Sigma_n_inv @ mu_n)

    b_n = hyp.b0 + 0.5 * (yTy + mu0_Sigma0inv_mu0 - mu_n_Sigmaninv_mu_n)
    b_n = b_n.clamp_min(1e-12)

    term1 = -0.5 * n * math.log(2.0 * math.pi)
    term2 = 0.5 * (logdet_Sigma_n - logdet_Sigma0)
    term3 = hyp.a0 * math.log(hyp.b0) - a_n * torch.log(b_n)
    term4 = torch.lgamma(torch.tensor(a_n, device=device, dtype=dtype)) - torch.lgamma(
        torch.tensor(hyp.a0, device=device, dtype=dtype)
    )
    return term1 + term2 + term3 + term4


def blr_posterior_from_design(
    y: torch.Tensor,
    T: torch.Tensor,
    hyp: BLRHyperparams,
    jitter: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    device, dtype = T.device, T.dtype
    n, K = T.shape

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    T = torch.nan_to_num(T, nan=0.0, posinf=1e6, neginf=-1e6)

    mu0 = hyp.mu0.to(device=device, dtype=dtype)
    Sigma0 = hyp.Sigma0.to(device=device, dtype=dtype)
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)

    L0 = robust_cholesky(Sigma0, jitter=jitter)
    Sigma0_inv = torch.cholesky_solve(torch.eye(K, device=device, dtype=dtype), L0)
    Sigma0_inv_mu0 = torch.cholesky_solve(mu0.unsqueeze(-1), L0).squeeze(-1)

    Sigma_n_inv = Sigma0_inv + (T.T @ T)
    ridge = (1e-6 if dtype == torch.float64 else 1e-4)
    Sigma_n_inv = Sigma_n_inv + ridge * torch.eye(K, device=device, dtype=dtype)
    Sigma_n_inv = 0.5 * (Sigma_n_inv + Sigma_n_inv.T)

    Ln_inv = robust_cholesky(Sigma_n_inv, jitter=jitter)
    Sigma_n = torch.cholesky_solve(torch.eye(K, device=device, dtype=dtype), Ln_inv)

    rhs = Sigma0_inv_mu0 + (T.T @ y)
    mu_n = Sigma_n @ rhs

    a_n = torch.tensor(hyp.a0 + 0.5 * n, device=device, dtype=dtype)

    yTy = torch.dot(y, y)
    mu0_Sigma0inv_mu0 = torch.dot(mu0, Sigma0_inv_mu0)
    mu_n_Sigmaninv_mu_n = torch.dot(mu_n, Sigma_n_inv @ mu_n)

    b_n = torch.tensor(hyp.b0, device=device, dtype=dtype) + 0.5 * (yTy + mu0_Sigma0inv_mu0 - mu_n_Sigmaninv_mu_n)
    b_n = b_n.clamp_min(torch.tensor(1e-12, device=device, dtype=dtype))

    if (a_n > 1.0).item():
        E_sigma2 = b_n / (a_n - 1.0)
    else:
        E_sigma2 = torch.tensor(float("nan"), device=device, dtype=dtype)

    return {"mu_n": mu_n, "E_sigma2": E_sigma2}


# ----------------------------
# KL helpers
# ----------------------------
def kl_bernoulli(qp: torch.Tensor, pp: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    qp = qp.clamp(eps, 1 - eps)
    pp = pp.clamp(eps, 1 - eps)
    return qp * (torch.log(qp) - torch.log(pp)) + (1 - qp) * (torch.log(1 - qp) - torch.log(1 - pp))

def kl_dirichlet(q_alpha: torch.Tensor, p_alpha: torch.Tensor) -> torch.Tensor:
    qa0 = q_alpha.sum()
    pa0 = p_alpha.sum()
    logB_q = torch.lgamma(q_alpha).sum() - torch.lgamma(qa0)
    logB_p = torch.lgamma(p_alpha).sum() - torch.lgamma(pa0)
    eq_logw = torch.digamma(q_alpha) - torch.digamma(qa0)
    return (logB_q - logB_p) + ((q_alpha - p_alpha) * eq_logw).sum()

def expected_cat_kl_to_dirichlet_prior(
    cat_probs: torch.Tensor,
    q_dir_alpha: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pi = cat_probs.clamp(eps, 1.0)
    pi = pi / pi.sum(dim=-1, keepdim=True)
    eq_logw = torch.digamma(q_dir_alpha) - torch.digamma(q_dir_alpha.sum())
    return (pi * torch.log(pi)).sum(dim=-1) - (pi * eq_logw).sum(dim=-1)


# ----------------------------
# Main VaSST model
# ----------------------------
class VaSST(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_trees: int,
        depth: int,
        operators: Optional[List[OperatorSpec]] = None,
        alpha_split: float = 0.95,
        delta0: float = 2.0,
        eta_op_prior: float = 1.0,
        eta_ft_prior: float = 1.0,
        blr_a0: float = 2.0,
        blr_b0: float = 2.0,
        blr_mu0: Optional[torch.Tensor] = None,
        blr_sigma0_scale: float = 1.0,
        tau_e: float = 1.0,
        tau_op: float = 1.0,
        tau_ft: float = 1.0,
        value_clip: float = 1e3,
        use_tanh_clip: bool = True,
        logits_clip: float = 10.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.p = int(n_features)
        self.K = int(n_trees)
        self.D = int(depth)
        self.ops = operators if operators is not None else default_operator_set()
        self.M = len(self.ops)

        self.alpha_split = float(alpha_split)
        self.delta0 = float(delta0)
        self.value_clip = float(value_clip)
        self.use_tanh_clip = bool(use_tanh_clip)
        self.logits_clip = float(logits_clip)

        self.N = num_nodes_full_binary(self.D)
        self.register_buffer("node_depth", node_depths(self.D, device=device))
        self.register_buffer("leaf_mask", (self.node_depth == self.D))  # [N] bool

        self.e_logits = nn.Parameter(torch.zeros(self.K, self.N, device=device, dtype=dtype))
        self.op_logits = nn.Parameter(torch.zeros(self.K, self.N, self.M, device=device, dtype=dtype))
        self.ft_logits = nn.Parameter(torch.zeros(self.K, self.N, self.p, device=device, dtype=dtype))

        self.op_dir_unconstrained = nn.Parameter(torch.zeros(self.M, device=device, dtype=dtype))
        self.ft_dir_unconstrained = nn.Parameter(torch.zeros(self.p, device=device, dtype=dtype))

        self.register_buffer("eta_op_prior", torch.full((self.M,), float(eta_op_prior), device=device, dtype=dtype))
        self.register_buffer("eta_ft_prior", torch.full((self.p,), float(eta_ft_prior), device=device, dtype=dtype))

        if blr_mu0 is None:
            mu0 = torch.zeros(self.K, device=device, dtype=dtype)
        else:
            mu0 = blr_mu0.to(device=device, dtype=dtype)
        Sigma0 = blr_sigma0_scale * torch.eye(self.K, device=device, dtype=dtype)
        self.blr_hyp = BLRHyperparams(mu0=mu0, Sigma0=Sigma0, a0=float(blr_a0), b0=float(blr_b0))

        self.tau_e = float(tau_e)
        self.tau_op = float(tau_op)
        self.tau_ft = float(tau_ft)

        # init (broadcast-safe)
        with torch.no_grad():
            self.op_logits.add_(0.01 * torch.randn_like(self.op_logits))
            self.ft_logits.add_(0.01 * torch.randn_like(self.ft_logits))

            d = self.node_depth.float().to(self.e_logits.device)      # [N]
            bias = (self.D - d) / max(self.D, 1)                      # [N]
            self.e_logits.add_(1.5 * bias.unsqueeze(0))               # [K,N]
            self.e_logits.masked_fill_(self.leaf_mask.unsqueeze(0).expand(self.K, -1), -10.0)

    def split_prior_probs(self) -> torch.Tensor:
        m = self.node_depth.to(dtype=torch.float32)
        p_m = self.alpha_split * (1.0 + m) ** (-self.delta0)
        p_m = torch.where(self.leaf_mask, torch.zeros_like(p_m), p_m)
        return p_m.clamp(1e-6, 1 - 1e-6)

    @staticmethod
    def sample_binary_concrete(logits: torch.Tensor, tau: float, eps: float = 1e-6) -> torch.Tensor:
        u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        g = torch.log(u) - torch.log1p(-u)
        z = (logits + g) / max(float(tau), 1e-4)
        return torch.sigmoid(z)

    @staticmethod
    def sample_gumbel_softmax(logits: torch.Tensor, tau: float, eps: float = 1e-6) -> torch.Tensor:
        u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        g = -torch.log(-torch.log(u))
        z = (logits + g) / max(float(tau), 1e-4)
        return F.softmax(z, dim=-1)

    def stabilize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.use_tanh_clip:
            c = self.value_clip
            return c * torch.tanh(x / c)
        return x.clamp(-self.value_clip, self.value_clip)

    def evaluate_soft_trees(self, X, e_soft, op_soft, ft_soft) -> torch.Tensor:
        n, p = X.shape
        assert p == self.p
        term_all = torch.matmul(ft_soft, X.T)  # [K,N,n]
        vals: List[Optional[torch.Tensor]] = [None] * self.N

        for idx in range(self.N - 1, -1, -1):
            if (2 * idx + 2) < self.N:
                left = vals[2 * idx + 1]
                right = vals[2 * idx + 2]
                assert left is not None and right is not None

                outs = []
                for op in self.ops:
                    outs.append(op.fn(left, None) if op.arity == 1 else op.fn(left, right))
                op_out = torch.stack(outs, dim=1)  # [K,M,n]
                op_mix = torch.sum(op_soft[:, idx, :].unsqueeze(-1) * op_out, dim=1)
            else:
                op_mix = torch.zeros(self.K, n, device=X.device, dtype=X.dtype)

            e = e_soft[:, idx].unsqueeze(-1)
            term = term_all[:, idx, :]
            vals[idx] = self.stabilize((1.0 - e) * term + e * op_mix)

        root = vals[0]
        assert root is not None
        return root.T.contiguous()

    def analytic_kl(self) -> torch.Tensor:
        device = self.e_logits.device
        dtype = self.e_logits.dtype
        eps = 1e-8

        # Variational Dirichlet params (positive)
        q_eta_op = F.softplus(self.op_dir_unconstrained) + 1e-4
        q_eta_ft = F.softplus(self.ft_dir_unconstrained) + 1e-4

        kl = torch.zeros((), device=device, dtype=dtype)

        # KL for Dirichlet parameters
        kl = kl + kl_dirichlet(q_eta_op, self.eta_op_prior)
        kl = kl + kl_dirichlet(q_eta_ft, self.eta_ft_prior)

        # Bernoulli splits
        q_e = torch.sigmoid(self.e_logits).clamp(eps, 1 - eps)

        # enforce leaves as terminals
        leaf = self.leaf_mask.unsqueeze(0).to(device=device)
        q_e = torch.where(leaf, torch.zeros_like(q_e), q_e)

        p_e = self.split_prior_probs().to(device=device, dtype=dtype)
        kl = kl + kl_bernoulli(q_e, p_e.unsqueeze(0)).sum()

        # Operator categorical distributions
        q_op = F.softmax(self.op_logits, dim=-1).clamp(eps, 1.0)  # [K,N,M]
        q_ft = F.softmax(self.ft_logits, dim=-1).clamp(eps, 1.0)  # [K,N,p]

        # Expected categorical KL to variational Dirichlet prior
        ek_op = expected_cat_kl_to_dirichlet_prior(q_op, q_eta_op)  # [K,N]
        kl = kl + (q_e * ek_op).sum()

        ek_ft = expected_cat_kl_to_dirichlet_prior(q_ft, q_eta_ft)  # [K,N]
        kl = kl + ((1.0 - q_e) * ek_ft).sum()

        return kl

    def elbo_mc(self, X, y, n_mc=2, include_intercept=True, jitter=1e-3):
        device, dtype = X.device, X.dtype
        n = X.shape[0]

        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        kl = self.analytic_kl()

        e_logits = self.e_logits.clamp(-self.logits_clip, self.logits_clip)
        op_logits = self.op_logits.clamp(-self.logits_clip, self.logits_clip)
        ft_logits = self.ft_logits.clamp(-self.logits_clip, self.logits_clip)

        logm_sum = torch.zeros((), device=device, dtype=dtype)
        for _ in range(int(n_mc)):
            e_soft = self.sample_binary_concrete(e_logits, self.tau_e)
            e_soft = torch.where(self.leaf_mask.unsqueeze(0), torch.zeros_like(e_soft), e_soft)

            op_soft = self.sample_gumbel_softmax(op_logits, self.tau_op)
            ft_soft = self.sample_gumbel_softmax(ft_logits, self.tau_ft)

            T = self.stabilize(self.evaluate_soft_trees(X, e_soft, op_soft, ft_soft))

            if include_intercept:
                ones = torch.ones(n, 1, device=device, dtype=dtype)
                T_aug = torch.cat([ones, T], dim=1)
                K_aug = self.K + 1
                mu0_aug = torch.zeros(K_aug, device=device, dtype=dtype)
                sigma0_diag = float(self.blr_hyp.Sigma0[0, 0].detach().cpu())
                Sigma0_aug = torch.eye(K_aug, device=device, dtype=dtype) * sigma0_diag
                hyp_aug = BLRHyperparams(mu0=mu0_aug, Sigma0=Sigma0_aug, a0=self.blr_hyp.a0, b0=self.blr_hyp.b0)
                logm = log_marginal_likelihood_blr(y, T_aug, hyp_aug, jitter=jitter)
            else:
                logm = log_marginal_likelihood_blr(y, T, self.blr_hyp, jitter=jitter)

            logm_sum = logm_sum + logm

        exp_logm = logm_sum / float(n_mc)
        elbo = exp_logm - kl
        diag = {"elbo": float(elbo.detach().cpu()),
                "E_log_marginal": float(exp_logm.detach().cpu()),
                "KL": float(kl.detach().cpu()),
                "tau": float(self.tau_e)}
        return elbo, diag


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-4
    n_steps: int = 800
    mc_samples: int = 100
    grad_clip: float = 5.0
    tau_start: float = 1.0
    tau_end: float = 0.2
    tau_anneal_steps: int = 600
    jitter: float = 1e-3
    log_every: int = 100
    kl_warmup_steps: int = 300
    kl_start: float = 0.0
    kl_end: float = 1.0


def linear_anneal(step: int, start: float, end: float, n: int) -> float:
    if n <= 0:
        return end
    t = min(max(step / n, 0.0), 1.0)
    return start + t * (end - start)


def train_VaSST(
    model,
    X,
    y,
    cfg: TrainConfig,
    include_intercept: bool = True,
    verbose: bool = True,
    print_every: int = 50,
    use_progress_bar: bool = True,
):
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.99),
        eps=1e-4,
        weight_decay=0.0
    )

    logs: List[Dict[str, float]] = []

    iterator = range(1, cfg.n_steps + 1)
    if use_progress_bar:
        iterator = tqdm(iterator, desc="VaSST", leave=True)

    for step in iterator:

        # Temperature annealing
        tau = linear_anneal(step, cfg.tau_start, cfg.tau_end, cfg.tau_anneal_steps)
        model.tau_e = tau
        model.tau_op = tau
        model.tau_ft = tau

        klw = linear_anneal(step, cfg.kl_start, cfg.kl_end, cfg.kl_warmup_steps)

        opt.zero_grad(set_to_none=True)

        elbo, diag = model.elbo_mc(
            X, y,
            n_mc=cfg.mc_samples,
            include_intercept=include_intercept,
            jitter=cfg.jitter
        )

        kl = model.analytic_kl()

        objective = elbo + (1.0 - klw) * kl
        loss = -objective

        # Skip non-finite loss
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # Guard against bad gradients
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                bad_grad = True
                break

        if bad_grad:
            opt.zero_grad(set_to_none=True)
            continue

        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt.step()

        # Logging
        if step % print_every == 0 or step == 1 or step == cfg.n_steps:

            out = dict(diag)
            out["step"] = step
            out["loss"] = float(loss.detach().cpu())
            out["klw"] = float(klw)
            logs.append(out)

            if verbose:
                print(
                    f"[{step:5d}] "
                    f"loss={out['loss']:.3f} "
                    f"ELBO={out['elbo']:.3f} "
                    f"Elogm={out['E_log_marginal']:.3f} "
                    f"KL={out['KL']:.3f} "
                    f"klw={out['klw']:.2f} "
                    f"tau={out['tau']:.3f}"
                )

        # Update progress bar display
        if use_progress_bar:
            iterator.set_postfix({
                "ELBO": f"{float(elbo.detach().cpu()):.2f}",
                "KL": f"{float(kl.detach().cpu()):.2f}",
                "tau": f"{tau:.2f}"
            })

    return logs


# ----------------------------
# Hard sampling + hard eval
# ----------------------------
def sample_hard_tree(model: VaSST, k: int) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        q_e = torch.sigmoid(model.e_logits[k])
        q_e = torch.where(model.leaf_mask, torch.zeros_like(q_e), q_e)
        e_hat = torch.bernoulli(q_e).long()

        op_probs = torch.softmax(model.op_logits[k], dim=-1)
        ft_probs = torch.softmax(model.ft_logits[k], dim=-1)
        op_hat = torch.multinomial(op_probs, num_samples=1).squeeze(-1).long()
        ft_hat = torch.multinomial(ft_probs, num_samples=1).squeeze(-1).long()

        e_hat = prune_by_expand(e_hat.float()).long()
    return {"e_hat": e_hat, "op_hat": op_hat, "ft_hat": ft_hat}


def evaluate_hard_trees(model: VaSST, X: torch.Tensor, e_hat: torch.Tensor, op_hat: torch.Tensor, ft_hat: torch.Tensor) -> torch.Tensor:
    n, p = X.shape
    K, N = e_hat.shape
    assert K == model.K and N == model.N and p == model.p

    vals: List[Optional[torch.Tensor]] = [None] * N
    for idx in range(N - 1, -1, -1):
        feat_idx = ft_hat[:, idx]
        term = torch.stack([X[:, int(feat_idx[k].item())] for k in range(K)], dim=0)

        if is_leaf(idx, N):
            vals[idx] = model.stabilize(term)
            continue

        l, r = heap_children(idx)
        left = vals[l]
        right = vals[r]
        assert left is not None and right is not None

        op_res_list = []
        for k in range(K):
            op = model.ops[int(op_hat[k, idx].item())]
            op_res_list.append(op.fn(left[k], None) if op.arity == 1 else op.fn(left[k], right[k]))
        op_res = torch.stack(op_res_list, dim=0)

        e = e_hat[:, idx].to(dtype=X.dtype).unsqueeze(-1)
        vals[idx] = model.stabilize((1.0 - e) * term + e * op_res)

    root = vals[0]
    assert root is not None
    return root.T.contiguous()


def sample_hard_models_and_make_tables_wide(
    model: VaSST,
    X: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 20,
    include_intercept: bool = True,
    jitter: float = 1e-3,
    standardize_xy: bool = False,
):
    device, dtype = X.device, X.dtype

    X_in = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y_in = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    if standardize_xy:
        with torch.no_grad():
            Xm = X_in.mean(dim=0, keepdim=True)
            Xs = X_in.std(dim=0, keepdim=True).clamp_min(1e-6)
            X_use = (X_in - Xm) / Xs

            ym = y_in.mean()
            ys = y_in.std().clamp_min(1e-6)
            y_use = (y_in - ym) / ys
    else:
        X_use, y_use = X_in, y_in

    expr_wide_rows: List[Dict[str, Any]] = []
    est_wide_rows: List[Dict[str, Any]] = []

    for s in range(int(n_samples)):
        expr_row: Dict[str, Any] = {"sample_id": s}
        e_hat_list, op_hat_list, ft_hat_list = [], [], []

        for k in range(model.K):
            smp = sample_hard_tree(model, k=k)
            e_hat_list.append(smp["e_hat"])
            op_hat_list.append(smp["op_hat"])
            ft_hat_list.append(smp["ft_hat"])

            expr_row[f"tree_{k}"] = tree_to_expression(
                smp["e_hat"], smp["op_hat"], smp["ft_hat"],
                operators=model.ops,
                feature_names=[f"x{j}" for j in range(model.p)],
            )

        expr_wide_rows.append(expr_row)

        # --- compute posterior beta_mean, sigma2_mean for this sampled set ---
        e_hat = torch.stack(e_hat_list, dim=0).to(device=device)
        op_hat = torch.stack(op_hat_list, dim=0).to(device=device)
        ft_hat = torch.stack(ft_hat_list, dim=0).to(device=device)

        e_hat = torch.where(model.leaf_mask.unsqueeze(0), torch.zeros_like(e_hat), e_hat)

        T = evaluate_hard_trees(model, X_use, e_hat, op_hat, ft_hat)  # [n,K]

        if include_intercept:
            ones = torch.ones(T.shape[0], 1, device=device, dtype=dtype)
            T_aug = torch.cat([ones, T], dim=1)  # [n,K+1]
            K_aug = model.K + 1

            mu0_aug = torch.zeros(K_aug, device=device, dtype=dtype)
            sigma0_diag = float(model.blr_hyp.Sigma0[0, 0].detach().cpu())
            Sigma0_aug = torch.eye(K_aug, device=device, dtype=dtype) * sigma0_diag
            hyp = BLRHyperparams(mu0=mu0_aug, Sigma0=Sigma0_aug, a0=model.blr_hyp.a0, b0=model.blr_hyp.b0)

            post = blr_posterior_from_design(y_use, T_aug, hyp, jitter=jitter)
            beta_mean = post["mu_n"].detach().cpu()
            sigma2_mean = float(post["E_sigma2"].detach().cpu())

            coef_names = ["intercept"] + [f"tree_{k}" for k in range(model.K)]
        else:
            post = blr_posterior_from_design(y_use, T, model.blr_hyp, jitter=jitter)
            beta_mean = post["mu_n"].detach().cpu()
            sigma2_mean = float(post["E_sigma2"].detach().cpu())
            coef_names = [f"tree_{k}" for k in range(model.K)]

        est_row: Dict[str, Any] = {"sample_id": s}
        for j, nm in enumerate(coef_names):
            est_row[nm] = float(beta_mean[j].item())
        est_row["sigma2_mean"] = sigma2_mean
        est_wide_rows.append(est_row)

    if pd is not None:
        return pd.DataFrame(expr_wide_rows), pd.DataFrame(est_wide_rows)
    return expr_wide_rows, est_wide_rows

def rank_hard_tree_samples_by_rmse(
    model: "VaSST",
    X: torch.Tensor,                # [n,p]
    y: torch.Tensor,                # [n]
    n_samples: int = 500,
    include_intercept: bool = True,
    jitter: float = 1e-3,
    standardize_xy: bool = False,
    top_k: int = 10,
    feature_names: Optional[List[str]] = None,
) -> Tuple["pd.DataFrame|list", "pd.DataFrame|list"]:

    device, dtype = X.device, X.dtype
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(model.p)]

    X_in = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y_in = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    if standardize_xy:
        with torch.no_grad():
            Xm = X_in.mean(dim=0, keepdim=True)
            Xs = X_in.std(dim=0, keepdim=True).clamp_min(1e-6)
            X_use = (X_in - Xm) / Xs

            ym = y_in.mean()
            ys = y_in.std().clamp_min(1e-6)
            y_use = (y_in - ym) / ys
    else:
        X_use, y_use = X_in, y_in

    rows_expr: List[Dict[str, Any]] = []
    rows_beta: List[Dict[str, Any]] = []
    rmse_list: List[float] = []

    for s in range(int(n_samples)):
        # ---- sample one hard structure for each tree k ----
        e_hat_list, op_hat_list, ft_hat_list = [], [], []

        expr_row: Dict[str, Any] = {"sample_id": s}
        for k in range(model.K):
            smp = sample_hard_tree(model, k=k)
            e_hat_list.append(smp["e_hat"])
            op_hat_list.append(smp["op_hat"])
            ft_hat_list.append(smp["ft_hat"])

            expr_row[f"tree_{k}"] = tree_to_expression(
                smp["e_hat"],
                smp["op_hat"],
                smp["ft_hat"],
                operators=model.ops,
                feature_names=feature_names,
            )

        # stack to [K,N]
        e_hat = torch.stack(e_hat_list, dim=0).to(device=device)
        op_hat = torch.stack(op_hat_list, dim=0).to(device=device)
        ft_hat = torch.stack(ft_hat_list, dim=0).to(device=device)

        # enforce leaves are terminal (broadcast-safe)
        e_hat = torch.where(model.leaf_mask.unsqueeze(0), torch.zeros_like(e_hat), e_hat)

        # ---- build design matrix ----
        T = evaluate_hard_trees(model, X_use, e_hat, op_hat, ft_hat)  # [n,K]

        # ---- posterior ----
        if include_intercept:
            ones = torch.ones(T.shape[0], 1, device=device, dtype=dtype)
            T_aug = torch.cat([ones, T], dim=1)  # [n,K+1]
            K_aug = model.K + 1

            mu0_aug = torch.zeros(K_aug, device=device, dtype=dtype)
            sigma0_diag = float(model.blr_hyp.Sigma0[0, 0].detach().cpu())
            Sigma0_aug = torch.eye(K_aug, device=device, dtype=dtype) * sigma0_diag
            hyp = BLRHyperparams(mu0=mu0_aug, Sigma0=Sigma0_aug, a0=model.blr_hyp.a0, b0=model.blr_hyp.b0)

            post = blr_posterior_from_design(y_use, T_aug, hyp, jitter=jitter)
            beta_mean = post["mu_n"]               # [K+1]
            sigma2_mean = float(post["E_sigma2"].detach().cpu())

            y_hat = T_aug @ beta_mean
            coef_names = ["intercept"] + [f"tree_{k}" for k in range(model.K)]
        else:
            post = blr_posterior_from_design(y_use, T, model.blr_hyp, jitter=jitter)
            beta_mean = post["mu_n"]               # [K]
            sigma2_mean = float(post["E_sigma2"].detach().cpu())

            y_hat = T @ beta_mean
            coef_names = [f"tree_{k}" for k in range(model.K)]

        # ---- RMSE ----
        resid = (y_use - y_hat)
        rmse = float(torch.sqrt(torch.mean(resid * resid)).detach().cpu())
        rmse_list.append(rmse)

        # store rows
        expr_row["rmse"] = rmse
        expr_row["sigma2_mean"] = sigma2_mean
        rows_expr.append(expr_row)

        beta_row: Dict[str, Any] = {"sample_id": s, "rmse": rmse, "sigma2_mean": sigma2_mean}
        beta_mean_cpu = beta_mean.detach().cpu()
        for j, nm in enumerate(coef_names):
            beta_row[nm] = float(beta_mean_cpu[j].item())
        rows_beta.append(beta_row)

    df_expr = pd.DataFrame(rows_expr)
    df_beta = pd.DataFrame(rows_beta)

    # ---- rank by RMSE (ascending) ----
    df_expr_sorted = df_expr.sort_values("rmse", ascending=True).reset_index(drop=True)
    df_beta_sorted = df_beta.sort_values("rmse", ascending=True).reset_index(drop=True)

    # keep consistent top_k sample_ids
    top_ids = df_expr_sorted["sample_id"].head(top_k).tolist()

    expr_top = df_expr_sorted[df_expr_sorted["sample_id"].isin(top_ids)].copy()
    beta_top = df_beta_sorted[df_beta_sorted["sample_id"].isin(top_ids)].copy()

    # re-sort to match RMSE order
    expr_top = expr_top.sort_values("rmse", ascending=True).reset_index(drop=True)
    beta_top = beta_top.sort_values("rmse", ascending=True).reset_index(drop=True)

    # add rank column (1..top_k)
    expr_top.insert(0, "rank", range(1, len(expr_top) + 1))
    beta_top.insert(0, "rank", range(1, len(beta_top) + 1))

    # make tables "wide" and readable: keep meta first, then trees/coeffs
    tree_cols = [f"tree_{k}" for k in range(model.K)]
    expr_cols = ["rank", "sample_id", "rmse", "sigma2_mean"] + tree_cols
    expr_top = expr_top[expr_cols]

    coef_cols = (["intercept"] if include_intercept else []) + [f"tree_{k}" for k in range(model.K)]
    beta_cols = ["rank", "sample_id", "rmse", "sigma2_mean"] + coef_cols
    beta_top = beta_top[beta_cols]

    # print
    print("\n=== TOP samples by RMSE (expressions) ===")
    print(expr_top.to_string(index=False))

    print("\n=== TOP samples by RMSE (beta means + sigma2 mean) ===")
    print(beta_top.to_string(index=False))

    return expr_top, beta_top

##############################################################################################################

# ============================================================
# Posterior-aware ranking of hard VaSST forests
# Ranking by:
#   LMPSE(T) = log p(y | T) + log pi(T)
# Also returns ranking by log p(y | T) alone.
# ============================================================

def _log_dirichlet_integrated_categorical_counts(
    counts: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    """
    Computes

        log int prod_m w_m^{counts_m} Dir(w; eta) dw

    which equals

        log B(eta + counts) - log B(eta).

    This is the integrated categorical likelihood under a Dirichlet prior.
    """
    counts = counts.to(device=eta.device, dtype=eta.dtype)

    eta0 = eta.sum()
    counts0 = counts.sum()

    return (
        torch.lgamma(eta0)
        - torch.lgamma(eta0 + counts0)
        + torch.lgamma(eta + counts).sum()
        - torch.lgamma(eta).sum()
    )


def _forest_counts_and_split_logprior(
    model: "VaSST",
    e_hat: torch.Tensor,   # [K, N]
    op_hat: torch.Tensor,  # [K, N]
    ft_hat: torch.Tensor,  # [K, N]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes:
      1. log split prior under depth-dependent Bernoulli splitting,
      2. operator counts over expanded internal nodes,
      3. feature counts over terminal nodes.

    Important:
    - Only reachable nodes are counted.
    - If a node is not expanded, its descendants are ignored.
    - If an internal node is not expanded, it becomes a terminal feature node.
    - Leaves are always terminal feature nodes.
    """

    device = e_hat.device
    dtype = model.e_logits.dtype

    K, N = e_hat.shape

    split_probs = model.split_prior_probs().to(device=device, dtype=dtype)

    op_counts = torch.zeros(model.M, device=device, dtype=dtype)
    ft_counts = torch.zeros(model.p, device=device, dtype=dtype)

    log_split_prior = torch.zeros((), device=device, dtype=dtype)

    def traverse_tree(k: int, i: int):
        nonlocal log_split_prior, op_counts, ft_counts

        if i >= N:
            return

        # Leaves are terminal with probability 1 under the skeleton.
        if is_leaf(i, N):
            fidx = int(ft_hat[k, i].item())
            ft_counts[fidx] += 1.0
            return

        expanded = int(e_hat[k, i].item()) == 1

        p_split_i = split_probs[i].clamp(1e-8, 1.0 - 1e-8)

        if expanded:
            # Split prior contribution
            log_split_prior = log_split_prior + torch.log(p_split_i)

            # Operator choice is only active if node is expanded
            oidx = int(op_hat[k, i].item())
            op_counts[oidx] += 1.0

            l, r = heap_children(i)
            traverse_tree(k, l)
            traverse_tree(k, r)

        else:
            # Stop prior contribution
            log_split_prior = log_split_prior + torch.log1p(-p_split_i)

            # Terminal feature choice
            fidx = int(ft_hat[k, i].item())
            ft_counts[fidx] += 1.0

            # Descendants ignored because this node is terminal.
            return

    for k in range(K):
        traverse_tree(k, 0)

    return log_split_prior, op_counts, ft_counts


def forest_log_prior_integrated(
    model: "VaSST",
    e_hat: torch.Tensor,
    op_hat: torch.Tensor,
    ft_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the integrated forest prior:

        log pi(T)
        =
        log pi_split(e)
        +
        log int prod active_op_nodes w_op[o_i] Dir(w_op; eta_op) dw_op
        +
        log int prod terminal_nodes w_ft[f_i] Dir(w_ft; eta_ft) dw_ft.

    This corresponds to integrating out the global operator and feature
    probability vectors.
    """

    device = e_hat.device
    dtype = model.e_logits.dtype

    log_split_prior, op_counts, ft_counts = _forest_counts_and_split_logprior(
        model=model,
        e_hat=e_hat,
        op_hat=op_hat,
        ft_hat=ft_hat,
    )

    eta_op = model.eta_op_prior.to(device=device, dtype=dtype)
    eta_ft = model.eta_ft_prior.to(device=device, dtype=dtype)

    log_op_prior = _log_dirichlet_integrated_categorical_counts(
        counts=op_counts,
        eta=eta_op,
    )

    log_ft_prior = _log_dirichlet_integrated_categorical_counts(
        counts=ft_counts,
        eta=eta_ft,
    )

    return log_split_prior + log_op_prior + log_ft_prior


def rank_hard_tree_samples_by_lmpse(
    model: "VaSST",
    X: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 500,
    include_intercept: bool = True,
    jitter: float = 1e-3,
    standardize_xy: bool = False,
    top_k: int = 10,
    feature_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Samples hard VaSST forests and ranks them using two criteria:

    1. Joint marginal posterior score:

           LMPSE(T) = log p(y | T) + log pi(T)

    2. Marginal likelihood alone:

           log p(y | T)

    Returns:
      - expr_top_lmpse:  top forests by LMPSE
      - beta_top_lmpse:  beta posterior means for top LMPSE forests
      - expr_top_logm: top forests by log p(y | T)
      - beta_top_logm: beta posterior means for top log p(y | T) forests
    """

    if pd is None:
        raise RuntimeError("pandas is required for posterior-aware ranking.")

    device, dtype = X.device, X.dtype

    if feature_names is None:
        feature_names = [f"x{j}" for j in range(model.p)]

    X_in = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y_in = torch.nan_to_num(y.reshape(-1), nan=0.0, posinf=1e6, neginf=-1e6)

    if standardize_xy:
        with torch.no_grad():
            Xm = X_in.mean(dim=0, keepdim=True)
            Xs = X_in.std(dim=0, keepdim=True).clamp_min(1e-6)
            X_use = (X_in - Xm) / Xs

            ym = y_in.mean()
            ys = y_in.std().clamp_min(1e-6)
            y_use = (y_in - ym) / ys
    else:
        X_use = X_in
        y_use = y_in

    rows_expr: List[Dict[str, Any]] = []
    rows_beta: List[Dict[str, Any]] = []

    for s in range(int(n_samples)):
        e_hat_list, op_hat_list, ft_hat_list = [], [], []

        expr_row: Dict[str, Any] = {
            "sample_id": s,
        }

        # ----------------------------------------------------
        # Sample one hard tree for each ensemble member
        # ----------------------------------------------------
        for k in range(model.K):
            smp = sample_hard_tree(model, k=k)

            e_hat_list.append(smp["e_hat"])
            op_hat_list.append(smp["op_hat"])
            ft_hat_list.append(smp["ft_hat"])

            expr_row[f"tree_{k}"] = tree_to_expression(
                smp["e_hat"],
                smp["op_hat"],
                smp["ft_hat"],
                operators=model.ops,
                feature_names=feature_names,
            )

        e_hat = torch.stack(e_hat_list, dim=0).to(device=device)
        op_hat = torch.stack(op_hat_list, dim=0).to(device=device)
        ft_hat = torch.stack(ft_hat_list, dim=0).to(device=device)

        # Enforce leaves as terminal.
        e_hat = torch.where(
            model.leaf_mask.unsqueeze(0),
            torch.zeros_like(e_hat),
            e_hat,
        )

        # ----------------------------------------------------
        # Build hard-tree design matrix
        # ----------------------------------------------------
        T = evaluate_hard_trees(
            model=model,
            X=X_use,
            e_hat=e_hat,
            op_hat=op_hat,
            ft_hat=ft_hat,
        )

        T = torch.nan_to_num(T, nan=0.0, posinf=1e6, neginf=-1e6)

        # ----------------------------------------------------
        # Compute log p(y | T), posterior beta mean, sigma2 mean
        # ----------------------------------------------------
        if include_intercept:
            ones = torch.ones(T.shape[0], 1, device=device, dtype=dtype)
            T_aug = torch.cat([ones, T], dim=1)

            K_aug = model.K + 1

            mu0_aug = torch.zeros(K_aug, device=device, dtype=dtype)

            sigma0_diag = float(model.blr_hyp.Sigma0[0, 0].detach().cpu())
            Sigma0_aug = torch.eye(K_aug, device=device, dtype=dtype) * sigma0_diag

            hyp = BLRHyperparams(
                mu0=mu0_aug,
                Sigma0=Sigma0_aug,
                a0=model.blr_hyp.a0,
                b0=model.blr_hyp.b0,
            )

            log_marginal = log_marginal_likelihood_blr(
                y=y_use,
                T=T_aug,
                hyp=hyp,
                jitter=jitter,
            )

            post = blr_posterior_from_design(
                y=y_use,
                T=T_aug,
                hyp=hyp,
                jitter=jitter,
            )

            beta_mean = post["mu_n"]
            sigma2_mean = float(post["E_sigma2"].detach().cpu())

            y_hat = T_aug @ beta_mean
            coef_names = ["intercept"] + [f"tree_{k}" for k in range(model.K)]

        else:
            log_marginal = log_marginal_likelihood_blr(
                y=y_use,
                T=T,
                hyp=model.blr_hyp,
                jitter=jitter,
            )

            post = blr_posterior_from_design(
                y=y_use,
                T=T,
                hyp=model.blr_hyp,
                jitter=jitter,
            )

            beta_mean = post["mu_n"]
            sigma2_mean = float(post["E_sigma2"].detach().cpu())

            y_hat = T @ beta_mean
            coef_names = [f"tree_{k}" for k in range(model.K)]

        # ----------------------------------------------------
        # Compute integrated tree prior log pi(T)
        # ----------------------------------------------------
        log_tree_prior = forest_log_prior_integrated(
            model=model,
            e_hat=e_hat,
            op_hat=op_hat,
            ft_hat=ft_hat,
        )

        lmpse = log_marginal + log_tree_prior

        # ----------------------------------------------------
        # RMSE against observed y
        # ----------------------------------------------------
        rmse = float(
            torch.sqrt(torch.mean((y_use - y_hat) ** 2)).detach().cpu()
        )

        log_marginal_float = float(log_marginal.detach().cpu())
        log_tree_prior_float = float(log_tree_prior.detach().cpu())
        lmpse_float = float(lmpse.detach().cpu())

        # ----------------------------------------------------
        # Store expression row
        # ----------------------------------------------------
        expr_row["rmse"] = rmse
        expr_row["sigma2_mean"] = sigma2_mean
        expr_row["log_marginal"] = log_marginal_float
        expr_row["log_tree_prior"] = log_tree_prior_float
        expr_row["lmpse"] = lmpse_float

        rows_expr.append(expr_row)

        # ----------------------------------------------------
        # Store beta row
        # ----------------------------------------------------
        beta_row: Dict[str, Any] = {
            "sample_id": s,
            "rmse": rmse,
            "sigma2_mean": sigma2_mean,
            "log_marginal": log_marginal_float,
            "log_tree_prior": log_tree_prior_float,
            "lmpse": lmpse_float,
        }

        beta_mean_cpu = beta_mean.detach().cpu()

        for j, nm in enumerate(coef_names):
            beta_row[nm] = float(beta_mean_cpu[j].item())

        rows_beta.append(beta_row)

    df_expr = pd.DataFrame(rows_expr)
    df_beta = pd.DataFrame(rows_beta)

    tree_cols = [f"tree_{k}" for k in range(model.K)]
    coef_cols = (["intercept"] if include_intercept else []) + tree_cols

    expr_cols = [
        "rank",
        "sample_id",
        "rmse",
        "sigma2_mean",
        "log_marginal",
        "log_tree_prior",
        "lmpse",
    ] + tree_cols

    beta_cols = [
        "rank",
        "sample_id",
        "rmse",
        "sigma2_mean",
        "log_marginal",
        "log_tree_prior",
        "lmpse",
    ] + coef_cols

    # ========================================================
    # Top forests by LMPSE
    # ========================================================
    df_expr_lmpse_sorted = df_expr.sort_values(
        "lmpse",
        ascending=False,
    ).reset_index(drop=True)

    top_ids_lmpse = df_expr_lmpse_sorted["sample_id"].head(top_k).tolist()

    expr_top_lmpse = df_expr[
        df_expr["sample_id"].isin(top_ids_lmpse)
    ].copy()

    beta_top_lmpse = df_beta[
        df_beta["sample_id"].isin(top_ids_lmpse)
    ].copy()

    expr_top_lmpse = expr_top_lmpse.sort_values(
        "lmpse",
        ascending=False,
    ).reset_index(drop=True)

    beta_top_lmpse = beta_top_lmpse.sort_values(
        "lmpse",
        ascending=False,
    ).reset_index(drop=True)

    expr_top_lmpse.insert(0, "rank", range(1, len(expr_top_lmpse) + 1))
    beta_top_lmpse.insert(0, "rank", range(1, len(beta_top_lmpse) + 1))

    expr_top_lmpse = expr_top_lmpse[expr_cols]
    beta_top_lmpse = beta_top_lmpse[beta_cols]

    # ========================================================
    # Top forests by log p(y | T)
    # ========================================================
    df_expr_logm_sorted = df_expr.sort_values(
        "log_marginal",
        ascending=False,
    ).reset_index(drop=True)

    top_ids_logm = df_expr_logm_sorted["sample_id"].head(top_k).tolist()

    expr_top_logm = df_expr[
        df_expr["sample_id"].isin(top_ids_logm)
    ].copy()

    beta_top_logm = df_beta[
        df_beta["sample_id"].isin(top_ids_logm)
    ].copy()

    expr_top_logm = expr_top_logm.sort_values(
        "log_marginal",
        ascending=False,
    ).reset_index(drop=True)

    beta_top_logm = beta_top_logm.sort_values(
        "log_marginal",
        ascending=False,
    ).reset_index(drop=True)

    expr_top_logm.insert(0, "rank", range(1, len(expr_top_logm) + 1))
    beta_top_logm.insert(0, "rank", range(1, len(beta_top_logm) + 1))

    expr_top_logm = expr_top_logm[expr_cols]
    beta_top_logm = beta_top_logm[beta_cols]

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print("\n=== TOP samples by LMPSE ===")
    print(expr_top_lmpse.to_string(index=False))

    print("\n=== TOP beta means by LMPSE ===")
    print(beta_top_lmpse.to_string(index=False))

    print("\n=== TOP samples by log marginal likelihood log p(y | T) ===")
    print(expr_top_logm.to_string(index=False))

    print("\n=== TOP beta means by log marginal likelihood ===")
    print(beta_top_logm.to_string(index=False))

    return expr_top_lmpse, beta_top_lmpse, expr_top_logm, beta_top_logm