# Copyright (c) Meta Platforms, Inc. and affiliates.

"""textGrain (Li et al., "textGrain: Entropy-Calibrated Watermarking for Language Model Text", OpenAI report, 2026),
implemented from the report: Algorithm 1 (keyed partition + column), Algorithm A.1 (Sinkhorn with entropy calibration).

The report gives no code and no hyperparameters; the choices (B = m = 256 blocks/columns in the paper, context h = 3,
tolerance, iteration limit, keyed PRF) are in TextGrainConfig. The keyed randomness is TextSeal's 13-bit PRF
(textseal.watermarking.core.prf_uniform), with three streams derived from the key by fixed offsets.
"""

import math
from dataclasses import dataclass

import torch
from scipy import special

from textseal.watermarking.core import prf_uniform

_M = 2**13 - 1
_EULER = 0.5772156649015329
_GUMBEL_SD = math.pi / math.sqrt(6.0)
_SUB_PART, _SUB_COST, _SUB_COL = 0x1F3D5B79, 0x2C4E6A8F, 0x3B5D7F91


@dataclass
class TextGrainConfig:
    key: int = 42
    h: int = 3
    B: int = 256
    m: int = 256
    beta: float = 0.5  # Target fraction of next-token entropy removed.
    eps: float = 5e-3  # Stopping tolerance on |beta_hat - beta| (Alg. A.1, line 17).
    n_it: int = 200


def marginal_correct(X: torch.Tensor, a: torch.Tensor, b: float) -> torch.Tensor:
    """Eq. (A.2): restore row marginals a (bsz, B) and column marginals b (scalar)."""
    rs = X.sum(2)
    x = torch.where(rs > 0, torch.clamp(a / torch.where(rs > 0, rs, 1.0), max=1.0), 1.0)
    X1 = X * x[:, :, None]
    cs = X1.sum(1)
    y = torch.where(cs > 0, torch.clamp(b / torch.where(cs > 0, cs, 1.0), max=1.0), 1.0)
    X2 = X1 * y[:, None, :]
    ea = torch.clamp(a - X2.sum(2), min=0.0)
    eb = torch.clamp(b - X2.sum(1), min=0.0)
    D = ea.sum(1)
    fill = ea[:, :, None] * eb[:, None, :] / torch.where(D > 0, D, 1.0)[:, None, None]
    return X2 + torch.where((D > 0)[:, None, None], fill, torch.zeros_like(fill))


def kl_to_independent(pi: torch.Tensor, p: torch.Tensor, um: float) -> torch.Tensor:
    """KL(pi || p x u_m) in nats, per batch element."""
    ref = p[:, :, None] * um
    mask = pi > 0
    terms = torch.where(mask, pi * (torch.log(torch.where(mask, pi, 1.0)) - torch.log(torch.where(mask, ref, 1.0))), 0.0)
    return terms.sum((1, 2))


def entropy(P: torch.Tensor) -> torch.Tensor:
    return -torch.where(P > 0, P * torch.log(torch.where(P > 0, P, 1.0)), 0.0).sum(-1)


def _setup(p_blk: torch.Tensor, H_P: torch.Tensor, cfg: TextGrainConfig, m: int, dt: torch.dtype):
    """Shared preamble of Algorithm A.1 (lines 1-8): targets, trivial rows, delta."""
    p, H_P = p_blk.to(dt), H_P.to(dt)
    active = p > 0
    logp = torch.where(active, torch.log(torch.where(active, p, 1.0)), -math.inf)
    H_blk = -torch.where(active, p * logp, 0.0).sum(1)
    Hs = torch.where(H_P > 0, H_P, 1.0)
    beta_t = torch.full_like(H_P, cfg.beta)
    trivial = (cfg.beta == 0) | (H_blk <= 0) | (H_P <= 0)
    beta_t = torch.minimum(beta_t, (1 - 1e-8) * H_blk / Hs)
    beta_t = torch.minimum(beta_t, math.log(m) / Hs)
    trivial = trivial | (beta_t <= 0)
    beta_t = torch.where(trivial, torch.zeros_like(beta_t), beta_t)
    delta = 1e-12 * H_blk / Hs
    return p, active, logp, Hs, beta_t, trivial, delta


def _iteration(G, lam, logv, logp, active, p, um: float, Hs, beta_t, delta, a: float, lam_min: float, lam_max: float):
    """Algorithm A.1 lines 10-16, plus the proposed lambda update (lines 20-21)."""
    logK = G / lam[:, None, None]
    logu = logp - torch.logsumexp(logK + logv[:, None, :], dim=2)
    logu = torch.where(active, logu, -math.inf)
    logv = math.log(um) - torch.logsumexp(logK + logu[:, :, None], dim=1)
    logv = logv - logv.mean(1, keepdim=True)
    logu = logp - torch.logsumexp(logK + logv[:, None, :], dim=2)
    logu = torch.where(active, logu, -math.inf)
    logX = logu[:, :, None] + logK
    pi_s = torch.exp(logX - torch.logsumexp(logX, dim=1, keepdim=True)) * um
    pi_c = marginal_correct(pi_s, p, um)
    beta_hat = kl_to_independent(pi_c, p, um) / Hs
    ratio = torch.maximum(torch.clamp(beta_hat, min=0.0), delta) / torch.where(beta_t > 0, beta_t, 1.0)
    lam_new = torch.clamp(lam * ratio**a, lam_min, lam_max)
    return pi_c, beta_hat, logv, lam_new


def solve_coupling(p_blk: torch.Tensor, G: torch.Tensor, H_P: torch.Tensor, cfg: TextGrainConfig, a: float = 0.5,
                   lam_min: float = 1e-2, lam_max: float = 1e2, check_every: int = 8):
    """Algorithm A.1, batched, without per-iteration host syncs.

    p_blk (bsz, B) block probabilities, G (bsz, B, m) standardised Gumbel costs (c = -G), H_P (bsz,) entropy in nats.
    Returns the corrected coupling (bsz, B, m), beta_hat (bsz,), the capped target (bsz,) and a status per row
    (0 independent coupling, 1 converged, 2 iteration limit). Rows record their coupling at the iteration they converge.
    """
    bsz, _, m = G.shape
    dt = torch.float64
    um = 1.0 / m
    p, active, logp, Hs, beta_t, trivial, delta = _setup(p_blk, H_P, cfg, m, dt)
    G = G.to(dt)
    pi_out = (p[:, :, None] * um).expand(bsz, -1, m).clone()
    beta_out = torch.zeros(bsz, dtype=dt, device=G.device)
    status = torch.zeros(bsz, dtype=torch.long, device=G.device)
    if bool(trivial.all()):
        return pi_out, beta_out, beta_t, status
    lam = torch.ones(bsz, dtype=dt, device=G.device)
    logv = torch.full((bsz, m), math.log(um), dtype=dt, device=G.device)
    done = trivial.clone()
    for s in range(cfg.n_it):
        pi_c, beta_hat, logv, lam_new = _iteration(G, lam, logv, logp, active, p, um, Hs, beta_t, delta, a, lam_min, lam_max)
        conv = (beta_hat - beta_t).abs() <= cfg.eps
        new = (~done) & (conv | (s == cfg.n_it - 1))
        pi_out = torch.where(new[:, None, None], pi_c, pi_out)
        beta_out = torch.where(new, beta_hat, beta_out)
        status = torch.where(new, torch.where(conv, 1, 2), status)
        done = done | new
        if s % check_every == check_every - 1 and bool(done.all()):
            break
        logv = logv * (lam / lam_new)[:, None]
        lam = torch.where(done, lam, lam_new)
    return pi_out, beta_out, beta_t, status


def textgrain_law(probs_sort: torch.Tensor, probs_idx: torch.Tensor, ctx: torch.Tensor, cfg: TextGrainConfig,
                  solver: str = "torch") -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Watermarked law Q(. | J) on the top-p support.

    probs_sort, probs_idx (bsz, V): sorted truncated law (sampling.ntp). solver: "torch" (float64, exact Algorithm A.1)
    or "chunked32" / "chunked64" (Triton kernel, the paper's speculative runs). Returns Q (bsz, n) float64, token ids
    (bsz, n) and solver statistics.
    """
    bsz = probs_sort.shape[0]
    n = int((probs_sort > 0).sum(1).max())
    ps = probs_sort[:, :n].double()
    ps = ps / ps.sum(1, keepdim=True)
    ids = probs_idx[:, :n]
    g = torch.clamp((prf_uniform(ctx, ids, cfg.key + _SUB_PART) * cfg.B).long(), max=cfg.B - 1)
    p_blk = torch.zeros(bsz, cfg.B, dtype=torch.float64, device=ps.device).scatter_add_(1, g, ps)
    H_P = entropy(ps)
    cells = torch.arange(cfg.B * cfg.m, device=ps.device).expand(bsz, -1)
    u = (prf_uniform(ctx, cells, cfg.key + _SUB_COST) + 0.5 / _M).view(bsz, cfg.B, cfg.m)
    G = (-torch.log(-torch.log(u.double())) - _EULER) / _GUMBEL_SD
    zero = torch.zeros(bsz, 1, dtype=torch.long, device=ps.device)
    J = torch.clamp((prf_uniform(ctx, zero, cfg.key + _SUB_COL).squeeze(1) * cfg.m).long(), max=cfg.m - 1)
    if solver.startswith("chunked"):
        from textseal.watermarking.sinkhorn_triton import solve_coupling_chunked

        dt = torch.float64 if solver == "chunked64" else torch.float32
        pi, beta_hat, beta_t, status = solve_coupling_chunked(p_blk, G, H_P, cfg, dtype=dt)
    else:
        pi, beta_hat, beta_t, status = solve_coupling(p_blk, G, H_P, cfg)
    r = cfg.m * pi[torch.arange(bsz, device=ps.device), :, J]  # bsz B
    scale = torch.where(p_blk > 0, r / torch.where(p_blk > 0, p_blk, 1.0), 0.0)
    Q = ps * scale.gather(1, g)
    Q = Q / Q.sum(1, keepdim=True)
    stats = {"H_P": H_P, "loss": kl_to_independent(pi, p_blk, 1.0 / cfg.m), "beta_hat": beta_hat, "beta_target": beta_t, "status": status}
    return Q, ids, stats


def score_pairs(ctx: torch.Tensor, tok: torch.Tensor, cfg: TextGrainConfig) -> torch.Tensor:
    """Detection scores Y = -log(1 - u) of (window, token) pairs, Exp(1) under H0. ctx (n, h), tok (n,) -> (n,) float64."""
    b = torch.clamp((prf_uniform(ctx, tok, cfg.key + _SUB_PART) * cfg.B).long(), max=cfg.B - 1)
    J = torch.clamp((prf_uniform(ctx, torch.zeros_like(tok), cfg.key + _SUB_COL) * cfg.m).long(), max=cfg.m - 1)
    u = prf_uniform(ctx, b * cfg.m + J, cfg.key + _SUB_COST).double() + 0.5 / _M
    return -torch.log1p(-u)


def score_positions(token_ids: list[int], cfg: TextGrainConfig) -> torch.Tensor:
    """Per-position scores at the first occurrence of each (window, token) pair; Exp(1) under H0."""
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    h = cfg.h
    seen, keep = set(), []
    for t in range(h, ids.numel()):
        w = tuple(ids[t - h : t + 1].tolist())
        if w not in seen:
            seen.add(w)
            keep.append(t)
    if not keep:
        return torch.zeros(0, dtype=torch.float64)
    ctx = torch.stack([ids[t - h : t] for t in keep])
    return score_pairs(ctx, ids[torch.tensor(keep)], cfg)


def detect(token_ids: list[int], cfg: TextGrainConfig) -> tuple[int, float, float]:
    """(n scored, score sum, p-value) with the sum ~ Gamma(n, 1) under H0."""
    Y = score_positions(token_ids, cfg)
    n = int(Y.numel())
    if n == 0:
        return 0, 0.0, 1.0
    return n, float(Y.sum()), float(special.gammaincc(n, float(Y.sum())))


if __name__ == "__main__":
    # Q averages to P over keys (distortion-free), and the solver removes about beta of the entropy.
    torch.manual_seed(0)
    V, trials = 400, 300
    logits = torch.randn(1, V) * 2.0
    probs = torch.softmax(logits, -1)
    ps, pidx = torch.sort(probs, -1, descending=True)
    mean_Q = torch.zeros(V, dtype=torch.float64)
    betas = []
    for t in range(trials):
        Q, ids, st = textgrain_law(ps, pidx, torch.tensor([[t, 2 * t, 3 * t]]), TextGrainConfig(key=1000 + t, B=64, m=64))
        mean_Q[ids[0]] += Q[0] / trials
        betas.append(float(st["beta_hat"][0]))
    print(f"max |E_key Q - P| = {float((mean_Q - probs[0].double()).abs().max()):.4f} (P max {float(probs.max()):.3f}), mean beta_hat {sum(betas) / trials:.3f} (target 0.5)")
