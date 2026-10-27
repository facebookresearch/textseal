# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Algorithm A.1 as a chunked Triton kernel, for tables too large for registers (e.g. 256 x 256).

One program per batch row, as in sinkhorn_triton.py, but the B x m table stays in
global memory (L2-resident) and is streamed in chunks of RC rows. Row reductions are
done inside a chunk; column reductions are accumulated across chunks with an online
log-sum-exp (running max and rescaled sum). Each iteration is eight streaming passes:

  A  logu = logp - LSE_j(logK + logv)            (line 11, rows)       -> scratch
  B  logv = log(1/m) - LSE_b(logK + logu)         (line 11, columns), centred (line 12)
  C  logu = logp - LSE_j(logK + logv)             (line 13)            -> scratch
  D  cL_j = LSE_b(logu + logK)                    column normaliser of line 14
  E  rs_b = sum_j pi_s ; x_b = min(p_b / rs_b, 1)                      -> scratch (Eq. A.2)
  F  cs_j = sum_b pi_s x_b ; y_j = min((1/m) / cs_j, 1)
  G  X2 = pi_s x_b y_j ; ea_b, eb_j, D                                 -> scratch
  H  X3 = X2 + ea eb^T / D ; KL(X3 || p x u_m)    (line 16) ; on convergence a final pass stores X3

DTYPE selects float64 (same arithmetic as the reference) or float32.
"""
import math
import torch
import triton
import triton.language as tl

from textseal.watermarking.textgrain import TextGrainConfig

NEG_HOST = -1.0e30
NEG = tl.constexpr(NEG_HOST)
# Divisions are guarded with > TINY rather than > 0: on the GPU a denormal denominator
# passes the comparison but is flushed to zero inside the division (0/0 = NaN).
TINY = tl.constexpr(1.0e-30)


@triton.jit
def _chunk(G_ptr, row, c, B, m, RC: tl.constexpr, MB: tl.constexpr, DTYPE: tl.constexpr):
    rb = c * RC + tl.arange(0, RC)
    cm = tl.arange(0, MB)
    rin = rb < B
    cok = cm < m
    mask = rin[:, None] & cok[None, :]
    G = tl.load(G_ptr + row * B * m + rb[:, None] * m + cm[None, :], mask=mask, other=0.0).to(DTYPE)
    return rb, rin, cok, mask, G


@triton.jit
def _sinkhorn_budget_chunked(G_ptr, p_ptr, logp_ptr, Hs_ptr, bt_ptr, delta_ptr, triv_ptr,
                             pi_ptr, beta_ptr, status_ptr, su_ptr, sx_ptr, se_ptr,
                             B, m, n_it, eps, lam_min, lam_max,
                             RC: tl.constexpr, MB: tl.constexpr, NCH: tl.constexpr, DTYPE: tl.constexpr):
    row = tl.program_id(0)
    cm = tl.arange(0, MB)
    cok = cm < m
    um = (1.0 / m.to(tl.float64)).to(DTYPE)
    log_um = tl.log(um)
    triv = tl.load(triv_ptr + row)
    if triv != 0:
        for c in range(NCH):
            rb = c * RC + tl.arange(0, RC)
            rin = rb < B
            p = tl.load(p_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
            tl.store(pi_ptr + row * B * m + rb[:, None] * m + cm[None, :],
                     (p[:, None] * um + tl.zeros((RC, MB), dtype=DTYPE)).to(tl.float64),
                     mask=rin[:, None] & cok[None, :])
        tl.store(beta_ptr + row, 0.0)
        tl.store(status_ptr + row, 0)
    else:
        Hs = tl.load(Hs_ptr + row).to(DTYPE)
        bt = tl.load(bt_ptr + row).to(DTYPE)
        delta = tl.load(delta_ptr + row).to(DTYPE)
        lam = (1.0 + tl.zeros((), dtype=tl.float64)).to(DTYPE)
        logv = tl.where(cok, log_um, 0.0).to(DTYPE)
        s = 0
        done = 0
        while (s < n_it) & (done == 0):
            # A: rows
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                p = tl.load(p_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
                lp = tl.load(logp_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                ok = rin & (p > 0)
                t = tl.where(mask, G / lam + logv[None, :], NEG)
                mx = tl.max(t, axis=1)
                lse = mx + tl.log(tl.sum(tl.where(mask, tl.exp(t - mx[:, None]), 0.0), axis=1))
                tl.store(su_ptr + row * B + rb, tl.where(ok, lp - lse, NEG), mask=rin)
            # B: columns (online LSE over row chunks)
            cmax = tl.full((MB,), NEG, DTYPE)
            csum = tl.zeros((MB,), DTYPE)
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                lu = tl.load(su_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                valid = mask & (lu > NEG * 0.5)[:, None]
                t = tl.where(valid, G / lam + lu[:, None], NEG)
                nmax = tl.maximum(cmax, tl.max(t, axis=0))
                csum = csum * tl.exp(cmax - nmax) + tl.sum(tl.where(valid, tl.exp(t - nmax[None, :]), 0.0), axis=0)
                cmax = nmax
            logv = tl.where(cok, log_um - (cmax + tl.log(csum)), 0.0)
            logv = tl.where(cok, logv - tl.sum(logv, axis=0) / m, 0.0)
            # C: rows again (refresh u)
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                p = tl.load(p_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
                lp = tl.load(logp_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                ok = rin & (p > 0)
                t = tl.where(mask, G / lam + logv[None, :], NEG)
                mx = tl.max(t, axis=1)
                lse = mx + tl.log(tl.sum(tl.where(mask, tl.exp(t - mx[:, None]), 0.0), axis=1))
                tl.store(su_ptr + row * B + rb, tl.where(ok, lp - lse, NEG), mask=rin)
            # D: column normaliser of diag(u) K
            cmax = tl.full((MB,), NEG, DTYPE)
            csum = tl.zeros((MB,), DTYPE)
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                lu = tl.load(su_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                valid = mask & (lu > NEG * 0.5)[:, None]
                t = tl.where(valid, lu[:, None] + G / lam, NEG)
                nmax = tl.maximum(cmax, tl.max(t, axis=0))
                csum = csum * tl.exp(cmax - nmax) + tl.sum(tl.where(valid, tl.exp(t - nmax[None, :]), 0.0), axis=0)
                cmax = nmax
            cL = cmax + tl.log(tl.where(csum > TINY, csum, 1.0))
            # E: row sums of pi_s -> x
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                lu = tl.load(su_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                p = tl.load(p_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
                valid = mask & (lu > NEG * 0.5)[:, None]
                pis = tl.where(valid, tl.exp(lu[:, None] + G / lam - cL[None, :]) * um, 0.0)
                rs = tl.sum(pis, axis=1)
                x = tl.where(rs > TINY, tl.minimum(p / tl.where(rs > TINY, rs, 1.0), 1.0), 1.0)
                tl.store(sx_ptr + row * B + rb, x, mask=rin)
            # F: column sums of pi_s x -> y
            cs = tl.zeros((MB,), DTYPE)
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                lu = tl.load(su_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                x = tl.load(sx_ptr + row * B + rb, mask=rin, other=1.0).to(DTYPE)
                valid = mask & (lu > NEG * 0.5)[:, None]
                pis = tl.where(valid, tl.exp(lu[:, None] + G / lam - cL[None, :]) * um, 0.0)
                cs += tl.sum(pis * x[:, None], axis=0)
            y = tl.where(cs > TINY, tl.minimum(um / tl.where(cs > TINY, cs, 1.0), 1.0), 1.0)
            # G: X2 = pi_s x y ; deficits
            cs2 = tl.zeros((MB,), DTYPE)
            D = tl.zeros((), DTYPE)
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                lu = tl.load(su_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                x = tl.load(sx_ptr + row * B + rb, mask=rin, other=1.0).to(DTYPE)
                p = tl.load(p_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
                valid = mask & (lu > NEG * 0.5)[:, None]
                X2 = tl.where(valid, tl.exp(lu[:, None] + G / lam - cL[None, :]) * um, 0.0) * x[:, None] * y[None, :]
                ea = tl.where(rin, tl.maximum(p - tl.sum(X2, axis=1), 0.0), 0.0)
                tl.store(se_ptr + row * B + rb, ea, mask=rin)
                D += tl.sum(ea, axis=0)
                cs2 += tl.sum(X2, axis=0)
            eb = tl.where(cok, tl.maximum(um - cs2, 0.0), 0.0)
            Dz = tl.where(D > TINY, D, 1.0)
            # H: corrected coupling and its KL to independence
            kl = tl.zeros((), DTYPE)
            for c in range(NCH):
                rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                lu = tl.load(su_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                x = tl.load(sx_ptr + row * B + rb, mask=rin, other=1.0).to(DTYPE)
                ea = tl.load(se_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
                p = tl.load(p_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
                valid = mask & (lu > NEG * 0.5)[:, None]
                X2 = tl.where(valid, tl.exp(lu[:, None] + G / lam - cL[None, :]) * um, 0.0) * x[:, None] * y[None, :]
                X3 = tl.where(mask, X2 + tl.where(D > TINY, ea[:, None] * eb[None, :] / Dz, 0.0), 0.0)
                ref = p[:, None] * um
                pos = (X3 > 0) & (ref > 0)
                kl += tl.sum(tl.sum(tl.where(pos, X3 * (tl.log(tl.where(pos, X3, 1.0)) - tl.log(tl.where(pos, ref, 1.0))),
                                             0.0), axis=1), axis=0)
            beta_hat = kl / Hs
            conv = tl.abs(beta_hat - bt) <= eps
            if conv | (s == n_it - 1):
                for c in range(NCH):
                    rb, rin, cok_, mask, G = _chunk(G_ptr, row, c, B, m, RC, MB, DTYPE)
                    lu = tl.load(su_ptr + row * B + rb, mask=rin, other=NEG).to(DTYPE)
                    x = tl.load(sx_ptr + row * B + rb, mask=rin, other=1.0).to(DTYPE)
                    ea = tl.load(se_ptr + row * B + rb, mask=rin, other=0.0).to(DTYPE)
                    valid = mask & (lu > NEG * 0.5)[:, None]
                    X2 = tl.where(valid, tl.exp(lu[:, None] + G / lam - cL[None, :]) * um, 0.0) * x[:, None] * y[None, :]
                    X3 = tl.where(mask, X2 + tl.where(D > TINY, ea[:, None] * eb[None, :] / Dz, 0.0), 0.0)
                    tl.store(pi_ptr + row * B * m + rb[:, None] * m + cm[None, :], X3.to(tl.float64), mask=mask)
                tl.store(beta_ptr + row, beta_hat.to(tl.float64))
                tl.store(status_ptr + row, tl.where(conv, 1, 2))
                done = 1
            else:
                ratio = tl.maximum(tl.maximum(beta_hat, 0.0), delta) / bt
                lam_new = tl.minimum(tl.maximum(lam * tl.sqrt(ratio), lam_min), lam_max)
                logv = logv * (lam / lam_new)
                lam = lam_new
            s += 1


def solve_coupling_chunked(p_blk, G, H_P, cfg: TextGrainConfig, lam_min=1e-2, lam_max=1e2,
                           dtype=torch.float64, rows_per_chunk=None, num_warps=8):
    bsz, B, m = G.shape
    p, G64, H_P = p_blk.double().contiguous(), G.double(), H_P.double()
    active = p > 0
    logp = torch.where(active, torch.log(torch.where(active, p, 1.0)), torch.full_like(p, NEG_HOST))
    H_blk = -torch.where(active, p * torch.where(active, logp, 0.0), 0.0).sum(1)
    Hs = torch.where(H_P > 0, H_P, 1.0)
    beta_t = torch.full_like(H_P, cfg.beta)
    trivial = (cfg.beta == 0) | (H_blk <= 0) | (H_P <= 0)
    beta_t = torch.minimum(beta_t, (1 - 1e-8) * H_blk / Hs)
    beta_t = torch.minimum(beta_t, math.log(m) / Hs)
    trivial = trivial | (beta_t <= 0)
    beta_t = torch.where(trivial, torch.zeros_like(beta_t), beta_t)
    delta = 1e-12 * H_blk / Hs
    MB = triton.next_power_of_2(m)
    # At least two chunks: with a single-trip chunk loop (e.g. B = m = 64, RC = 64) the compiled
    # kernel returned wrong couplings on the GPU (max |beta_hat - ref| up to 0.48) while RC <= 32
    # matched the reference to 1e-9; see validate_kernels.py.
    RC = rows_per_chunk or max(1, min(triton.next_power_of_2(B) // 2, 8192 // MB))
    nch = triton.cdiv(B, RC)
    pi = torch.empty((bsz, B, m), dtype=torch.float64, device=G.device)
    beta = torch.empty(bsz, dtype=torch.float64, device=G.device)
    status = torch.empty(bsz, dtype=torch.int32, device=G.device)
    su = torch.empty((bsz, B), dtype=dtype, device=G.device)
    sx = torch.empty_like(su)
    se = torch.empty_like(su)
    tdt = tl.float64 if dtype == torch.float64 else tl.float32
    _sinkhorn_budget_chunked[(bsz,)](G64.to(dtype).contiguous(), p, logp.contiguous(), Hs.contiguous(),
                                     beta_t.contiguous(), delta.contiguous(), trivial.to(torch.int32).contiguous(),
                                     pi, beta, status, su, sx, se, B, m, cfg.n_it, cfg.eps, lam_min, lam_max,
                                     RC=RC, MB=MB, NCH=nch, DTYPE=tdt, num_warps=num_warps)
    bad = ~(torch.isfinite(pi).flatten(1).all(1) & torch.isfinite(beta))
    if bool(bad.any()):                                   # never expected; never sample from a broken table
        from textseal.watermarking.textgrain import solve_coupling
        idx = bad.nonzero().flatten()
        r = solve_coupling(p_blk[idx], G[idx], H_P[idx], cfg, lam_min=lam_min, lam_max=lam_max)
        pi[idx], beta[idx], status[idx] = r[0], r[1], r[3].to(status.dtype)
    return pi, beta, beta_t, status.long()
