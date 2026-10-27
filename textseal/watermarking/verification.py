# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Exact verification rules for watermarked speculative decoding (used by textseal.watermarking.speculative).

Token-level race coupling ("agreement"): the draft race x = argmin E/q and the target race x_p = argmin E/p share the
keyed clocks E. Accept if x = x_p, else accept with probability beta_x; on rejection emit x_p with probability
min(1, (p - q)_+ / L) and otherwise a fresh draw from the leftover. Exact output law p and acceptance 1 - TV, with
J[w, v] = P(x = w, x_p = v) computed in O(K log K) from t = q / p:
    J[v, v] = p_v / (sum_{t_u <= t_v} p_u + sum_{t_u > t_v} q_u / t_v),
    J[w, v] = q_w p_v (Phi(t_w) - Phi(t_v)) for t_v < t_w, Phi(t) = int_0^t dl / F(l)^2, F(l) = sum_u max(q_u, l p_u).

Block verification (Sun et al., 2025): a call has rows 0..N; row i < N drew draft d_i from the drafter law q_i, row N is
the bonus row; p_i is the target law at row i. Row clocks E_i are i.i.d. Exp(1) (keyed copula clocks, or fresh ones
when the row's window is not free). Block verification emits (L, X) with probability W_L R_L(X) given the drafts:
    r_0 = 1, r_i = min(1, r_{i-1} p_{i-1}(d_{i-1}) / q_{i-1}(d_{i-1})),  rho_l = (r_l p_l - q_l)_+ (l < N), R_l = rho_l / z_l, R_N = p_N,
    h_i = z_i / (z_i + 1 - r_i) (1 <= i < N), h_N = r_N,  W_l = h_l prod_{j > l} (1 - h_j), W_0 = prod_j (1 - h_j).
Decisions (all exact; keyed drafts d_i = argmin E_i / q_i with time tau_i, leftovers E'_i = E_i - tau_i q_i, E'_N = E_N):
    residj  (L, X) = argmin over (l, x) of E'_l(x) / (W_l R_l(x))                       (TextSeal2: residual J race)
    ts2b    native stop L ~ W (fresh), X = argmin E'_L / R_L
    taub    L = max{i : 1 - exp(-tau_i) <= h_i} (0 if none), X = argmin E'_L / R_L
    agreeb  plain-decoding outcome O* on the same clocks with probability min(1, V(O*) / P*(O*)), else a race on (V - P*)_+
            with fresh clocks for rows <= l*
    jonly   native drafts d_i ~ q_i, (L, X) = argmin E_l(x) / (W_l R_l(x))
    residjraw  keyed drafts, J race on the raw clocks E instead of the leftovers (biased negative control)
"""

import torch

def _sorted_ratios(p: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.where(p > 0, q / p.clamp(min=1e-300), torch.full_like(p, float("inf")))
    ts, order = torch.sort(t)
    return t, ts, order


def diag_joint(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """J[v, v] for all v (K,), float64 1-D inputs on the union support."""
    t, ts, order = _sorted_ratios(p, q)
    ps, qs = p[order], q[order]
    cum_p = torch.cumsum(ps, 0)  # Sum of p over t_u <= t_(k).
    tail_q = qs.sum() - torch.cumsum(qs, 0)  # Sum of q over t_u > t_(k).
    denom = cum_p + tail_q / ts.clamp(min=1e-300)
    diag_sorted = torch.where((ps > 0) & (qs > 0), ps / denom, torch.zeros_like(ps))
    return torch.empty_like(p).scatter_(0, order, diag_sorted)


def phi(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Phi(t_v) for all v (K,)."""
    t, ts, order = _sorted_ratios(p, q)
    n = len(ts)
    zero = torch.zeros(1, dtype=p.dtype, device=p.device)
    cum_p = torch.cat([zero, torch.cumsum(p[order], 0)])  # (n + 1,): segment k has the first k tokens at l p.
    cum_q = torch.cat([zero, torch.cumsum(q[order], 0)])
    coef_a, coef_b = q.sum() - cum_q, cum_p
    lo = torch.cat([zero, ts])
    hi = torch.cat([ts, torch.full_like(zero, float("inf"))])
    inv = lambda x: torch.where(torch.isinf(x), torch.zeros_like(x), 1 / x)
    linear = (inv(coef_a + coef_b * lo) - inv(coef_a + coef_b * hi)) / coef_b.clamp(min=1e-300)
    flat = (hi - lo) / coef_a.clamp(min=1e-300) ** 2
    pieces = torch.where(hi > lo, torch.where(coef_b > 0, linear, flat), torch.zeros_like(lo))
    phi_sorted = torch.cumsum(pieces, 0)[:n]
    return torch.empty_like(p).scatter_(0, order, phi_sorted)


def verify(p: torch.Tensor, q: torch.Tensor, x: int, xp: int, generator: torch.Generator | None = None) -> tuple[bool, int, bool]:
    """Agreement-first decision for draft x and target-race winner xp (1-D laws, same vocabulary).
    Returns (accepted, emitted token, emitted token is the fresh fallback)."""
    if x == xp:
        return True, x, False
    support = torch.nonzero((p > 0) | (q > 0))[:, 0]
    ps, qs = p[support].double(), q[support].double()
    ix = int(torch.searchsorted(support, torch.tensor(x, device=support.device)))
    ixp = int(torch.searchsorted(support, torch.tensor(xp, device=support.device)))
    jdiag = diag_joint(ps, qs)
    mins = torch.minimum(ps, qs)
    gap = qs - jdiag
    beta = torch.where(gap > 1e-12, (mins - jdiag) / gap.clamp(min=1e-12), torch.zeros_like(gap))
    if float(torch.rand((), generator=generator, device=p.device)) < float(beta[ix]):
        return True, x, False
    t, ts, order = _sorted_ratios(ps, qs)
    phis = phi(ps, qs)
    weight = qs * (1 - beta)  # Mass of a w != v draft that rejects.
    ws, phs = weight[order], phis[order]
    suffix_w = torch.flip(torch.cumsum(torch.flip(ws, [0]), 0), [0]) - ws  # Sum over strictly later sorted positions.
    suffix_wphi = torch.flip(torch.cumsum(torch.flip(ws * phs, [0]), 0), [0]) - ws * phs
    rej_sorted = ps[order] * (suffix_wphi - phs * suffix_w)
    rej = torch.empty_like(ps).scatter_(0, order, rej_sorted).clamp(min=0)  # L_v: rejection mass landing on x_p = v.
    excess = (ps - qs).clamp(min=0)  # TV r_v.
    keep = torch.where(rej > 1e-15, torch.minimum(torch.ones_like(rej), excess / rej.clamp(min=1e-300)), torch.zeros_like(rej))
    if float(torch.rand((), generator=generator, device=p.device)) < float(keep[ixp]):
        return False, xp, False
    leftover = (excess - rej).clamp(min=0)
    leftover = leftover if float(leftover.sum()) > 1e-15 else excess
    pick = int(torch.multinomial(leftover.float() / float(leftover.sum()), 1, generator=generator))
    return False, int(support[pick]), True


def block_law(p: torch.Tensor, q: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """p (N+1, V), q (N, V), d (N,) -> W (N+1,), R (N+1, V), h (N+1,). float64."""
    n = q.shape[0]
    r = torch.ones(n + 1, dtype=torch.float64, device=p.device)
    ratio = p[torch.arange(n), d] / q[torch.arange(n), d].clamp(min=1e-300)  # N
    for ii in range(1, n + 1):
        r[ii] = torch.clamp(r[ii - 1] * ratio[ii - 1], max=1.0)
    res = (r[:n, None] * p[:n] - q).clamp(min=0)  # N V
    z = res.sum(1)
    R = torch.cat([res / torch.where(z > 0, z, torch.ones_like(z))[:, None], p[n:n + 1]], 0)  # N+1 V
    h = torch.ones(n + 1, dtype=torch.float64, device=p.device)
    if n > 1:
        den = z[1:] + (1 - r[1:n])
        h[1:n] = torch.where(den > 0, z[1:] / den.clamp(min=1e-300), torch.ones_like(den)).clamp(0, 1)
    h[n] = r[n]
    one_minus = 1 - h[1:]  # N
    tail = torch.cat([torch.flip(torch.cumprod(torch.flip(one_minus, [0]), 0), [0]), torch.ones(1, dtype=torch.float64, device=p.device)])  # N+1
    W = tail.clone()
    W[1:] = W[1:] * h[1:]
    return W, R, h


def race2d(clock: torch.Tensor, mass: torch.Tensor) -> tuple[int, int]:
    """argmin clock / mass over cells of positive mass of an (N+1, V) table -> (row, token)."""
    ratio = torch.where(mass > 0, clock / mass.clamp(min=1e-300), torch.full_like(mass, float("inf")))
    flat = int(torch.argmin(ratio))
    return flat // mass.shape[1], flat % mass.shape[1]


def joint_row(p: torch.Tensor, q: torch.Tensor, w: int) -> tuple[torch.Tensor, float]:
    """J[w, v] for all v (V,) and J[w, w], for one row's laws (V,), on the union support."""
    support = torch.nonzero((p > 0) | (q > 0))[:, 0]
    ps, qs = p[support], q[support]
    phis = phi(ps, qs)
    t = torch.where(ps > 0, qs / ps.clamp(min=1e-300), torch.full_like(ps, float("inf")))
    iw = int(torch.searchsorted(support, torch.tensor(w, device=support.device)))
    row = torch.where(t < t[iw], qs[iw] * ps * (phis[iw] - phis), torch.zeros_like(ps))
    jww = float(diag_joint(ps, qs)[iw])
    row[iw] = jww
    return torch.zeros_like(p).scatter_(0, support, row), jww


def decide(scheme: str, p: torch.Tensor, q: torch.Tensor, d: torch.Tensor, E: torch.Tensor, generator: torch.Generator | None = None) -> dict:
    """One block decision. p (N+1, V), q (N, V), d (N,), E (N+1, V) row clocks. Returns L, X and diagnostics."""
    n = q.shape[0]
    W, R, h = block_law(p, q, d)
    mass = W[:, None] * R
    rows = torch.arange(n, device=p.device)
    tau = E[rows, d] / q[rows, d].clamp(min=1e-300)  # N
    resid = E.clone()
    resid[:n] = E[:n] - tau[:, None] * q
    info = {"fallback": 0, "plain_match": 0}
    if scheme == "ts2b":
        L = int(torch.multinomial(W.float() / float(W.sum()), 1, generator=generator))
        X = int(torch.argmin(torch.where(R[L] > 0, resid[L] / R[L].clamp(min=1e-300), torch.full_like(R[L], float("inf")))))
    elif scheme == "taub":
        succ = torch.nonzero(-torch.expm1(-tau) <= h[1:])[:, 0]
        L = int(succ[-1]) + 1 if len(succ) else 0
        X = int(torch.argmin(torch.where(R[L] > 0, resid[L] / R[L].clamp(min=1e-300), torch.full_like(R[L], float("inf")))))
    elif scheme == "residj":
        L, X = race2d(resid, mass)
    elif scheme in ("jonly", "residjraw"):
        L, X = race2d(E, mass)
    elif scheme == "agreeb":
        y = torch.argmin(torch.where(p > 0, E / p.clamp(min=1e-300), torch.full_like(p, float("inf"))), 1)  # N+1
        dis = torch.nonzero(y[:n] != d)[:, 0]
        l0 = int(dis[0]) if len(dis) else n
        x0 = int(y[l0])
        pstar = torch.zeros_like(mass)
        A = 1.0
        for ii in range(n):
            jrow, jdd = joint_row(p[ii], q[ii], int(d[ii]))
            pstar[ii] = A * jrow / float(q[ii, d[ii]])
            pstar[ii, d[ii]] = 0.0
            A *= jdd / float(q[ii, d[ii]])
        pstar[n] = A * p[n]
        u = float(torch.rand((), generator=generator, dtype=torch.float64, device=p.device))
        if u * float(pstar[l0, x0]) < float(mass[l0, x0]):
            L, X = l0, x0
        else:
            clock = resid.clone()
            clock[:l0 + 1] = torch.empty_like(clock[:l0 + 1]).exponential_(generator=generator)
            L, X = race2d(clock, (mass - pstar).clamp(min=0))
            info["fallback"] = 1
        info["plain_match"] = int((L, X) == (l0, x0))
    else:
        raise ValueError(scheme)
    return {"L": L, "X": X, **info}

if __name__ == "__main__":
    # J against Monte Carlo, acceptance and output law of the token rule; law of (d, L, X) of every block decision.
    def check_token_rule():
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(1)
        def dense_joint(p, q):
            phis, jd = phi(p, q), diag_joint(p, q)
            t = torch.where(p > 0, q / p.clamp(min=1e-300), torch.full_like(p, float("inf")))
            jj = torch.where(t[None, :] < t[:, None], q[:, None] * p[None, :] * (phis[:, None] - phis[None, :]), torch.zeros(len(p), len(p), dtype=p.dtype))
            jj[range(len(p)), range(len(p))] = jd
            return jj
        vocab = 12
        for case in range(3):
            zz = torch.randn(vocab, dtype=torch.float64) * 1.5
            p = torch.softmax(zz, 0)
            q = torch.softmax(zz + torch.randn(vocab, dtype=torch.float64), 0)
            p[torch.randperm(vocab)[:3]] = 0  # Top-p style zeros, different for p and q.
            q[torch.randperm(vocab)[:3]] = 0
            p, q = p / p.sum(), q / q.sum()
            n_draws = 400_000
            clocks = torch.empty(n_draws, vocab, dtype=torch.float64).exponential_()
            x = torch.argmin(torch.where(q > 0, clocks / q.clamp(min=1e-300), torch.full_like(clocks, float("inf"))), 1)
            xp = torch.argmin(torch.where(p > 0, clocks / p.clamp(min=1e-300), torch.full_like(clocks, float("inf"))), 1)
            emp = torch.zeros(vocab, vocab, dtype=torch.float64).index_put_((x, xp), torch.ones(n_draws, dtype=torch.float64), accumulate=True) / n_draws
            jj = dense_joint(p, q)
            print(f"case {case}: |J - MC| max {float((emp - jj).abs().max()):.4f} (se ~{(0.25 / n_draws) ** 0.5:.4f}), row sums - q {float((jj.sum(1) - q).abs().max()):.1e}, col sums - p {float((jj.sum(0) - p).abs().max()):.1e}")
            n_scheme = 40_000
            out, acc, fb = torch.zeros(vocab), 0, 0
            for ii in range(n_scheme):
                ok, tok, fallback = verify(p, q, int(x[ii]), int(xp[ii]), gen)
                out[tok] += 1; acc += ok; fb += fallback
            tv = float((p - q).clamp(min=0).sum())
            se_out = (float(p.max()) / n_scheme) ** 0.5
            print(f"         acceptance {acc / n_scheme:.4f} vs 1 - TV {1 - tv:.4f} (se {(tv * (1 - tv) / n_scheme) ** 0.5:.4f}), max |emp - p| {float((out / n_scheme - p).abs().max()):.4f} (se ~{se_out:.4f}), fallback {fb / n_scheme:.3f}")

    def check_block_rules():
        # Exactness: fixed laws per row (V = 5, N = 3), fresh clocks per call; law of (d, L, X) against Q(d) W_L R_L(X).
        import itertools
        from collections import Counter
        from scipy.stats import chi2
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(1)
        vocab, n, calls = 5, 3, 60_000
        P = torch.softmax(torch.randn(n + 1, vocab, dtype=torch.float64) * 1.3, 1)
        Q = torch.softmax(torch.log(P[:n]) + torch.randn(n, vocab, dtype=torch.float64), 1)
        P[1, 4] = 0.0; P[1] /= P[1].sum()  # Top-p style zeros.
        Q[2, 0] = 0.0; Q[2] /= Q[2].sum()
        expected = {}
        for dd in itertools.product(range(vocab), repeat=n):
            dt = torch.tensor(dd)
            qd = float(torch.prod(Q[torch.arange(n), dt]))
            if qd == 0:
                continue
            W, R, _ = block_law(P, Q, dt)
            for l, x in itertools.product(range(n + 1), range(vocab)):
                if float(W[l] * R[l, x]) > 0:
                    expected[(dd, l, x)] = calls * qd * float(W[l] * R[l, x])
        for scheme in ("ts2b", "taub", "residj", "agreeb", "jonly", "residjraw"):
            obs, fallback = Counter(), 0
            for _ in range(calls):
                E = torch.empty(n + 1, vocab, dtype=torch.float64).exponential_(generator=gen)
                if scheme == "jonly":
                    d = torch.multinomial(Q, 1, generator=gen)[:, 0]
                else:
                    d = torch.argmin(torch.where(Q > 0, E[:n] / Q.clamp(min=1e-300), torch.full_like(Q, float("inf"))), 1)
                out = decide(scheme, P, Q, d, E, gen)
                obs[(tuple(d.tolist()), out["L"], out["X"])] += 1
                fallback += out["fallback"]
            big = [k for k in expected if expected[k] >= 5]
            small = [k for k in expected if expected[k] < 5]
            o = [obs.get(k, 0) for k in big] + [sum(obs.get(k, 0) for k in small)]
            e = [expected[k] for k in big] + [sum(expected[k] for k in small)]
            stat = sum((a - b) ** 2 / b for a, b in zip(o, e) if b > 0)
            impossible = sum(c for k, c in obs.items() if k not in expected)
            print(f"{scheme:7s} chi-squared {stat:7.1f}, {len(o) - 1} dof, p = {chi2.sf(stat, len(o) - 1):.3f}, impossible outcomes {impossible}, fallback {fallback / calls:.3f}")

    check_token_rule()
    check_block_rules()
