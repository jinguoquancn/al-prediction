"""
stats_util.py — Shared statistics helpers used across the pipeline.

bh_fdr() is deliberately NaN-safe: cBioPortal matrices carry missing values,
so several tests legitimately return p = NaN. The textbook BH implementation
(np.minimum.accumulate over the reversed ranked p*n/rank) smears a trailing
NaN backwards through the whole array and silently turns every FDR into NaN,
which in turn makes every "significant" call fail. Non-finite p-values are
therefore excluded from the ranking and returned as NaN.
"""
import numpy as np


def bh_fdr(p):
    """Benjamini-Hochberg step-up FDR, NaN-safe.

    Parameters
    ----------
    p : array-like of float, may contain NaN/inf

    Returns
    -------
    np.ndarray of float with the same length; entries corresponding to
    non-finite p are NaN, the rest are BH-adjusted and capped at 1.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    fdr = ranked * m / np.arange(1, m + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.minimum(fdr, 1.0)
    tmp = np.empty(m)
    tmp[order] = fdr
    out[ok] = tmp
    return out
