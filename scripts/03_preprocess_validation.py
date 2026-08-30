"""
03_preprocess_validation.py — Validation-cohort preprocessing (coad_silu_2022).
Applies the SAME pipeline as discovery: log2-norm, filtering, z-score,
AL molecular signature score (same gene weights), median split.
Clinical standardization adapted to available attributes (MUTATION_COUNT/TMB).
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from gene_panel import AL_SIGNATURE_UP, AL_SIGNATURE_DOWN
from scipy.stats import mannwhitneyu

from stats_util import bh_fdr  # NaN-safe shared implementation

def main():
    print("=== 03 validation preprocessing ===")
    expr = pd.read_csv(os.path.join(cfg.PROC, "validation_expr.tsv"), sep="\t", index_col=0)
    print("raw expr:", expr.shape)
    expr = np.log2(expr.astype(float) + 1)
    keep = (expr > 1).sum(axis=1) >= max(5, 0.10 * expr.shape[1])
    expr = expr.loc[keep]
    print("after filter:", expr.shape)

    clin = pd.read_csv(os.path.join(cfg.PROC, "validation_clinical.tsv"), sep="\t", index_col=0)
    print("clinical:", clin.shape)
    # standardize available clinical fields
    cstd = pd.DataFrame(index=clin.index)
    cstd["cancer_type"] = clin.get("CANCER_TYPE_DETAILED", pd.Series(index=clin.index))
    cstd["sample_type"] = clin.get("SAMPLE_TYPE", pd.Series(index=clin.index))
    cstd["mutation_count"] = pd.to_numeric(clin.get("MUTATION_COUNT"), errors="coerce")
    cstd["TMB"] = pd.to_numeric(clin.get("TMB_NONSYNONYMOUS"), errors="coerce")
    cstd["fraction_genome_altered"] = pd.to_numeric(clin.get("FRACTION_GENOME_ALTERED"), errors="coerce")
    cstd["cohort"] = "validation"
    cstd.to_csv(os.path.join(cfg.PROC, "validation_clin_std.tsv"), sep="\t")
    print("std clinical:", cstd.shape)

    common = [s for s in expr.columns if s in cstd.index]
    expr = expr[common]
    print("expr after clinical intersect:", expr.shape)
    expr.to_csv(os.path.join(cfg.PROC, "validation_expr_norm.tsv"), sep="\t")

    # AL score with SAME signature (score = mean z(up) - mean z(down))
    mu = expr.mean(axis=1); sd = expr.std(axis=1).replace(0, np.nan)
    ez = expr.sub(mu, axis=0).div(sd, axis=0)
    up = [g for g in AL_SIGNATURE_UP if g in ez.index]
    dn = [g for g in AL_SIGNATURE_DOWN if g in ez.index]
    print(f"signature genes present: up={len(up)} down={len(dn)}")
    score = ez.loc[up].mean(axis=0) - ez.loc[dn].mean(axis=0)
    score = pd.Series(score, name="AL_sig_score")
    med = score.median()
    group = pd.Series((score > med).map({True: "AL_high", False: "AL_low"}), name="AL_group")
    pd.concat([score, group], axis=1).to_csv(os.path.join(cfg.PROC, "validation_AL_score.tsv"), sep="\t")
    print(f"AL_high={int((group=='AL_high').sum())} AL_low={int((group=='AL_low').sum())} median={med:.3f}")

    # DEG in validation cohort (reproduces discovery structure)
    hi = [s for s in expr.columns if group[s] == "AL_high"]
    lo = [s for s in expr.columns if group[s] == "AL_low"]
    rows = []
    for g in expr.index:
        a = expr.loc[g, hi].values.astype(float)
        b = expr.loc[g, lo].values.astype(float)
        if np.std(a) == 0 and np.std(b) == 0: continue
        try:
            st, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            continue
        rows.append((g, np.mean(a) - np.mean(b), st, p))
    deg = pd.DataFrame(rows, columns=["gene", "log2FC", "stat", "pval"])
    deg["FDR"] = bh_fdr(deg["pval"].values)
    deg["sig"] = np.where((deg.FDR < 0.05) & (deg.log2FC.abs() > 0.5),
                  np.where(deg.log2FC > 0, "Up", "Down"), "ns")
    deg = deg.sort_values("pval")
    deg.to_csv(os.path.join(cfg.PROC, "validation_DEG.tsv"), sep="\t", index=False)
    print("DEG total:", len(deg), "Up:", int((deg.sig == "Up").sum()), "Down:", int((deg.sig == "Down").sum()))
    print("=== DONE ===")

if __name__ == "__main__":
    main()
