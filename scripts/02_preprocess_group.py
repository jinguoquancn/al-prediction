"""
02_preprocess_group.py — Preprocessing & AL molecular grouping.
  * Load discovery (TCGA-COADREAD) expression + clinical
  * Log2-normalize, filter low-expression, z-score
  * Build literature AL molecular signature score (up - down) per sample
  * AL-high vs AL-low median split
  * Differential expression (Wilcoxon + BH-FDR) between groups
  * Save normalized matrix, clinical, AL scores, DEG table
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from gene_panel import AL_SIGNATURE_UP, AL_SIGNATURE_DOWN
from scipy.stats import ranksums, mannwhitneyu

def log2norm(df):
    """df: gene x sample (RSEM). log2(x+1)."""
    return np.log2(df.astype(float) + 1)

def zscore_genes(df):
    mu = df.mean(axis=1); sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)

from stats_util import bh_fdr  # NaN-safe shared implementation

def main():
    print("=== 02 preprocessing & AL grouping ===")
    expr = cfg.load_expr("discovery")
    print("raw expr:", expr.shape)
    # keep only primary tumor samples (TCGA -01A/-01B); barcodes already sample ids
    # filter low expression: keep genes expressed in >=10% samples (>log2(2))
    expr = log2norm(expr)
    keep = (expr > 1).sum(axis=1) >= max(10, 0.10*expr.shape[1])
    expr = expr.loc[keep]
    print("after filter:", expr.shape)
    # clinical
    clin = cfg.load_clin("discovery")
    print("clinical:", clin.shape, "attrs sample:", list(clin.columns[:12]))
    # standardized clinical extraction
    cstd = extract_clinical_tcga(clin)
    cstd.to_csv(os.path.join(cfg.PROC, "discovery_clin_std.tsv"), sep="\t")
    print("std clinical:", cstd.shape)
    # restrict samples to those with clinical + primary tumor
    common = [s for s in expr.columns if s in cstd.index]
    expr = expr[common]
    print("expr after clinical intersect:", expr.shape)
    expr.to_csv(os.path.join(cfg.PROC, "discovery_expr_norm.tsv"), sep="\t")

    # z-score for scoring
    ez = zscore_genes(expr)
    up = [g for g in AL_SIGNATURE_UP if g in ez.index]
    dn = [g for g in AL_SIGNATURE_DOWN if g in ez.index]
    print(f"signature genes present: up={len(up)} down={len(dn)}")
    score = ez.loc[up].mean(axis=0) - ez.loc[dn].mean(axis=0)
    score = pd.Series(score, name="AL_sig_score")
    med = score.median()
    group = pd.Series((score > med).map({True:"AL_high",False:"AL_low"}), name="AL_group")
    pd.concat([score, group], axis=1).to_csv(os.path.join(cfg.PROC, "discovery_AL_score.tsv"), sep="\t")
    print(f"AL_high={int((group=='AL_high').sum())} AL_low={int((group=='AL_low').sum())} median={med:.3f}")

    # DEG: Wilcoxon AL_high vs AL_low on all expressed genes
    hi = [s for s in expr.columns if group[s]=="AL_high"]
    lo = [s for s in expr.columns if group[s]=="AL_low"]
    rows=[]
    for g in expr.index:
        a = expr.loc[g, hi].values.astype(float)
        b = expr.loc[g, lo].values.astype(float)
        if np.std(a)==0 and np.std(b)==0: continue
        try:
            st,p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            continue
        fc = np.mean(a)-np.mean(b)
        rows.append((g, fc, st, p))
    deg = pd.DataFrame(rows, columns=["gene","log2FC","stat","pval"])
    deg["FDR"] = bh_fdr(deg["pval"].values)
    deg["sig"] = np.where((deg.FDR<0.05)&(deg.log2FC.abs()>0.5),
                  np.where(deg.log2FC>0,"Up","Down"), "ns")
    deg = deg.sort_values("pval")
    deg.to_csv(os.path.join(cfg.PROC, "discovery_DEG.tsv"), sep="\t", index=False)
    print("DEG total:", len(deg), "Up:", int((deg.sig=="Up").sum()), "Down:", int((deg.sig=="Down").sum()))
    print("top DEG:\n", deg.head(8).to_string(index=False))

def extract_clinical_tcga(clin):
    """Build standardized clinical table indexed by sample barcode."""
    cols = list(clin.columns)
    def pick(*candidates):
        for c in candidates:
            if c in cols: return c
        return None
    out = pd.DataFrame(index=clin.index)
    age = pick("AGE","AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS","AGE_binned"); 
    if age: out["age"]=pd.to_numeric(clin[age], errors="coerce")
    sex = pick("SEX","GENDER")
    if sex: out["sex"]=clin[sex]
    st = pick("AJCC_PATHOLOGIC_TUMOR_STAGE","CLINICAL_STAGE","Pathologic_Stage","Tumor_Stage_Other")
    if st: out["stage"]=clin[st]
    t=pick("PATH_T_STAGE","T");
    if t: out["T"]=clin[t]
    n=pick("PATH_N_STAGE","N");
    if n: out["N"]=clin[n]
    m=pick("PATH_M_STAGE","M");
    if m: out["M"]=clin[m]
    msi=pick("MSI_SCORE_MANTIS","MSI_SCORE","MSI_BINARY","MSI_SENSOR_SCORE","MSI_Status")
    if msi: out["MSI"]=clin[msi]
    osm=pick("OS_MONTHS","OS_Months")
    if osm: out["OS_time"]=pd.to_numeric(clin[osm], errors="coerce")
    oss=pick("OS_STATUS","OS_Status")
    if oss:
        out["OS_event"]=clin[oss].astype(str).str.startswith("1").astype(int)
    dfim=pick("DFI_MONTHS","DFS_MONTHS","PFS_MONTHS")
    if dfim: out["DFI_time"]=pd.to_numeric(clin[dfim], errors="coerce")
    dfis=pick("DFI_STATUS","DFS_STATUS","PFS_STATUS")
    if dfis: out["DFI_event"]=clin[dfis].astype(str).str.startswith("1").astype(int)
    mut=pick("MUTATION_COUNT","Mutation_Count")
    if mut: out["mutation_count"]=pd.to_numeric(clin[mut], errors="coerce")
    ctype=pick("SAMPLE_TYPE","Sample_Type")
    if ctype: out["sample_type"]=clin[ctype]
    # determine colon vs rectal from barcode (TCGA-XX-####): rectal ~ 'READ' - use patient? we use clinical sample type
    out["cohort"]="discovery"
    return out

if __name__=="__main__":
    main()
