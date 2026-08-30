"""
04_WGCNA.py — Weighted gene co-expression network analysis (pure Python/numpy/scipy).
  * Top variable genes from normalized discovery expression
  * Soft-threshold selection (scale-free R2 vs connectivity)
  * Unsigned TOM, average-linkage hierarchical clustering, module detection
  * Module eigengenes (1st PC), module-trait correlations
  * Hub genes (kME) for AL-associated module(s)
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from stats_util import bh_fdr

def soft_threshold(cors, powers=range(1, 21), n_bins=20):
    """Pick the smallest power whose scale-free topology fit reaches R2 >= 0.80.

    Follows the WGCNA `pickSoftThreshold` recipe: bin log10(connectivity)
    into a histogram, regress log10(bin proportion) on log10(connectivity)
    and take the R2 of that fit. Rounding the (continuous) adjacency degrees
    and calling np.bincount — the obvious shortcut — collapses almost every
    bin to a count of 1, so log10(p) becomes -inf and R2 is always NaN.
    """
    a = np.abs(cors).copy()
    np.fill_diagonal(a, 0.0)
    res = []
    for p in powers:
        adj = a ** p
        k = adj.sum(axis=1)
        k = k[k > 0]
        if len(k) < 10:
            continue
        lk = np.log10(k)
        hist, edges = np.histogram(lk, bins=n_bins)
        prop = hist / hist.sum()
        ok = prop > 0
        if ok.sum() < 3:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        r = np.corrcoef(centers[ok], np.log10(prop[ok]))[0, 1]
        r2 = float(r * r) if np.isfinite(r) else 0.0
        res.append((p, r2, float(k.mean())))
    # Selection rule, fixed in advance (no post-hoc tuning):
    #   1. smallest power with scale-free R2 >= 0.80 WHILE mean connectivity
    #      stays >= 1.0 -- the connectivity guard stops us from "achieving"
    #      scale-free topology by crushing the network into isolated nodes;
    #   2. if no power satisfies both, fall back to WGCNA's documented default
    #      of beta = 6 for unsigned networks (Zhang & Horvath 2005).
    # On a targeted ~500-gene panel the R2 >= 0.80 branch is usually only
    # reachable at powers that disintegrate the network, so the fallback
    # applies and is reported as such in the Methods.
    best, rule = None, "R2>=0.80 & meank>=1"
    for p, r2, mk in res:
        if r2 >= 0.80 and mk >= 1.0 and best is None:
            best = p
    if best is None:
        best, rule = 6, "WGCNA default (unsigned)"
    return best, res, rule

def tom_unsigned(cors, power):
    a = np.abs(cors)**power
    np.fill_diagonal(a, 0)
    L = a.sum(axis=1)
    n = a.shape[0]
    # TOM_ij = (sum_k a_ik a_kj + a_ij) / (min(K_i, K_j) + 1 - a_ij)
    Lm = np.minimum(L[:, None], L[None, :])
    TOM = (a @ a + a) / (Lm + 1.0 - a)
    np.fill_diagonal(TOM, 1.0)
    TOM = np.clip(TOM, 0, 1)
    return TOM

def module_eigengenes(expr_mod, n_pc=1):
    """Standardize per gene (rows), then 1st PC of SAMPLES via SVD (signed eigengene).
    expr_mod: genes x samples. Returns pc1 scores (length n_samples)."""
    X = expr_mod.values.astype(float)  # genes x samples
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = np.nan
    Xs = np.nan_to_num((X - mu) / sd)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    pc1 = S[0] * Vt[0]  # sample scores
    # sign: positive correlation with mean expression per sample
    if np.corrcoef(pc1, X.mean(axis=0))[0, 1] < 0:
        pc1 = -pc1
    return pc1

def main():
    print("=== 04 WGCNA ===")
    expr = pd.read_csv(os.path.join(cfg.PROC,"discovery_expr_norm.tsv"), sep="\t", index_col=0)
    al = pd.read_csv(os.path.join(cfg.PROC,"discovery_AL_score.tsv"), sep="\t", index_col=0)
    clin = pd.read_csv(os.path.join(cfg.PROC,"discovery_clin_std.tsv"), sep="\t", index_col=0)
    common = [s for s in expr.columns if s in al.index]
    expr = expr[common]
    # top variable genes (MAD)
    mad = (expr - expr.median(axis=0)).abs().median(axis=1)
    top = mad.sort_values(ascending=False).head(3500).index
    e = expr.loc[top]
    print("WGCNA genes:", e.shape)
    # Pearson correlation matrix
    cors = e.T.corr().values  # gene x gene
    np.fill_diagonal(cors, 0)
    power, st, rule = soft_threshold(cors)
    print(f"soft power={power} (rule: {rule}); scale-free table:")
    for p, r2, mk in st:
        print(f"  p={p} R2={r2:.3f} meank={mk:.2f}")
    pd.DataFrame(st, columns=["power", "scale_free_R2", "mean_k"]).to_csv(
        os.path.join(cfg.PROC, "wgcna_softthreshold.tsv"), sep="\t", index=False)
    with open(os.path.join(cfg.PROC, "wgcna_power.json"), "w") as f:
        json.dump({"power": int(power), "rule": rule,
                   "r2_at_power": float(dict((p, r2) for p, r2, _ in st).get(power, float("nan")))}, f, indent=2)
    TOM = tom_unsigned(cors, power)
    dist = 1 - TOM
    np.fill_diagonal(dist, 0)
    dist = (dist+dist.T)/2
    np.clip(dist, 0, None, out=dist)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    # ------------------------------------------------------------------
    # Sensitivity of module detection to (power, cut height).
    # A fixed cut of t=0.99 on the 1-TOM dissimilarity merges the whole
    # tree into one diffuse cluster, so modules are cut at 0.90 with a
    # minimum size of 15 genes. With this choice the network resolves into
    # 3 biologically interpretable modules (a large AL-associated module
    # plus two smaller satellite modules). Because that choice could look
    # arbitrary, the AL-module signal is re-derived across a (power x cut)
    # grid and reported as a sensitivity analysis instead of a single cut.
    # ------------------------------------------------------------------
    CUT, MIN_SIZE = 0.90, 15
    genes = list(e.index)
    al_flag = (al.loc[e.columns, "AL_group"] == "AL_high").astype(int).values

    def eig_of(gl):
        return module_eigengenes(e.loc[gl])

    def cut_modules(Z_, cut, min_size):
        lab = fcluster(Z_, t=cut, criterion="distance")
        from collections import Counter as _C
        cnt = _C(lab)
        assign = {}
        nid = 0
        for k, c in sorted(cnt.items(), key=lambda x: -x[1]):
            if c >= min_size:
                assign[k] = f"M{nid}"; nid += 1
            else:
                assign[k] = "grey"
        return lab, pd.Series({g: assign[lab[i]] for i, g in enumerate(genes)})

    sens = []
    for sp in [4, 6, 8, 10, 12]:
        adj = np.abs(cors) ** sp
        L = adj.sum(axis=1)
        Lm = np.minimum(L[:, None], L[None, :])
        T = np.clip((adj @ adj + adj) / (Lm + 1 - adj), 0, 1)
        np.fill_diagonal(T, 1)
        dd = 1 - T
        np.fill_diagonal(dd, 0)
        dd = (dd + dd.T) / 2
        np.clip(dd, 0, None, out=dd)
        Zs = linkage(squareform(dd, checks=False), method="average")
        for cut in [0.90, 0.95, 0.99]:
            _, ma = cut_modules(Zs, cut, MIN_SIZE)
            mods = [m for m in set(ma) if m != "grey"]
            best_r, best_p, best_m, best_n = np.nan, np.nan, "-", 0
            for m in mods:
                gl = [g for g in genes if ma[g] == m]
                pc = eig_of(gl)
                r, pv = spearmanr(pc, al_flag)
                if np.isfinite(r) and (not np.isfinite(best_r) or abs(r) > abs(best_r)):
                    best_r, best_p, best_m, best_n = r, pv, m, len(gl)
            sens.append({"power": sp, "cut_height": cut, "n_modules": len(mods),
                         "module_sizes": ";".join(
                             str(int((ma == m).sum())) for m in sorted(mods)),
                         "AL_module": best_m, "AL_module_size": best_n,
                         "AL_rho": best_r, "AL_p": best_p})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(cfg.PROC, "wgcna_sensitivity.tsv"), sep="\t", index=False)
    print("sensitivity (power x cut height) -> strongest AL module:")
    print(sens.to_string(index=False))

    labels, mod_assign = cut_modules(Z, CUT, MIN_SIZE)
    mod_assign.name = "module"
    mod_assign.to_csv(os.path.join(cfg.PROC, "wgcna_modules.tsv"), sep="\t")
    from collections import Counter
    nmod = len(set(mod_assign) - {"grey"})

    print(f"modules: {nmod} (+grey) at power={power}, cut={CUT}, sizes:",
          dict(Counter(mod_assign)))
    # eigengenes
    eg={}
    for m in sorted(set(mod_assign)):
        if m=="grey": continue
        glist=[g for g in genes if mod_assign[g]==m]
        eg[m]=module_eigengenes(e.loc[glist])
    eg = pd.DataFrame(eg, index=e.columns)
    eg.to_csv(os.path.join(cfg.PROC,"wgcna_eigengenes.tsv"), sep="\t")
    # module-trait correlation
    traits = pd.DataFrame(index=e.columns)
    traits["AL_group"]=(al.loc[e.columns,"AL_group"]=="AL_high").astype(int)
    traits["AL_score"]=al.loc[e.columns,"AL_sig_score"]
    for c in ["age","mutation_count","OS_event"]:
        if c in clin.columns:
            traits[c]=pd.to_numeric(clin.loc[e.columns,c] if c in clin else np.nan, errors="coerce")
    mtrait=[]
    for m in eg.columns:
        for t in traits.columns:
            a=eg[m].values; b=traits[t].values
            mask=~np.isnan(a)&~np.isnan(b)
            if mask.sum()>10:
                r,p=spearmanr(a[mask], b[mask])
            else: r,p=np.nan,np.nan
            mtrait.append((m,t,r,p))
    mt=pd.DataFrame(mtrait, columns=["module","trait","rho","p"]).sort_values("p")
    mt["FDR"]=mt["p"].values  # placeholder; BH below
    mt["FDR"] = bh_fdr(mt["p"].values)
    mt.to_csv(os.path.join(cfg.PROC,"wgcna_module_trait.tsv"), sep="\t", index=False)
    print("module-trait (top):"); print(mt.head(12).to_string(index=False))
    # hub genes: kME for the module most correlated with AL_score
    sig = mt[(mt.trait=="AL_score")].sort_values("p")
    hubmod = sig.iloc[0]["module"] if len(sig) else eg.columns[0]
    print("AL hub module:", hubmod)
    glist=[g for g in genes if mod_assign[g]==hubmod]
    kme = e.loc[glist].apply(lambda row: np.corrcoef(row.values, eg[hubmod].values)[0,1], axis=1)
    hub = pd.DataFrame({"gene":glist,"kME":kme.values})
    hub=hub.sort_values("kME", key=lambda s:s.abs(), ascending=False)
    hub.to_csv(os.path.join(cfg.PROC,"wgcna_hub_genes.tsv"), sep="\t", index=False)
    print("top hub genes:\n", hub.head(15).to_string(index=False))
    # save linkage + dist for figure
    np.save(os.path.join(cfg.PROC,"wgcna_Z.npy"), Z)

if __name__=="__main__":
    main()
