"""
05_enrichment.py — Functional enrichment for AL-high vs AL-low DEGs.
  * Over-representation analysis (ORA) on curated pathway gene sets (self-contained,
    hypergeometric) — robust, no external API needed.
  * Also attempts gseapy enrichr (GO_BP / KEGG / Reactome) if the Enrichr API is
    reachable; results saved if successful.
  * Preranked GSEA on the DEG log2FC using the curated pathway sets (weighted
    Kolmogorov-Smirnov) — self-contained.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from scipy.stats import hypergeom
from gene_panel import PANEL_GROUPS
from stats_util import bh_fdr as bh

# ---- curated pathway gene sets (from panel + biology) ----------------------
def build_pathways():
    P={}
    P["Extracellular_matrix_organization"]=set(PANEL_GROUPS["ECM_collagen"])
    P["Collagen_chain_trimerization"]={g for g in PANEL_GROUPS["ECM_collagen"] if g.startswith("COL")}
    P["Matrix_metalloproteinase_activity"]=set(PANEL_GROUPS["MMP_ADP"])
    P["Inflammatory_response"]=set(PANEL_GROUPS["Cytokines"])|set(PANEL_GROUPS["Chemokines"])
    P["Cytokine_signaling"]=set(PANEL_GROUPS["Cytokines"])
    P["Chemokine_signaling"]=set(PANEL_GROUPS["Chemokines"])
    P["Hypoxia_response"]=set(PANEL_GROUPS["Hypoxia"])
    P["Angiogenesis"]={"VEGFA","VEGFB","VEGFC","VEGFD","KDR","FLT1","FLT4","PECAM1","CDH5","ANGPT1","ANGPT2","TEK","PDGFB","DLL4","NOS3","VWF","CD34","ESAM"}
    P["EMT_mesenchymal"]=set(PANEL_GROUPS["EMT"])
    P["TGFbeta_signaling"]=set(PANEL_GROUPS["CSF_growth"])&set(PANEL_GROUPS["ECM_collagen"])|{"TGFB1","TGFB2","TGFB3","TGFBR1","TGFBR2","TGFBR3","SMAD2","SMAD3","SMAD4","SMAD7","INHBA"}
    P["Wound_healing"]={"TGFB1","TGFB3","PDGFA","PDGFB","VEGFA","VEGFB","FGF2","IGF1","EGF","HGF","CTGF","COL1A1","COL3A1","FN1","VEGFC","KDR","FLT1","PECAM1","CDH1","TIMP1","TIMP2","TIMP3","LOX","PDGFRB","ACTA2"}
    P["Inflammasome_pyroptosis"]=set(PANEL_GROUPS["Inflammasome"])
    P["T_cell_activation"]={"CD3D","CD3E","CD3G","CD4","CD8A","CD8B","GZMA","GZMB","GZMH","GZMK","PRF1","NKG7","IFNG","IL2","TNFRSF9","ICOS","CD28"}
    P["Macrophage"]={"CD14","CD68","CD163","MRC1","CSF1R","ITGAM","ITGAX","FCGR1A","FCGR3A","C1QA","C1QB","C1QC","MS4A7","LST1","FCER1G","AIF1"}
    P["B_cell"]={"CD19","MS4A1","CD79A","CD79B","CD22","CD19","BANK1","BLK","TCL1A","VPREB3","CD24","CD72","FCRL5"}
    P["NK_cell"]={"NCAM1","KLRD1","KLRF1","NKG7","GNLY","KIR2DL1","KIR2DL3","KIR3DL1","KIR3DL2","NCR1","NCR3","PRF1","GZMB"}
    P["Treg"]={"FOXP3","IL2RA","CTLA4","TIGIT","LAG3","HAVCR2","CCR8","TNFRSF9","IKZF2"}
    P["Neutrophil_NETs"]={"MPO","ELANE","CTSG","PRTN3","LCN2","S100A8","S100A9","S100A12","PADI4","CEACAM3","CEACAM8"}
    P["Ferroptosis"]=set(PANEL_GROUPS["Ferroptosis"])
    P["Oxidative_stress"]=set(PANEL_GROUPS["Oxidative"])
    P["WNT_signaling"]=set(PANEL_GROUPS["WNT_Notch"])&{g for g in PANEL_GROUPS["WNT_Notch"] if g.startswith("WNT") or g.startswith("FZD") or g in ("LRP5","LRP6","AXIN2","CTNNB1")}
    P["CRC_hallmark"]=set(PANEL_GROUPS["CRC_hallmark"])
    P["Antigen_presentation"]=set(PANEL_GROUPS["HLA_antigen"])
    P["Tight_junction"]=set(PANEL_GROUPS["Adhesion_TJ"])
    return {k:set(g.upper() for g in v) for k,v in P.items()}

def hyper_ora(test_set, universe, pathway):
    # over-representation (one-sided)
    k=len(test_set & pathway); n=len(test_set); K=len(pathway); N=len(universe)
    if n==0 or K==0 or N==0: return 0,1.0
    p=hypergeom.sf(k-1, N, K, n)
    return k, p

def _es_from_scores(rv, mask, weight=1.0):
    """Weighted running-sum enrichment score (classic GSEA, pre-ranked form)."""
    n = len(rv)
    k = int(mask.sum())
    if k < 3 or k == n:
        return 0.0
    wpos = np.abs(rv) ** weight
    nr = wpos[mask].sum()
    if nr <= 0:
        return 0.0
    inc = np.where(mask, wpos / nr, -1.0 / (n - k))
    run = np.cumsum(inc)
    mx, mn = run.max(), run.min()
    return float(mx if abs(mx) >= abs(mn) else mn)


def gsea_preranked(ranking, pathways, weight=1.0, n_perm=1000, seed=42):
    """Pre-ranked GSEA with a real permutation null.

    Gene labels are permuted (the ranking is shuffled while pathway membership
    stays fixed), which is the standard null for a pre-ranked list. ES is
    normalised against the permutation mean of the same sign to give NES, and
    the nominal p and FDR follow the original GSEA procedure.
    """
    r = ranking.sort_values(ascending=False)
    genes = list(r.index)
    rv = r.values.astype(float)
    n = len(genes)
    rng = np.random.default_rng(seed)

    masks, sizes = {}, {}
    for name, gset in pathways.items():
        m = np.array([g in gset for g in genes], dtype=bool)
        masks[name] = m
        sizes[name] = int(m.sum())

    obs = {name: _es_from_scores(rv, m, weight) for name, m in masks.items()}

    # permutation null: shuffle the ranking, keep membership fixed
    null = {name: np.empty(n_perm) for name in masks}
    for b in range(n_perm):
        rp = rng.permutation(rv)
        for name, m in masks.items():
            null[name][b] = _es_from_scores(rp, m, weight)

    rows = []
    for name, m in masks.items():
        es = obs[name]
        nl = null[name]
        pos_mean = nl[nl >= 0].mean() if (nl >= 0).any() else np.nan
        neg_mean = abs(nl[nl < 0].mean()) if (nl < 0).any() else np.nan
        if es >= 0:
            denom = pos_mean if (np.isfinite(pos_mean) and pos_mean > 0) else np.nan
            p = (np.sum(nl >= es) + 1) / (n_perm + 1) if n_perm else np.nan
        else:
            denom = neg_mean if (np.isfinite(neg_mean) and neg_mean > 0) else np.nan
            p = (np.sum(nl <= es) + 1) / (n_perm + 1) if n_perm else np.nan
        nes = es / denom if (np.isfinite(denom) and denom > 0) else np.nan
        rows.append({"pathway": name, "size": sizes[name], "ES": es,
                     "NES": nes, "pval": p, "null_mean_pos": pos_mean,
                     "null_mean_neg": neg_mean})

    res = pd.DataFrame(rows)

    # GSEA-style FDR. The permutation ES must be normalised the same way as
    # the observed ES before the two tails are compared -- pooling raw null ES
    # and testing it against a normalised NES (the classic mistake) drives
    # every q-value to 0.
    if n_perm:
        pooled = []
        for name in masks:
            nl = null[name]
            d = res.loc[res.pathway == name, "NES"].values
            if not len(d) or not np.isfinite(d[0]):
                continue
            pm = res.loc[res.pathway == name, "null_mean_pos"].values[0]
            nm = res.loc[res.pathway == name, "null_mean_neg"].values[0]
            nn = np.where(nl >= 0,
                          nl / pm if (np.isfinite(pm) and pm > 0) else np.nan,
                          nl / nm if (np.isfinite(nm) and nm > 0) else np.nan)
            pooled.append(nn[np.isfinite(nn)])
        pooled = np.concatenate(pooled) if pooled else np.array([])
        obs_nes = res["NES"].values.astype(float)
        for i in res.index:
            nes = res.at[i, "NES"]
            if not np.isfinite(nes) or not len(pooled):
                res.at[i, "FDR"] = np.nan
                continue
            same = pooled > 0 if nes >= 0 else pooled < 0
            f_null = np.mean(np.abs(pooled[same]) >= abs(nes)) if same.any() else 0.0
            same_o = obs_nes > 0 if nes >= 0 else obs_nes < 0
            f_obs = np.mean(np.abs(obs_nes[same_o]) >= abs(nes)) if same_o.any() else 0.0
            res.at[i, "FDR"] = float(np.clip(f_null / f_obs, 0, 1)) if f_obs > 0 else np.nan
    else:
        res["FDR"] = np.nan
    return res

def main():
    print("=== 05 enrichment ===")
    deg=pd.read_csv(os.path.join(cfg.PROC,"discovery_DEG.tsv"), sep="\t")
    PW=build_pathways()
    universe=set(deg.gene)
    up=set(deg[deg.sig=="Up"].gene); dn=set(deg[deg.sig=="Down"].gene)
    rows=[]
    for name,pset in PW.items():
        for label,test in [("Up",up),("Down",dn)]:
            k,p=hyper_ora(test, universe, pset & universe)
            rows.append((name,label,k,len(test),len(pset&universe),p))
    res=pd.DataFrame(rows, columns=["pathway","direction","overlap","DEG_size","pathway_size","p"])
    res["FDR"]=bh(res["p"].values)
    res=res.sort_values("p")
    res.to_csv(os.path.join(cfg.PROC,"enrichment_ORA.tsv"), sep="\t", index=False)
    print("ORA top:\n", res.head(15).to_string(index=False))
    # GSEA preranked
    rank=deg.set_index("gene")["log2FC"]
    gsea=gsea_preranked(rank, PW)
    gsea=gsea.sort_values("NES", key=lambda s:s.abs(), ascending=False)
    gsea.to_csv(os.path.join(cfg.PROC,"enrichment_GSEA.tsv"), sep="\t", index=False)
    print("GSEA (1000 permutations):\n", gsea.head(12).to_string(index=False))
    # try gseapy enrichr (online, optional)
    try:
        import gseapy
        for label,test in [("up",up),("down",dn)]:
            if len(test)<5: continue
            try:
                enr=gseapy.enrichr(gene_list=list(test), organism="human",
                                   gene_sets=["GO_Biological_Process_2023",
                                              "KEGG_2021_Human","Reactome_2022"])
            except TypeError:
                # older/newer gseapy signatures differ; drop unsupported kwargs
                enr=gseapy.enrichr(gene_list=list(test), organism="human",
                                   gene_sets=["GO_Biological_Process_2023"])
            if enr and hasattr(enr,"results"):
                enr.results.to_csv(os.path.join(cfg.PROC,f"enrichr_{label}.tsv"), sep="\t", index=False)
        print("gseapy enrichr: done (if reachable)")
    except Exception as e:
        print(f"gseapy enrichr skipped: {type(e).__name__}: {str(e)[:100]}")

if __name__=="__main__":
    main()
