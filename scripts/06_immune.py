"""
06_immune.py — Immune microenvironment characterization.
  * ssGSEA for 22 immune cell types (Bindea/Charoentong signatures, panel-restricted)
  * Stromal / immune ESTIMATE-like scores (ssGSEA of stromal & immune signatures)
  * Inflammatory cytokine expression comparison AL-high vs AL-low
  * Group comparisons + correlations with AL signature score
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from scipy.stats import ranksums, mannwhitneyu, spearmanr
from gene_panel import PANEL_GROUPS
from stats_util import bh_fdr

IMMUNE_SIGS = {
 "Activated_B_cell":{"MS4A1","CD19","CD79A","CD79B","BLK","TCL1A"},
 "Activated_CD8_T_cell":{"CD8A","CD8B","GZMA","GZMB","GZMK","PRF1","NKG7","IFNG"},
 "Activated_dendritic_cell":{"CD1C","CLEC10A","FCER1A","ITGAX","CD83","LAMP3","CCL19"},
 "CD56bright_NK":{"NCAM1","KLRF1","KLRD1","NKG7","GZMB"},
 "Central_memory_T":{"CCR7","SELL","IL7R","TCF7","LEF1","CD27"},
 "Effector_memory_T":{"GZMK","GZMA","CX3CR1","KLRG1"},
 "Gamma_delta_T":{"TRGC1","TRDC","TRGV9"},
 "Macrophage":{"CD14","CD68","CD163","MRC1","CSF1R","ITGAM","FCGR1A","C1QA","C1QB","MS4A7"},
 "Mast_cell":{"TPSAB1","TPSB2","CPA3","MS4A2","KIT"},
 "MDSC":{"ITGAM","CD33","CEACAM8","S100A8","S100A9","ARG1"},
 "Monocyte":{"CD14","FCGR3A","S100A8","S100A9","FCN1","VCAN"},
 "Neutrophil":{"MPO","ELANE","CTSG","PRTN3","LCN2","FCGR3B","CSF3R","S100A12"},
 "NK_cell":{"NCAM1","KLRD1","KLRF1","NKG7","GNLY","PRF1","GZMB","KIR2DL3"},
 "Plasmacytoid_DC":{"LILRA4","IL3RA","CLEC4C","IRF7","TLR7","GZMB"},
 "Regulatory_T_cell":{"FOXP3","IL2RA","CTLA4","TIGIT","LAG3","HAVCR2","TNFRSF9","IKZF2"},
 "T follicular_helper":{"CXCR5","ICOS","BCL6","PDCD1","CD200"},
 "Type_1_T_helper":{"IFNG","TBX21","CXCR3","IL12RB2","STAT1"},
 "Type_2_T_helper":{"IL4","IL5","IL13","GATA3","STAT6"},
 "Type_17_T_helper":{"IL17A","IL17F","IL22","RORC","IL23R","CCR6"},
 "Activated_T":{"CD3D","CD3E","CD3G","CD4","CD8A","CD28","TNFRSF9","ICOS"},
 "Endothelial":{"PECAM1","VWF","CDH5","KDR","FLT1","ESAM","CD34","NOS3"},
 "Fibroblast":{"ACTA2","TAGLN","PDGFRB","PDGFRA","FAP","COL1A1","COL3A1","DCN","LUM","POSTN"},
 "Pericyte":{"PDGFRB","CSPG4","RGS5","NOTCH3"},
}
STROMAL_SIG={"ACTA2","TAGLN","PDGFRB","FAP","COL1A1","COL1A2","COL3A1","COL5A1","COL6A1","DCN","LUM","POSTN","FN1","SPARC","THBS2","TNC","PDGFRA","PDGFRB"}
IMMUNE_SCORE_SIG=set().union(*[set(IMMUNE_SIGS[k]) for k in IMMUNE_SIGS]) | set(PANEL_GROUPS["Immune_markers"]) | set(PANEL_GROUPS["HLA_antigen"])
INFLAMM_CYTOKINES=["IL1A","IL1B","IL6","TNF","CXCL8","CXCL1","CXCL2","CCL2","IL18","IL17A","IFNG","CSF1","CSF2","IL12A","IL12B","IL33","NLRP3","GSDMD","CASP1"]

def ssgsea(expr, signatures):
    """expr: gene x sample (log2). Returns sample x signature ssGSEA scores."""
    # rank genes per sample (descending)
    out={}
    for signame,gset in signatures.items():
        gset=[g for g in gset if g in expr.index]
        if len(gset)<2: continue
        scores=[]
        for s in expr.columns:
            v=expr[s].sort_values(ascending=False)
            n=len(v)
            in_set=np.array([g in gset for g in v.index])
            # Missing values are treated as "at the mean" (0 for z-scored
            # input) rather than propagated: a NaN would poison the running
            # sum and turn the whole sample score into NaN.
            vals=np.nan_to_num(v.values.astype(float), nan=0.0)
            # weighted KS: hits are weighted by |expression|, misses by 1/miss
            phit=float(np.abs(vals[in_set]).sum())
            if not np.isfinite(phit) or phit<=0:
                phit=1.0
            pmiss=max(int((~in_set).sum()), 1)
            inc=np.where(in_set, np.abs(vals)/phit, -1.0/pmiss)
            cum=np.cumsum(inc)
            es=float(cum.max() if abs(cum.max())>abs(cum.min()) else cum.min())
            scores.append(es)
        out[signame]=scores
    df=pd.DataFrame(out, index=expr.columns)
    return df

def main():
    print("=== 06 immune ===")
    expr=pd.read_csv(os.path.join(cfg.PROC,"discovery_expr_norm.tsv"), sep="\t", index_col=0)
    al=pd.read_csv(os.path.join(cfg.PROC,"discovery_AL_score.tsv"), sep="\t", index_col=0)
    common=[s for s in expr.columns if s in al.index]
    expr=expr[common]
    # ssGSEA
    imm=ssgsea(expr, IMMUNE_SIGS)
    imm.to_csv(os.path.join(cfg.PROC,"immune_ssgsea.tsv"), sep="\t")
    print("ssGSEA cells:", imm.shape)
    # estimate-like
    est=ssgsea(expr, {"Stromal":STROMAL_SIG,"Immune":IMMUNE_SCORE_SIG})
    est.to_csv(os.path.join(cfg.PROC,"immune_estimate.tsv"), sep="\t")
    # group comparisons
    grp=al.loc[common,"AL_group"]
    hi=[s for s in common if grp[s]=="AL_high"]; lo=[s for s in common if grp[s]=="AL_low"]
    comp=[]
    for c in imm.columns:
        a=imm.loc[hi,c].values; b=imm.loc[lo,c].values
        try: st,p=mannwhitneyu(a,b,alternative="two-sided")
        except: st,p=np.nan,np.nan
        comp.append((c, a.mean(), b.mean(), a.mean()-b.mean(), p))
    comp=pd.DataFrame(comp, columns=["cell","mean_ALhigh","mean_ALlow","diff","p"]).sort_values("p")
    comp["FDR"] = bh_fdr(comp.p.values)
    comp.to_csv(os.path.join(cfg.PROC,"immune_group_comparison.tsv"), sep="\t", index=False)
    print("immune comparison (top):\n", comp.head(10).to_string(index=False))
    # correlation with AL score
    score=al.loc[common,"AL_sig_score"]
    corr=[]
    for c in imm.columns:
        r,p=spearmanr(imm[c].values, score.values)
        corr.append((c,r,p))
    corr=pd.DataFrame(corr, columns=["cell","rho","p"]).sort_values("p")
    corr.to_csv(os.path.join(cfg.PROC,"immune_AL_correlation.tsv"), sep="\t", index=False)
    # inflammatory cytokines
    cy=[g for g in INFLAMM_CYTOKINES if g in expr.index]
    cdf=expr.loc[cy].T
    cdf.to_csv(os.path.join(cfg.PROC,"immune_cytokines.tsv"), sep="\t")
    print("cytokines:", cdf.shape)

if __name__=="__main__":
    main()
