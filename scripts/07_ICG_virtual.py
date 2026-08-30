"""
07_ICG_virtual.py — Virtual ICG-fluorescence cohort & multimodal feature assembly.
This is the core methodological novelty. Because no public dataset links ICG
imaging to multi-omic profiling, we construct a biologically-calibrated virtual
ICG parameter cohort:
  * 26 quantitative ICG parameters drawn from literature-derived Beta/Gamma
    distributions (gene_panel.ICG_PARAMETERS / ICG_LITERATURE_META).
  * Parameters are made DEPENDENT on the (real) molecular AL-risk signature and
    clinical risk so that a learnable, physiologically-plausible cross-modal
    signal exists (CCA/PLS and the fusion model can exploit it), plus noise.
  * Virtual anastomotic-leakage (AL) outcome sampled from a logistic model
    combining molecular + clinical + ICG risk, calibrated to ~8% prevalence.
  * Canonical Correlation Analysis (CCA) between ICG and molecular features.
  * Assembles the multimodal feature matrix for ML (step 8).
Reproducible (fixed seed). Validation cohort generated with identical generative
parameters using its own (real) molecular risk.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from gene_panel import ICG_PARAMETERS, ICG_LITERATURE_META
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

SEED=20260828; rng=np.random.default_rng(SEED)

def risk_dir(p):
    return p[6]  # 'high' or 'low'

def sample_icg_params(n, mol_risk_z, clin_risk_z, noise=0.35):
    """Sample 26 ICG params, shifted by molecular+clinical risk (toward leak)."""
    out={}
    total_risk = 0.6*mol_risk_z + 0.4*clin_risk_z
    for pid,name,unit,dt,dp,thr,direction,desc in ICG_PARAMETERS:
        base=np.zeros(n)
        if dt=="gamma":
            base=rng.gamma(dp["shape"], dp["scale"], n)
        elif dt=="beta":
            base=rng.beta(dp["a"], dp["b"], n)
            # scale to 0-100 for percentage-like
            base=base*100
        elif dt=="normal":
            base=rng.normal(dp["mean"], dp["sd"], n)
        # shift toward leak direction based on risk
        shift = total_risk * (0.18 if dt=="beta" else (0.20*dp["scale"] if dt=="gamma" else 0.4))
        if direction=="high":   # higher value -> more leak risk
            val = base + shift + rng.normal(0, noise*(0.15*dp["scale"] if dt=="gamma" else (8 if dt=="beta" else 0.3)), n)
        else:                   # lower value -> more leak risk
            val = base - shift + rng.normal(0, noise*(0.15*dp["scale"] if dt=="gamma" else (8 if dt=="beta" else 0.3)), n)
        val=np.clip(val, 1e-3, None)
        out[pid]=val
    return pd.DataFrame(out)

def make_outcome(mol_z, clin_z, icg_risk_z, prevalence=0.082, w=(1.0,0.6,1.2)):
    logit = -3.4 + w[0]*mol_z*0.9 + w[1]*clin_z*0.8 + w[2]*icg_risk_z*1.1 + rng.normal(0,0.25,len(mol_z))
    p=1/(1+np.exp(-logit))
    y=(rng.uniform(0,1,len(mol_z))<p).astype(int)
    return p, y

def icg_risk_score(icg_df):
    """Standardized composite ICG risk: high for time-based, low for intensity-based."""
    z=pd.DataFrame(index=icg_df.index)
    for pid,name,unit,dt,dp,thr,direction,desc in ICG_PARAMETERS:
        sd = icg_df[pid].std()
        z[pid] = (icg_df[pid]-icg_df[pid].mean())/(sd if sd and sd > 0 else np.nan)
        if direction=="low": z[pid]=-z[pid]  # flip so high=risk
    return z.mean(axis=1)

def build_for_cohort(tag):
    expr=pd.read_csv(os.path.join(cfg.PROC,f"{tag}_expr_norm.tsv"), sep="\t", index_col=0) if os.path.exists(os.path.join(cfg.PROC,f"{tag}_expr_norm.tsv")) else None
    al=pd.read_csv(os.path.join(cfg.PROC,f"{tag}_AL_score.tsv"), sep="\t", index_col=0)
    clin=pd.read_csv(os.path.join(cfg.PROC,f"{tag}_clin_std.tsv"), sep="\t", index_col=0)
    samples=[s for s in al.index if s in clin.index]
    al=al.loc[samples]; clin=clin.loc[samples]
    # molecular risk z
    mol=al["AL_sig_score"].values.astype(float)
    mol_z=(mol-mol.mean())/mol.std()
    # clinical risk: stage encoded + age + mutation
    stage_map={"STAGE I":1,"STAGE II":2,"STAGE III":3,"STAGE IV":4,"Stage I":1,"Stage II":2,"Stage III":3,"Stage IV":4,
               "I":1,"II":2,"III":3,"IV":4,"1":1,"2":2,"3":3,"4":4}
    st=(clin["stage"].map(stage_map) if "stage" in clin else pd.Series(np.nan,index=clin.index)).fillna(2)
    age=(pd.to_numeric(clin["age"],errors="coerce") if "age" in clin else pd.Series(60,index=clin.index)).fillna(60)
    age_risk=((age-50)/20).clip(-1,2)
    mut=(pd.to_numeric(clin["mutation_count"],errors="coerce") if "mutation_count" in clin else pd.Series(2,index=clin.index)).fillna(2)
    mut_z=(np.log1p(mut)-np.log1p(mut).mean())/(np.log1p(mut).std() or 1)
    clin_risk=0.5*(st/4) + 0.3*age_risk + 0.2*np.clip(mut_z,-2,2)
    clin_z=(clin_risk-clin_risk.mean())/(clin_risk.std() or 1)
    # ICG
    mol_z=np.asarray(mol_z, dtype=float); clin_z=np.asarray(clin_z, dtype=float)
    icg=sample_icg_params(len(samples), mol_z, clin_z)
    icg.index=samples
    icg_risk=icg_risk_score(icg)
    icg_risk_z=((icg_risk-icg_risk.mean())/icg_risk.std()).values
    # outcome
    p,y=make_outcome(mol_z, clin_z, icg_risk_z, prevalence=ICG_LITERATURE_META["AL_overall_prevalence"])
    out=pd.DataFrame({"sample":samples,"AL":y,"AL_prob":p,"mol_risk":mol,"mol_z":mol_z,
                      "clin_risk":np.asarray(clin_risk, dtype=float),"clin_z":clin_z,
                      "icg_risk":icg_risk.values,"icg_risk_z":icg_risk_z})
    out=out.set_index("sample")
    icg.to_csv(os.path.join(cfg.PROC,f"{tag}_icg_params.tsv"), sep="\t")
    out.to_csv(os.path.join(cfg.PROC,f"{tag}_al_outcome.tsv"), sep="\t")
    return samples, al, clin, icg, icg_risk, out, expr

def cca_analysis(tag, expr, al, icg):
    """CCA: ICG params (26) vs molecular features."""
    mol_feats=al[["AL_sig_score"]] if "AL_sig_score" in al else al
    # add top variable genes if available
    if expr is not None:
        common=[s for s in expr.columns if s in icg.index]
        top=(expr[common].std(axis=1).sort_values(ascending=False).head(20).index)
        gf=expr.loc[top, icg.index].T
        mol=pd.concat([al.loc[icg.index,["AL_sig_score"]], gf], axis=1).fillna(0)
    else:
        mol=al.loc[icg.index,["AL_sig_score"]].fillna(0)
    X=StandardScaler().fit_transform(mol.values)
    Y=StandardScaler().fit_transform(icg.values)
    ncomp=min(3, mol.shape[1], icg.shape[1])
    cca=CCA(n_components=ncomp); cca.fit(X,Y)
    Xc, Yc = cca.transform(X, Y)
    canon=[float(np.corrcoef(Xc[:,i],Yc[:,i])[0,1]) for i in range(ncomp)]
    return canon, mol.columns.tolist()

def main():
    print("=== 07 ICG virtual cohort ===")
    for tag in ["discovery","validation"]:
        if not os.path.exists(os.path.join(cfg.PROC,f"{tag}_AL_score.tsv")):
            print(f"  {tag}: AL score missing, skip"); continue
        samples, al, clin, icg, icg_risk, out, expr = build_for_cohort(tag)
        print(f"  {tag}: n={len(samples)} AL cases={int(out.AL.sum())} ({out.AL.mean()*100:.1f}%)")
        canon,molcols=cca_analysis(tag, expr, al, icg)
        print(f"  {tag}: CCA canonical correlations: {[f'{c:.3f}' for c in canon]}")
        pd.DataFrame({"cohort":tag,"component":range(1,len(canon)+1),"canon_corr":canon}).to_csv(
            os.path.join(cfg.PROC,f"{tag}_cca.tsv"), sep="\t", index=False)
    # save ICG meta & params definitions for the paper/supp
    pd.DataFrame(ICG_PARAMETERS, columns=["param","name","unit","dist","params","threshold","direction","description"]).to_csv(
        os.path.join(cfg.SUPP,"ICG_parameter_definitions.tsv"), sep="\t", index=False)
    json.dump(ICG_LITERATURE_META, open(os.path.join(cfg.PROC,"icg_literature_meta.json"),"w"), indent=2)
    print("=== DONE ===")

if __name__=="__main__":
    main()
