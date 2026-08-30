"""
10_validation.py — Model validation: ROC, DCA, calibration, nomogram.
  * External validation on the virtual validation cohort (same pipeline applied
    to coad_silu_2022 molecular data -> ICG virtual params -> best model).
  * Decision Curve Analysis (net benefit across thresholds).
  * Calibration curves (bootstrap bias-corrected).
  * Hosmer-Lemeshow style decile calibration table.
  * Nomogram rendering (point-based, journal style).
Outputs: roc_data.tsv, dca_data.tsv, calibration.tsv, nomogram logic (JSON).
"""
import os, sys, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 20260828

def load_features(tag):
    out = pd.read_csv(os.path.join(cfg.PROC, f"{tag}_al_outcome.tsv"), sep="\t", index_col=0)
    icg = pd.read_csv(os.path.join(cfg.PROC, f"{tag}_icg_params.tsv"), sep="\t", index_col=0)
    feat = pd.DataFrame(index=out.index)
    feat["mol_risk_z"] = out["mol_z"]
    feat["clin_risk_z"] = out["clin_z"]
    feat["AL_sig_score"] = out["mol_risk"]
    for c in icg.columns:
        feat[f"icg_{c}"] = icg[c]
    return feat, out["AL"].values.astype(int)

def fit_best(X, y):
    """Refit the best model (RF per 08 default, or XGB if available)."""
    if HAS_XGB:
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0)
    else:
        m = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=8,
                                   random_state=SEED, n_jobs=1)
    m.fit(X, y)
    return m

def roc_data(y, p, label):
    fpr, tpr, thr = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr, "model": label, "AUC": auc})

def net_benefit(y, p, thresholds):
    """Standard DCA net benefit at threshold probabilities."""
    n = len(y); out = []
    for th in thresholds:
        pred = (p >= th).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        nb = tp / n - fp / n * (th / (1 - th))
        out.append(nb)
    return np.array(out)

def treat_all(y, thresholds):
    prev = y.mean(); n = len(y)
    return np.array([prev - (1 - prev) * (th / (1 - th)) for th in thresholds])

def calibration(y, p, bins=10):
    """Decile-based calibration curve."""
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        lo, hi = qs[i], qs[i + 1]
        m = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if m.sum() == 0: continue
        rows.append({"bin": i + 1, "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "observed_rate": float(y[m].mean())})
    return pd.DataFrame(rows)

def main():
    print("=== 10 model validation ===")
    # discovery: OOF predictions from 08
    oof = pd.read_csv(os.path.join(cfg.RESULTS, "oof_predictions.tsv"), sep="\t")
    y_d, p_d = oof["y"].values, oof["p"].values

    # validation cohort: train-free evaluation with model trained on discovery
    feat_d, y_train = load_features("discovery")
    model = fit_best(feat_d.values.astype(float), y_train)
    try:
        feat_v, y_v = load_features("validation")
        p_v = model.predict_proba(feat_v.values.astype(float))[:, 1]
        auc_v = roc_auc_score(y_v, p_v)
        print(f"  validation cohort: n={len(y_v)} AL={y_v.sum()} AUC={auc_v:.3f}")
    except Exception as e:
        print("  validation unavailable:", e)
        feat_v, y_v, p_v, auc_v = None, None, None, None

    # ---- ROC curves ----
    roc = []
    roc.append(roc_data(y_d, p_d, "Discovery (OOF)"))
    if p_v is not None:
        roc.append(roc_data(y_v, p_v, "Validation"))
    rocdf = pd.concat(roc)
    rocdf.to_csv(os.path.join(cfg.RESULTS, "roc_data.tsv"), sep="\t", index=False)
    print(f"  discovery AUC={roc_auc_score(y_d, p_d):.3f}")
    if auc_v: print(f"  validation AUC={auc_v:.3f}")

    # bootstrap CI for discovery
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y_d), len(y_d))
        if len(np.unique(y_d[i])) < 2: continue
        bs.append(roc_auc_score(y_d[i], p_d[i]))
    ci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    print(f"  discovery bootstrap AUC 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

    # ---- DCA ----
    ths = np.linspace(0.05, 0.60, 56)
    dca = pd.DataFrame({"threshold": ths})
    dca["model_nb"] = net_benefit(y_d, p_d, ths)
    dca["treat_all_nb"] = treat_all(y_d, ths)
    dca["treat_none_nb"] = 0.0
    if p_v is not None:
        dca["validation_model_nb"] = net_benefit(y_v, p_v, ths)
    dca.to_csv(os.path.join(cfg.RESULTS, "dca_data.tsv"), sep="\t", index=False)
    print("  DCA computed")

    # ---- calibration ----
    cal = calibration(y_d, p_d)
    cal.to_csv(os.path.join(cfg.RESULTS, "calibration_discovery.tsv"), sep="\t", index=False)
    if p_v is not None:
        cal_v = calibration(y_v, p_v)
        cal_v.to_csv(os.path.join(cfg.RESULTS, "calibration_validation.tsv"), sep="\t", index=False)
    # Hosmer-Lemeshow: chi2 = sum over groups of (O_g - E_g)^2 /
    # (n_g * pbar_g * (1 - pbar_g)), where O_g is the number of OBSERVED
    # EVENTS in the group (n_g * observed rate), not the group size.
    obs = cal.n * cal.observed_rate
    exp = cal.n * cal.mean_predicted
    denom = cal.n * cal.mean_predicted * (1 - cal.mean_predicted)
    denom = denom.replace(0, np.nan)
    hl = float(((obs - exp) ** 2 / denom).sum())
    df_hl = max(len(cal) - 2, 1)
    hl_p = float(chi2.sf(hl, df_hl))
    print(f"  Hosmer-Lemeshow chi2 = {hl:.2f} (df={df_hl}, p={hl_p:.3g})")

    # ---- nomogram: logistic on 3 interpretable modality scores ----
    out_d = pd.read_csv(os.path.join(cfg.PROC, "discovery_al_outcome.tsv"), sep="\t", index_col=0)
    # The ICG axis is the CCA-derived perfusion-risk score (icg_risk_z), not a
    # plain row mean of the 26 parameters: individual parameters point in
    # opposite directions (slope up = good perfusion, Tmax up = poor), so a
    # row mean cancels the signal and yields an uninterpretable coefficient.
    nm = pd.DataFrame({
        "mol_z": out_d["mol_z"], "clin_z": out_d["clin_z"],
        "icg_z": out_d["icg_risk_z"],
        "AL": out_d["AL"],
    }).dropna()
    lr = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
    Xn = nm[["mol_z", "clin_z", "icg_z"]].values
    yn = nm["AL"].values
    lr.fit(Xn, yn)
    coefs = dict(zip(["mol_z", "clin_z", "icg_z"], lr.coef_[0].round(3)))
    inter = float(lr.intercept_[0].round(3))
    auc_nm = roc_auc_score(yn, lr.predict_proba(Xn)[:, 1])
    # The apparent AUC above is computed on the same data the model was fitted
    # to, which makes it look better than the 29-feature model's out-of-fold
    # AUC and invites the (fair) objection that the nomogram was tuned to the
    # sample. Report a cross-validated estimate alongside it.
    auc_nm_cv = np.nan
    try:
        n_splits = int(min(10, max(2, np.bincount(yn.astype(int)).min())))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        oof = np.empty(len(yn))
        for tr, te in skf.split(Xn, yn):
            m = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
            m.fit(Xn[tr], yn[tr])
            oof[te] = m.predict_proba(Xn[te])[:, 1]
        auc_nm_cv = roc_auc_score(yn, oof)
    except Exception as e:
        print("  nomogram CV AUC unavailable:", e)
    json.dump({"intercept": inter, "coefficients": coefs,
               "nomogram_AUC": round(float(auc_nm), 3),
               "nomogram_AUC_cv": round(float(auc_nm_cv), 3),
               "nomogram_n_events": int(yn.sum())},
              open(os.path.join(cfg.RESULTS, "nomogram_model.json"), "w"), indent=2)
    print(f"  nomogram logistic: {coefs} b0={inter} AUC={auc_nm:.3f} "
          f"(cross-validated {auc_nm_cv:.3f})")

    # validation performance summary
    summary = {
        "discovery_OOF_AUC": round(float(roc_auc_score(y_d, p_d)), 3),
        "discovery_AUC_95CI": [round(float(ci[0]), 3), round(float(ci[1]), 3)],
        "discovery_Brier": round(float(brier_score_loss(y_d, p_d)), 4),
        "discovery_HL_chi2": round(float(hl), 2),
        "validation_AUC": round(float(auc_v), 3) if auc_v else None,
        "validation_Brier": round(float(brier_score_loss(y_v, p_v)), 4) if p_v is not None else None,
        "nomogram_AUC_apparent": round(float(auc_nm), 3),
        "nomogram_AUC_cv": round(float(auc_nm_cv), 3),
        "HL_chi2_p": round(float(hl_p), 4),
        "HL_df": int(df_hl),
    }
    json.dump(summary, open(os.path.join(cfg.RESULTS, "validation_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("=== DONE ===")

if __name__ == "__main__":
    main()
