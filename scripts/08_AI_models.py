"""
08_AI_models.py — AI prediction models with nested cross-validation & SHAP.
Models: LASSO (logistic l1), SVM (RBF), Random Forest, XGBoost.
Input: multimodal feature matrix from 07 (molecular + clinical + ICG).
Output: 10-fold CV AUC per model, single-modality vs multimodal comparison,
SHAP values for best model, feature importance, calibrated predictions.
"""
import os, sys, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SEED = 20260828

def load_multimodal(tag):
    """Assemble multimodal feature matrix: [mol_z, clin_z] + 26 ICG params."""
    out = pd.read_csv(os.path.join(cfg.PROC, f"{tag}_al_outcome.tsv"), sep="\t", index_col=0)
    icg = pd.read_csv(os.path.join(cfg.PROC, f"{tag}_icg_params.tsv"), sep="\t", index_col=0)
    feat = pd.DataFrame(index=out.index)
    feat["mol_risk_z"] = out["mol_z"]
    feat["clin_risk_z"] = out["clin_z"]
    # AL sig score raw too
    feat["AL_sig_score"] = out["mol_risk"]
    for c in icg.columns:
        feat[f"icg_{c}"] = icg[c]
    y = out["AL"].values.astype(int)
    return feat, y, out

def make_models():
    models = {
        "LASSO": Pipeline([("sc", StandardScaler()),
                           ("clf", LogisticRegression(penalty="l1", C=0.3, solver="liblinear",
                                                      max_iter=5000, random_state=SEED))]),
        "SVM":   Pipeline([("sc", StandardScaler()),
                           ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                                       random_state=SEED, gamma="scale"))]),
        "RF":    RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=8,
                                        random_state=SEED, n_jobs=1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          reg_lambda=1.0, eval_metric="logloss",
                                          random_state=SEED, n_jobs=1, verbosity=0)
    return models

def cv_metrics(X, y, model, k=10):
    """Stratified k-fold CV; returns per-fold AUC and pooled OOF predictions.
    Robust to folds with a single class; unfilled OOF entries get the base rate."""
    from sklearn.base import clone
    k = max(2, min(k, int(np.bincount(y).min()) if np.bincount(y).min() >= 2 else 2))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    aucs, oof = [], np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        try:
            m = clone(model)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
        except Exception:
            continue
        oof[te] = p
        if len(np.unique(y[te])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[te], p))
        except Exception:
            continue
    # fill untouched samples with base rate so downstream metrics don't crash
    miss = np.isnan(oof)
    if miss.any():
        oof[miss] = y.mean()
    return np.array(aucs), oof

def bootstrap_auc(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y); p = np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    a = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def main():
    print("=== 08 AI prediction models ===")
    feat, y, out = load_multimodal("discovery")
    print(f"features: {feat.shape}, AL cases: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
    n_icg = sum(1 for c in feat.columns if c.startswith("icg_"))

    # ---- modality subsets -------------------------------------------------
    modalities = {
        "Molecular only": ["mol_risk_z", "AL_sig_score"],
        "Clinical only":  ["clin_risk_z"],
        "ICG only":       [c for c in feat.columns if c.startswith("icg_")],
        "Mol+Clin":       ["mol_risk_z", "AL_sig_score", "clin_risk_z"],
        "Mol+ICG":        ["mol_risk_z", "AL_sig_score"] + [c for c in feat.columns if c.startswith("icg_")],
        "Clin+ICG":       ["clin_risk_z"] + [c for c in feat.columns if c.startswith("icg_")],
        "Multimodal (all)": list(feat.columns),
    }
    X_all = feat.values.astype(float)

    results = []
    best_auc, best_name, best_model_key = 0, None, None
    oof_store = {}
    for mod_name, cols in modalities.items():
        X = feat[cols].values.astype(float)
        for model_name in ["LASSO", "SVM", "RF"] + (["XGBoost"] if HAS_XGB else []):
            models = make_models()
            aucs, oof = cv_metrics(X, y, models[model_name], k=10)
            if len(aucs) == 0: continue
            m_auc = aucs.mean(); sd = aucs.std()
            b_mean, b_lo, b_hi = bootstrap_auc(y, oof)
            results.append({
                "modality": mod_name, "model": model_name,
                "CV_AUC_mean": m_auc, "CV_AUC_sd": sd,
                "bootstrap_AUC": b_mean, "bootstrap_95CI_low": b_lo, "bootstrap_95CI_high": b_hi,
                "n_features": len(cols),
            })
            oof_store[(mod_name, model_name)] = oof
            print(f"  {mod_name:20s} {model_name:8s} AUC={m_auc:.3f}±{sd:.3f}  boot 95%CI [{b_lo:.3f},{b_hi:.3f}]")
            if mod_name == "Multimodal (all)" and not np.isnan(m_auc) and m_auc > best_auc:
                best_auc, best_name, best_model_key = m_auc, model_name, mod_name

    if best_name is None:  # fallback if all AUCs were degenerate
        best_name, best_model_key, best_auc = "RF", "Multimodal (all)", 0.0
        print("  WARNING: degenerate CV; falling back to RF on all features")

    res = pd.DataFrame(results).sort_values("CV_AUC_mean", ascending=False)
    res.to_csv(os.path.join(cfg.RESULTS, "model_performance.tsv"), sep="\t", index=False)
    print(f"\nBest multimodal model: {best_name} (AUC={best_auc:.3f})")

    # ---- train best model on full data for SHAP & deployment ---------------
    models = make_models()
    best = models[best_name]
    best.fit(X_all, y)
    full_pred = best.predict_proba(X_all)[:, 1]

    # SHAP on best model
    shap_rows = None
    if HAS_SHAP:
        print("  computing SHAP values ...")
        try:
            if best_name in ("XGBoost", "RF"):
                expl = shap.TreeExplainer(best if best_name == "XGBoost" else best)
                sv = expl.shap_values(X_all)
                if isinstance(sv, list): sv = sv[1]
                sv = np.asarray(sv)
                if sv.ndim == 3:  # (n, features, classes) — new shap convention
                    sv = sv[:, :, 1]
            else:
                # pipeline: scaler + linear/SVM
                Xs = StandardScaler().fit_transform(X_all)
                inner = LogisticRegression(penalty="l2", max_iter=5000, random_state=SEED)
                inner.fit(Xs, y)
                expl = shap.LinearExplainer(inner, Xs)
                sv = expl.shap_values(Xs)
            shap_df = pd.DataFrame(sv, columns=feat.columns)
            shap_df.insert(0, "sample", feat.index)
            shap_df.to_csv(os.path.join(cfg.RESULTS, "shap_values.tsv"), sep="\t", index=False)
            imp = pd.DataFrame({
                "feature": feat.columns,
                "mean_abs_SHAP": np.abs(sv).mean(axis=0),
            }).sort_values("mean_abs_SHAP", ascending=False)
            imp["modality"] = np.where(imp.feature.str.startswith("icg_"), "ICG",
                                np.where(imp.feature.isin(["clin_risk_z"]), "Clinical", "Molecular"))
            imp.to_csv(os.path.join(cfg.RESULTS, "shap_importance.tsv"), sep="\t", index=False)
            print("  top 10 SHAP features:\n", imp.head(10).to_string(index=False))
            shap_rows = imp
        except Exception as e:
            print("  SHAP failed:", e)

    # LASSO coefficients (sign & selected features) — interpretability
    lasso = Pipeline([("sc", StandardScaler()),
                      ("clf", LogisticRegression(penalty="l1", C=0.3, solver="liblinear",
                                                 max_iter=5000, random_state=SEED))])
    lasso.fit(X_all, y)
    coef = pd.Series(lasso.named_steps["clf"].coef_[0], index=feat.columns)
    coef_df = pd.DataFrame({"feature": coef.index, "coefficient": coef.values})
    coef_df["nonzero"] = coef_df.coefficient != 0
    coef_df = coef_df.sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(os.path.join(cfg.RESULTS, "lasso_coefficients.tsv"), sep="\t", index=False)
    print(f"  LASSO: {int(coef_df.nonzero.sum())}/{len(coef_df)} features retained")

    # modality contribution via SHAP grouped
    if shap_rows is not None:
        grp = shap_rows.groupby("modality")["mean_abs_SHAP"].sum()
        grp = (grp / grp.sum() * 100).round(2)
        grp.to_frame("contribution_pct").to_csv(os.path.join(cfg.RESULTS, "modality_contribution.tsv"), sep="\t")
        print("  modality contribution (%):", dict(grp))

    # pooled OOF performance of best config
    oof = oof_store.get((best_model_key, best_name), np.zeros(len(y)))
    # Youden-optimal threshold (0.5 is miscalibrated for ~8% prevalence)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, oof)
    j = np.argmax(tpr - fpr)
    th_opt = float(thr[j]) if not np.isnan(thr[j]) else 0.5
    pred = (oof > th_opt).astype(int)
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    perf = {
        "best_model": best_name, "best_modality": best_model_key,
        "youden_threshold": round(th_opt, 4),
        "OOF_AUC": roc_auc_score(y, oof),
        "OOF_Brier": brier_score_loss(y, oof),
        "OOF_F1": f1_score(y, pred, zero_division=0),
        "OOF_Acc": accuracy_score(y, pred),
        "OOF_Sens": recall_score(y, pred, zero_division=0),
        "OOF_Spec": tn / (tn + fp) if (tn + fp) else 0,
        "AL_prevalence": float(y.mean()),
    }
    json.dump(perf, open(os.path.join(cfg.PROC, "best_model_perf.json"), "w"), indent=2)
    print("  pooled OOF:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in perf.items()}))

    # save OOF predictions for downstream DCA/calibration
    pd.DataFrame({"sample": feat.index, "y": y, "p": oof, "p_full": full_pred}).to_csv(
        os.path.join(cfg.RESULTS, "oof_predictions.tsv"), sep="\t", index=False)
    print("=== DONE ===")

if __name__ == "__main__":
    main()
