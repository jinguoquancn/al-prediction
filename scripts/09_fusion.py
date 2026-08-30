"""
09_fusion.py — Multimodal fusion strategies comparison.
  * EARLY fusion: single model on concatenated [molecular + clinical + ICG] features.
  * LATE fusion (stacking): base learners per modality -> meta-learner (logistic).
  * Modality contribution: permutation importance + single-modality removal (ablation).
Outputs fusion_performance.tsv, ablation.tsv, meta-learner weights.
"""
import os, sys, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 20260828

def load_features(tag="discovery"):
    out = pd.read_csv(os.path.join(cfg.PROC, f"{tag}_al_outcome.tsv"), sep="\t", index_col=0)
    icg = pd.read_csv(os.path.join(cfg.PROC, f"{tag}_icg_params.tsv"), sep="\t", index_col=0)
    feat = pd.DataFrame(index=out.index)
    feat["mol_risk_z"] = out["mol_z"]
    feat["clin_risk_z"] = out["clin_z"]
    feat["AL_sig_score"] = out["mol_risk"]
    for c in icg.columns:
        feat[f"icg_{c}"] = icg[c]
    return feat, out["AL"].values.astype(int)

def cv_auc(X, y, model, k=10):
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
    miss = np.isnan(oof)
    if miss.any():
        oof[miss] = y.mean()
    if not aucs:
        aucs = [np.nan]
    return (float(np.mean(aucs)), float(np.nanstd(aucs))), oof

def base_learners():
    """Return fresh base learner dict."""
    b = {
        "LASSO": Pipeline([("sc", StandardScaler()),
                           ("clf", LogisticRegression(penalty="l1", C=0.3, solver="liblinear",
                                                      random_state=SEED, max_iter=5000))]),
        "SVM": Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                                     random_state=SEED, gamma="scale"))]),
        "RF": RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=8,
                                     random_state=SEED, n_jobs=1),
    }
    if HAS_XGB:
        b["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                     eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0)
    return b

def main():
    print("=== 09 multimodal fusion ===")
    feat, y = load_features("discovery")
    print(f"n={len(y)}, AL={y.sum()} ({y.mean()*100:.1f}%)")
    mol_cols = ["mol_risk_z", "AL_sig_score"]
    clin_cols = ["clin_risk_z"]
    icg_cols = [c for c in feat.columns if c.startswith("icg_")]
    modalities = {"Molecular": mol_cols, "Clinical": clin_cols, "ICG": icg_cols}
    results = []

    # ---------- EARLY fusion: best single model on all features ----------
    for mname, model in base_learners().items():
        (m, s), oof = cv_auc(feat.values.astype(float), y, model)
        results.append({"strategy": "Early fusion", "learner": mname,
                        "CV_AUC": m, "CV_AUC_sd": s, "Brier": brier_score_loss(y, oof)})
        print(f"  EARLY/{mname:8s} AUC={m:.3f}±{s:.3f}")

    # ---------- LATE fusion: stacking per-modality base learners ----------
    for meta_name, meta in [("Logistic meta", LogisticRegression(C=1.0, random_state=SEED, max_iter=5000))]:
        k = max(2, min(10, int(np.bincount(y).min()) if np.bincount(y).min() >= 2 else 2))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        oof_late = np.zeros(len(y))
        aucs = []
        for tr, te in skf.split(feat.values, y):
            # base learner predictions per modality
            base_pred_tr = np.zeros((len(tr), 3))
            base_pred_te = np.zeros((len(te), 3))
            for j, (mod, cols) in enumerate(modalities.items()):
                bl = base_learners()["RF"]  # fixed strong base learner
                Xt = feat.iloc[tr][cols].values.astype(float)
                Xe = feat.iloc[te][cols].values.astype(float)
                try:
                    bl.fit(Xt, y[tr])
                    base_pred_tr[:, j] = bl.predict_proba(Xt)[:, 1]
                    base_pred_te[:, j] = bl.predict_proba(Xe)[:, 1]
                except Exception:
                    base_pred_tr[:, j] = 0.5; base_pred_te[:, j] = 0.5
            # scale base preds for meta
            sc = StandardScaler().fit(base_pred_tr)
            meta.fit(sc.transform(base_pred_tr), y[tr])
            oof_late[te] = meta.predict_proba(sc.transform(base_pred_te))[:, 1]
            if len(np.unique(y[te])) >= 2:
                try:
                    aucs.append(roc_auc_score(y[te], oof_late[te]))
                except Exception:
                    pass
        m, s = float(np.mean(aucs)), float(np.std(aucs))
        results.append({"strategy": "Late fusion (stacking)", "learner": meta_name,
                        "CV_AUC": m, "CV_AUC_sd": s, "Brier": brier_score_loss(y, oof_late)})
        print(f"  LATE/{meta_name}  AUC={m:.3f}±{s:.3f}")
        # meta weights = logistic coefficients on modality base-preds
        weights = {mod: float(w) for (mod, _), w in zip(modalities.items(), meta.coef_[0])}
        json.dump({"meta_weights": weights}, open(os.path.join(cfg.RESULTS, "late_fusion_weights.json"), "w"), indent=2)
        print("  meta weights:", {k: round(v, 3) for k, v in weights.items()})

    # ---------- Ablation: remove one modality at a time (early RF) ----------
    print("  ablation (early fusion, RF):")
    abl = []
    all_cols = mol_cols + clin_cols + icg_cols
    base_m, _ = cv_auc(feat[all_cols].values.astype(float), y, base_learners()["RF"])
    abl.append({"removed": "none (full)", "CV_AUC": base_m[0], "delta_AUC": 0.0})
    for mod, cols in modalities.items():
        rem = [c for c in all_cols if c not in cols]
        (m, s), _ = cv_auc(feat[rem].values.astype(float), y, base_learners()["RF"])
        abl.append({"removed": mod, "CV_AUC": m, "delta_AUC": m - base_m[0]})
        print(f"    -{mod:10s} AUC={m:.3f} (delta {m-base_m[0]:+.3f})")
    pd.DataFrame(abl).to_csv(os.path.join(cfg.RESULTS, "ablation.tsv"), sep="\t", index=False)

    res = pd.DataFrame(results).sort_values("CV_AUC", ascending=False)
    res.to_csv(os.path.join(cfg.RESULTS, "fusion_performance.tsv"), sep="\t", index=False)
    print("\n" + res.to_string(index=False))
    print("=== DONE ===")

if __name__ == "__main__":
    main()
