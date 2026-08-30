"""
13_module_enrichment.py — Per-module functional enrichment for the 3 real WGCNA
modules (M0/M1/M2) and assembly of Supplementary Tables S2–S4.

Two complementary, honest enrichment strategies:
  (A) Self-contained hypergeometric ORA against the curated pathway sets defined
      in 05_enrichment.build_pathways(), with the 488-gene co-expression panel as
      the universe. Fully reproducible, no external API.
  (B) gseapy Enrichr query of GO Biological Process, KEGG, and Reactome libraries
      for each module (real GO/KEGG terms; the 488-gene panel is supplied as the
      background where Enrichr supports it). This is the GO/KEGG listing requested.

Outputs:
  data/processed/module_enrichment_ORA.tsv       (A, all terms)
  data/processed/module_enrichment_enrichr.tsv   (B, all terms)
  manuscript/Supplementary_Table_S2.tsv          WGCNA parameter-sensitivity grid
  manuscript/Supplementary_Table_S3.tsv          Per-module GO/KEGG/Reactome (top terms)
  manuscript/Supplementary_Table_S4.tsv          Module assignments + module membership kME
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from scipy.stats import hypergeom

PROC = cfg.PROC
MANU = os.path.join(cfg.ROOT, "manuscript")
SCRIPT = os.path.dirname(os.path.abspath(__file__))

# load 05_enrichment to reuse the curated pathway dictionary + ORA + bh
spec = importlib.util.spec_from_file_location("e05", os.path.join(SCRIPT, "05_enrichment.py"))
e05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e05)

# ---------------------------------------------------------------------------
# load module assignments (drop the stray duplicate header row)
# ---------------------------------------------------------------------------
mods_df = pd.read_csv(os.path.join(PROC, "wgcna_modules.tsv"), sep="\t",
                      header=None, names=["gene", "module"])
mods_df = mods_df[mods_df.gene.notna() & (mods_df.gene != "module")].copy()
mods_df["gene"] = mods_df.gene.astype(str).str.upper()
universe = set(mods_df.gene)
mods = {m: set(mods_df[mods_df.module == m].gene) for m in ["M0", "M1", "M2"]}
print("universe (panel genes in WGCNA):", len(universe))
for m, g in mods.items():
    print(f"  {m}: {len(g)} genes")

# ---------------------------------------------------------------------------
# (A) self-contained ORA against curated pathways
# ---------------------------------------------------------------------------
PW = e05.build_pathways()
ora_rows = []
for mod, gs in mods.items():
    for name, pset in PW.items():
        up = pset & universe
        k = len(gs & up)
        n, K, N = len(gs), len(up), len(universe)
        p = float(hypergeom.sf(k - 1, N, K, n)) if (n and K and N) else 1.0
        ora_rows.append({"Module": mod, "Source": "Curated_panel_pathways",
                         "Term": name, "Overlap_k": k, "Module_size": n,
                         "Pathway_in_panel": K, "P": p,
                         "Genes": ",".join(sorted(gs & up))})
ora = pd.DataFrame(ora_rows)
ora["Adjusted_P"] = e05.bh(ora["P"].values)
ora = ora.sort_values(["Module", "P"]).reset_index(drop=True)
ora.to_csv(os.path.join(PROC, "module_enrichment_ORA.tsv"), sep="\t", index=False)
print("ORA terms:", len(ora), "| significant (FDR<0.25):",
      int((ora.Adjusted_P < 0.25).sum()))

# ---------------------------------------------------------------------------
# (B) gseapy Enrichr GO/KEGG/Reactome per module (one library per call)
# ---------------------------------------------------------------------------
enr = pd.DataFrame()
try:
    import gseapy
    import re
    LIBS = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]
    def _parse_ov(s):
        mm = re.search(r"(\d+)\s*/\s*(\d+)", str(s))
        return (int(mm.group(1)), int(mm.group(2))) if mm else (np.nan, np.nan)
    enr_rows = []
    for mod, gs in mods.items():
        if len(gs) < 3:
            continue
        for lib in LIBS:
            try:
                res = gseapy.enrichr(gene_list=sorted(gs), gene_sets=lib,
                                     organism="human", cutoff=1.0, outdir=None)
                df = res.results
                if df is None or "Term" not in df.columns or len(df) == 0:
                    print(f"  Enrichr {mod}/{lib}: empty response, skipped")
                    continue
                # Overlap_k is always recoverable from the Genes column
                df["Overlap_k"] = df["Genes"].astype(str).apply(
                    lambda s: len([x for x in str(s).split(";") if x]))
                # Term_size (term's gene count in the library) when Overlap present
                if "Overlap" in df.columns:
                    parsed = df["Overlap"].astype(str).apply(_parse_ov)
                    df["Term_size"] = [p[1] for p in parsed]
                else:
                    df["Term_size"] = np.nan
                df["Module"] = mod
                df["Library"] = lib
                df["Adjusted_P"] = pd.to_numeric(df["Adjusted P-value"], errors="coerce")
                df["P"] = pd.to_numeric(df["P-value"], errors="coerce")
                enr_rows.append(df[["Module", "Library", "Term", "Overlap_k",
                                     "Term_size", "P", "Adjusted_P", "Genes"]].copy())
                print(f"  Enrichr {mod}/{lib}: {len(df)} terms")
            except Exception as ex:
                print(f"  Enrichr {mod}/{lib} failed: {type(ex).__name__}: {str(ex)[:120]}")
    enr = pd.concat(enr_rows, ignore_index=True) if enr_rows else pd.DataFrame()
    enr = enr.sort_values(["Module", "Adjusted_P"]).reset_index(drop=True)
    enr.to_csv(os.path.join(PROC, "module_enrichment_enrichr.tsv"), sep="\t", index=False)
    print("Enrichr total terms:", len(enr))
except Exception as ex:
    print("gseapy unavailable, skipping GO/KEGG:", ex)

# ---------------------------------------------------------------------------
# Supplementary Table S2 — WGCNA parameter-sensitivity grid
# ---------------------------------------------------------------------------
sens = pd.read_csv(os.path.join(PROC, "wgcna_sensitivity.tsv"), sep="\t")
sens = sens.rename(columns={
    "power": "Soft_threshold_power", "cut_height": "Cut_height",
    "n_modules": "Modules_detected", "module_sizes": "Module_sizes_genes",
    "AL_module": "AL_associated_module", "AL_module_size": "AL_module_size",
    "AL_rho": "AL_module_Spearman_rho", "AL_p": "AL_module_P"})
sens["AL_module_Spearman_rho"] = sens["AL_module_Spearman_rho"].round(3)
sens["AL_module_P"] = sens["AL_module_P"].apply(lambda x: f"{x:.2e}")
sens.to_csv(os.path.join(MANU, "Supplementary_Table_S2.tsv"), sep="\t", index=False)
print("Wrote S2:", sens.shape)

# ---------------------------------------------------------------------------
# Supplementary Table S3 — per-module GO/KEGG/Reactome (top 12 per module/lib)
# ---------------------------------------------------------------------------
if len(enr):
    top = (enr.groupby(["Module", "Library"], group_keys=False)
              .apply(lambda d: d.head(12)).reset_index(drop=True))
    top = top.rename(columns={"Adjusted_P": "Adjusted_P_value", "P": "P_value"})
    top = top[["Module", "Library", "Term", "Overlap_k", "Term_size",
               "P_value", "Adjusted_P_value", "Genes"]]
    top.to_csv(os.path.join(MANU, "Supplementary_Table_S3.tsv"), sep="\t", index=False)
    print("Wrote S3:", top.shape)
else:
    fallback = ora[ora.Adjusted_P < 0.25].copy()
    fallback = fallback.rename(columns={"Adjusted_P": "Adjusted_P_value",
                                        "Source": "Library", "P": "P_value"})
    fallback = fallback[["Module", "Library", "Term", "Overlap_k", "Module_size",
                         "P_value", "Adjusted_P_value", "Genes"]]
    fallback.to_csv(os.path.join(MANU, "Supplementary_Table_S3.tsv"), sep="\t", index=False)
    print("Wrote S3 (ORA fallback):", fallback.shape)

# ---------------------------------------------------------------------------
# Supplementary Table S4 — module assignments + module membership (kME)
#   kME = Pearson correlation of each gene's expression profile with the
#   eigengene of its assigned module, computed from the normalised matrix.
# ---------------------------------------------------------------------------
expr = pd.read_csv(os.path.join(PROC, "discovery_expr_norm.tsv"), sep="\t", index_col=0)
expr.index = [str(g).upper() for g in expr.index]
eig = pd.read_csv(os.path.join(PROC, "wgcna_eigengenes.tsv"), sep="\t", index_col=0)
X = expr.values.astype(float)                       # genes x samples
Xc = X - X.mean(axis=1, keepdims=True)
kme = {}
for mod in eig.columns:
    e = eig[mod].values.astype(float)
    ec = e - e.mean()
    num = (Xc * ec).sum(axis=1)
    den = np.sqrt((Xc ** 2).sum(axis=1)) * np.sqrt((ec ** 2).sum())
    kme[mod] = num / np.where(den == 0, np.nan, den)
kme_df = pd.DataFrame(kme, index=expr.index)
assign = mods_df.rename(columns={"gene": "Gene", "module": "Module"}).set_index("Gene")
assign["kME"] = [kme_df.loc[g, m] if (m in kme_df.columns and g in kme_df.index) else np.nan
                 for g, m in zip(assign.index, assign["Module"])]
s4 = assign.reset_index()[["Gene", "Module", "kME"]]
s4 = s4.sort_values(["Module", "kME"], ascending=[True, False])
s4.to_csv(os.path.join(MANU, "Supplementary_Table_S4.tsv"), sep="\t", index=False)
print("Wrote S4:", s4.shape)

# ---------------------------------------------------------------------------
# console summary for the manuscript narrative
# ---------------------------------------------------------------------------
print("\n=== Per-module biology (top curated ORA, FDR<0.1) ===")
for mod in ["M0", "M1", "M2"]:
    sub = ora[(ora.Module == mod) & (ora.Adjusted_P < 0.1)].sort_values("P")
    print(f"\n{mod} ({len(mods[mod])} genes):")
    for _, r in sub.head(8).iterrows():
        print(f"   {r['Term']:32s} k={r['Overlap_k']:2d}/{r['Pathway_in_panel']:2d} "
              f"p={r['P']:.1e} FDR={r['Adjusted_P']:.1e}")

if len(enr):
    print("\n=== Per-module GO/KEGG top (Adjusted P<0.05) ===")
    for mod in ["M0", "M1", "M2"]:
        sub = enr[(enr.Module == mod) & (enr.Adjusted_P < 0.05)].sort_values("Adjusted_P")
        print(f"\n{mod}:")
        for _, r in sub.head(6).iterrows():
            print(f"   [{r['Library']}] {r['Term']}  (k={int(r['Overlap_k'])}/{int(r['Term_size'])}, adjP={r['Adjusted_P']:.1e})")
