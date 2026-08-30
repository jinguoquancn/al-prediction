"""
11_figures.py — Journal-quality composite figures (Fig 1-10).

Each figure is a SINGLE multi-panel image (panels labelled A, B, C, ...)
saved as a 300-DPI LZW-compressed TIF and as an SVG, in figures/.
Style: Arial (registered from Windows fonts), Okabe-Ito colourblind-safe
palette, 7-9 pt labels, thin axes, letter labels in the top-left corner.
Panels are generated from the actual pipeline outputs; missing inputs are
skipped gracefully so the figure pass can run incrementally.
"""
import os, sys, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")

# ---- ensure Arial is available (Windows font) -----------------------------
try:
    from matplotlib import font_manager as fm
    _af = r"C:/Windows/Fonts/arial.ttf"
    _bf = r"C:/Windows/Fonts/arialbd.ttf"
    if os.path.exists(_af):
        fm.fontManager.addfont(_af)
    if os.path.exists(_bf):
        fm.fontManager.addfont(_bf)
    plt.rcParams["font.family"] = "Arial"
    FAM = "Arial"
except Exception:
    FAM = "DejaVu Sans"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

R = cfg.PALETTE
SEED = 20260828

def exists(*parts):
    return os.path.exists(os.path.join(*parts))

def P(name):
    return os.path.join(cfg.RESULTS, name)

def PR(name):
    return os.path.join(cfg.PROC, name)

def pl(ax, letter):
    """Panel letter at the upper-left of the panel, placed just above the top
    spine and aligned to the left edge of the panel (including its y-axis
    label area) so it never overlaps plotted data or the panel title -- a
    Nature/SCI-style placement. ``bbox_inches='tight'`` keeps it in the image."""
    pos = ax.get_position()
    ax.figure.text(pos.x0 - 0.014, pos.y1 + 0.015, letter,
                   fontsize=11, fontweight="bold", va="bottom", ha="left",
                   color="#000000")


def sig_bracket(ax, x1, x2, y, pval, h=None, lw=0.7, fs=7, always=False):
    """Draw a horizontal significance bracket between x1 and x2 at height y,
    annotating asterisks above (SCI convention: * p<0.05, ** p<1e-3, *** p<1e-4)
    or 'ns' when not significant. ``y`` is the top of the compared bars/dots;
    the bracket sits just above it. When ``always`` is False (default) a
    non-significant comparison draws nothing; when True it always draws the
    bracket and labels it 'ns' if p>=0.05 (used for panels that must always
    show the comparison, e.g. Figure 5 C/D)."""
    if pval is None or (isinstance(pval, float) and np.isnan(pval)):
        if not always:
            return
        s = "ns"
    elif pval < 1e-4:
        s = "***"
    elif pval < 1e-3:
        s = "**"
    elif pval < 0.05:
        s = "*"
    else:
        s = "ns" if always else None
    if s is None:
        return
    rng = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
    if h is None:
        h = max(0.025, 0.04 * rng)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color="black")
    ax.text((x1 + x2) / 2, y + h, s, ha="center", va="bottom", fontsize=fs,
            color="black")


import textwrap as _tw
def wrap(txt, n=42):
    """Wrap a long label onto multiple lines so it never exceeds the panel width."""
    if txt is None:
        return ""
    return "\n".join(_tw.wrap(str(txt), n))

def save_composite(fig, n, dpi=300):
    d = cfg.FIGDIR
    os.makedirs(d, exist_ok=True)
    tif = os.path.join(d, f"figure{n}.tif")
    svg = os.path.join(d, f"figure{n}.svg")
    fig.savefig(tif, dpi=dpi, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(svg, format="svg")
    plt.close(fig)
    print(f"  figure{n} saved (tif+svg)")

# ============================================================ FIGURE 1
def figure1():
    print("Figure 1: study design + AL signature")
    fig = plt.figure(figsize=(7.0, 7.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.2, 1.0],
                  height_ratios=[1.0, 1.0], wspace=0.35, hspace=0.35,
                  left=0.07, right=0.97, top=0.96, bottom=0.05)
    axw = fig.add_subplot(gs[:, 0])
    # (A) workflow schematic
    steps = [
        ("TCGA-COAD/READ (n=594)\nDiscovery cohort", R["blue"]),
        ("Literature AL signature\n(194 up / 110 down genes)", R["sky"]),
        ("AL molecular score\n& AL-high/AL-low grouping", R["orange"]),
        ("WGCNA + enrichment\n+ immune infiltration", R["green"]),
        ("Virtual ICG cohort\n(26 parameters, literature-calibrated)", R["purple"]),
        ("Multimodal AI fusion\nLASSO/SVM/RF/XGBoost + SHAP", R["red"]),
        ("Validation (coad_silu_2022, n=348)\n+ ROC/DCA/calibration/nomogram", R["grey"]),
    ]
    n = len(steps); h = 0.80 / n; gap = 0.20 / n
    for i, (txt, c) in enumerate(steps):
        ytop = 1.0 - i * (h + gap)
        axw.add_patch(Rectangle((0.06, ytop - h), 0.88, h, transform=axw.transAxes,
                                facecolor=c, alpha=0.15, edgecolor=c, lw=1.0))
        axw.text(0.5, ytop - h / 2, txt, transform=axw.transAxes, ha="center",
                 va="center", fontsize=7.5, color="#000000")
        if i < n - 1:
            axw.add_patch(FancyArrowPatch((0.5, ytop - h), (0.5, ytop - h - gap),
                                          transform=axw.transAxes, arrowstyle="-|>",
                                          mutation_scale=8, lw=0.9, color=R["black"]))
    axw.axis("off")
    pl(axw, "A")

    # (B) AL score distribution by group
    axb = fig.add_subplot(gs[0, 1])
    if exists(PR("discovery_AL_score.tsv")):
        al = pd.read_csv(PR("discovery_AL_score.tsv"), sep="\t", index_col=0)
        hi = al.loc[al.AL_group == "AL_high", "AL_sig_score"]
        lo = al.loc[al.AL_group == "AL_low", "AL_sig_score"]
        axb.hist(lo, bins=30, alpha=0.65, color=R["blue"],
                 label=f"AL-low (n={len(lo)})", edgecolor="white", lw=0.2)
        axb.hist(hi, bins=30, alpha=0.65, color=R["red"],
                 label=f"AL-high (n={len(hi)})", edgecolor="white", lw=0.2)
        axb.axvline(al.AL_sig_score.median(), color=R["black"], ls="--", lw=0.8)
    axb.set_xlabel("AL molecular signature score")
    axb.set_ylabel("Number of samples")
    axb.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    pl(axb, "B")

    # (C) signature gene heatmap
    axc = fig.add_subplot(gs[1, 1])
    if exists(PR("discovery_expr_norm.tsv")):
        expr = pd.read_csv(PR("discovery_expr_norm.tsv"), sep="\t", index_col=0)
        from gene_panel import AL_SIGNATURE_UP, AL_SIGNATURE_DOWN
        up = [g for g in AL_SIGNATURE_UP if g in expr.index][:40]
        dn = [g for g in AL_SIGNATURE_DOWN if g in expr.index][:40]
        genes = up + dn
        al = pd.read_csv(PR("discovery_AL_score.tsv"), sep="\t", index_col=0)
        order = al.sort_values("AL_sig_score").index
        order = [s for s in order if s in expr.columns]
        sub = expr.loc[genes, order]
        mu = sub.mean(axis=1); sd = sub.std(axis=1).replace(0, np.nan)
        sub = sub.sub(mu, axis=0).div(sd, axis=0)
        im = axc.imshow(sub.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                        interpolation="nearest")
        axc.set_xticks([]); axc.set_yticks(range(0, len(genes), 10))
        axc.set_yticklabels([genes[i] for i in range(0, len(genes), 10)], fontsize=5)
        for gi, g in enumerate(genes):
            axc.plot([len(order) * 0.995], [gi], marker=">", ms=2,
                     color=R["red"] if g in up else R["blue"])
        axc.set_xlabel("Samples (sorted by AL score)")
        axc.set_ylabel("Signature genes")
        cb = fig.colorbar(im, ax=axc, fraction=0.04); cb.set_label("z-score", fontsize=6)
    pl(axc, "C")
    save_composite(fig, 1)

# ============================================================ FIGURE 2
def figure2():
    print("Figure 2: DEG")
    if not exists(PR("discovery_DEG.tsv")):
        return
    deg = pd.read_csv(PR("discovery_DEG.tsv"), sep="\t")
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.85],
                  hspace=0.42, wspace=0.35, left=0.10, right=0.97, top=0.95, bottom=0.08)
    # (A) volcano
    axa = fig.add_subplot(gs[0, 0])
    sig = deg[deg.sig != "ns"]; ns = deg[deg.sig == "ns"]
    axa.scatter(ns.log2FC, -np.log10(ns.pval.clip(1e-300)), s=4, color=R["grey"],
                alpha=0.35, lw=0)
    axa.scatter(sig.loc[sig.sig == "Up", "log2FC"], -np.log10(sig.loc[sig.sig == "Up", "pval"]),
                s=6, color=R["red"], alpha=0.7, lw=0, label=f"Up (n={(sig.sig=='Up').sum()})")
    axa.scatter(sig.loc[sig.sig == "Down", "log2FC"], -np.log10(sig.loc[sig.sig == "Down", "pval"]),
                s=6, color=R["blue"], alpha=0.7, lw=0, label=f"Down (n={(sig.sig=='Down').sum()})")
    axa.axhline(-np.log10(0.05), color=R["black"], ls=":", lw=0.5)
    axa.axvline(0.5, color=R["black"], ls=":", lw=0.5)
    axa.axvline(-0.5, color=R["black"], ls=":", lw=0.5)
    top = deg.head(8)
    for _, r in top.iterrows():
        axa.annotate(r.gene, (r.log2FC, -np.log10(max(r.pval, 1e-300))), fontsize=5,
                     xytext=(2, 2), textcoords="offset points")
    axa.set_xlabel("log2 fold change (AL-high vs AL-low)")
    axa.set_ylabel("-log10(P)")
    axa.legend(frameon=False, markerscale=2, fontsize=6)
    pl(axa, "A")

    # (B) top DEG heatmap
    axb = fig.add_subplot(gs[0, 1])
    if exists(PR("discovery_expr_norm.tsv")):
        expr = pd.read_csv(PR("discovery_expr_norm.tsv"), sep="\t", index_col=0)
        al = pd.read_csv(PR("discovery_AL_score.tsv"), sep="\t", index_col=0)
        topg = deg.head(30).gene.tolist()
        topg = [g for g in topg if g in expr.index]
        order = al.sort_values("AL_sig_score").index
        order = [s for s in order if s in expr.columns]
        sub = expr.loc[topg, order]
        mu = sub.mean(axis=1); sd = sub.std(axis=1).replace(0, np.nan)
        sub = sub.sub(mu, axis=0).div(sd, axis=0)
        im = axb.imshow(sub.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                        interpolation="nearest")
        axb.set_xticks([]); axb.set_yticks(range(len(topg)))
        axb.set_yticklabels(topg, fontsize=5)
        axb.set_xlabel("Samples")
        cb = fig.colorbar(im, ax=axb, fraction=0.04); cb.set_label("z-score", fontsize=6)
    pl(axb, "B")

    # (C) top DEG bar (full width)
    axc = fig.add_subplot(gs[1, :])
    top = deg.head(15).iloc[::-1]
    colors = [R["red"] if v > 0 else R["blue"] for v in top.log2FC]
    axc.barh(range(len(top)), top.log2FC, color=colors, height=0.65, edgecolor="white", lw=0.2)
    axc.set_yticks(range(len(top))); axc.set_yticklabels(top.gene, fontsize=6)
    axc.axvline(0, color=R["black"], lw=0.6)
    axc.set_xlabel("log2FC (AL-high vs AL-low)")
    pl(axc, "C")
    save_composite(fig, 2)

# ============================================================ FIGURE 3
def figure3():
    print("Figure 3: WGCNA")
    if not exists(PR("wgcna_module_trait.tsv")):
        return
    mt = pd.read_csv(PR("wgcna_module_trait.tsv"), sep="\t")
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0],
                  hspace=0.45, wspace=0.45, left=0.10, right=0.97, top=0.94, bottom=0.09)
    # (A) module-trait correlation heatmap
    axa = fig.add_subplot(gs[0, 0])
    piv = mt.pivot_table(index="module", columns="trait", values="rho")
    pmat = mt.pivot_table(index="module", columns="trait", values="p")
    im = axa.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                    interpolation="nearest")
    axa.set_xticks(range(piv.shape[1])); axa.set_xticklabels(piv.columns, fontsize=6, rotation=45, ha="right")
    axa.set_yticks(range(piv.shape[0])); axa.set_yticklabels(piv.index, fontsize=6)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            pv = pmat.values[i, j]
            star = "**" if (not np.isnan(pv) and pv < 0.01) else ("*" if (not np.isnan(pv) and pv < 0.05) else "")
            axa.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=5)
    cb = fig.colorbar(im, ax=axa, fraction=0.04); cb.set_label("Spearman rho", fontsize=6)
    axa.set_title("Module-trait relationships", fontsize=8)
    pl(axa, "A")

    # (B) soft threshold
    axb = fig.add_subplot(gs[0, 1])
    if exists(PR("wgcna_softthreshold.tsv")):
        st = pd.read_csv(PR("wgcna_softthreshold.tsv"), sep="\t")
        axb.plot(st.power, st.scale_free_R2, "o-", color=R["blue"], ms=4, lw=1)
        axb.axhline(0.8, color=R["red"], ls="--", lw=0.7)
        axb.set_xlabel("Soft threshold beta")
        axb.set_ylabel("Scale-free R2")
    pl(axb, "B")

    # (C) hub genes of the AL-associated module
    axc = fig.add_subplot(gs[1, 0])
    if exists(PR("wgcna_hub_genes.tsv")):
        hub = pd.read_csv(PR("wgcna_hub_genes.tsv"), sep="\t").head(20)
        axc.barh(range(len(hub))[::-1], hub.kME, color=R["orange"], height=0.65,
                 edgecolor="white", lw=0.2)
        axc.set_yticks(range(len(hub))[::-1]); axc.set_yticklabels(hub.gene, fontsize=6)
        axc.set_xlabel("kME (module membership)")
    pl(axc, "C")

    # (D) module eigengenes across samples (shows the resolved modules)
    axd = fig.add_subplot(gs[1, 1])
    if exists(PR("wgcna_eigengenes.tsv")):
        eg = pd.read_csv(PR("wgcna_eigengenes.tsv"), sep="\t", index_col=0)
        cols = [c for c in ["M0", "M1", "M2"] if c in eg.columns]
        if cols:
            order = eg[cols[1]].sort_values(ascending=False).index if len(cols) > 1 else eg.index
            mat = eg.loc[order, cols].values
            vmax = np.nanmax(np.abs(mat)) or 1.0
            imd = axd.imshow(mat.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             interpolation="nearest")
            axd.set_xticks([])
            axd.set_yticks(range(len(cols))); axd.set_yticklabels(cols, fontsize=6)
            axd.set_title("Module eigengenes", fontsize=8)
            cb2 = fig.colorbar(imd, ax=axd, fraction=0.04); cb2.set_label("Eigengene", fontsize=6)
    pl(axd, "D")
    save_composite(fig, 3)

# ============================================================ FIGURE 4
def figure4():
    print("Figure 4: enrichment")
    fig = plt.figure(figsize=(7.4, 4.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.5, left=0.20, right=0.97, top=0.92, bottom=0.12)
    # (A) ORA
    axa = fig.add_subplot(gs[0, 0])
    if exists(PR("enrichment_ORA.tsv")):
        ora = pd.read_csv(PR("enrichment_ORA.tsv"), sep="\t")
        up = ora[(ora.direction == "Up") & (ora.FDR < 0.3)].sort_values("p").head(12)
        if len(up) == 0:
            up = ora[ora.direction == "Up"].sort_values("p").head(12)
        axa.barh(range(len(up))[::-1], -np.log10(up.FDR.clip(1e-50, 1)), color=R["blue"],
                 height=0.6, edgecolor="white", lw=0.2)
        axa.set_yticks(range(len(up))[::-1])
        axa.set_yticklabels([wrap(t, 42) for t in up.pathway], fontsize=6)
        axa.set_xlabel("-log10(FDR)")
        axa.set_title("Pathway over-representation (Up-DEGs)", fontsize=8)
    pl(axa, "A")
    # (B) GSEA
    axb = fig.add_subplot(gs[0, 1])
    if exists(PR("enrichment_GSEA.tsv")):
        gs_df = pd.read_csv(PR("enrichment_GSEA.tsv"), sep="\t").head(10).sort_values("ES")
        colors = [R["red"] if n > 0 else R["blue"] for n in gs_df.ES]
        axb.barh(range(len(gs_df)), gs_df.ES, color=colors, height=0.6, edgecolor="white", lw=0.2)
        axb.set_yticks(range(len(gs_df))); axb.set_yticklabels([wrap(t, 42) for t in gs_df.pathway], fontsize=6)
        axb.axvline(0, color=R["black"], lw=0.6)
        axb.set_xlabel("Enrichment score (ES)")
        axb.set_title("GSEA: AL-high vs AL-low", fontsize=8)
    pl(axb, "B")
    save_composite(fig, 4)

# ============================================================ FIGURE 5
def figure5():
    print("Figure 5: immune infiltration")
    if not exists(PR("immune_group_comparison.tsv")):
        return
    ig = pd.read_csv(PR("immune_group_comparison.tsv"), sep="\t")
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.95],
                  hspace=0.45, wspace=0.4, left=0.10, right=0.97, top=0.95, bottom=0.08)
    # (A) lollipop: ssGSEA score by cell type, AL-low vs AL-high
    axa = fig.add_subplot(gs[0, 0])
    ig = ig.sort_values("p"); show = ig.head(16)
    axa.grid(axis="y", ls=":", lw=0.4, color="#cccccc", zorder=0)
    ymax = 0
    for i, (_, r) in enumerate(show.iterrows()):
        axa.plot([i, i], [r.mean_ALlow, r.mean_ALhigh], color=R["grey"], lw=0.5, zorder=1)
        axa.plot(i, r.mean_ALlow, "o", ms=4, color=R["blue"], zorder=2)
        axa.plot(i, r.mean_ALhigh, "o", ms=4, color=R["red"], zorder=2)
        ymax = max(ymax, r.mean_ALlow, r.mean_ALhigh)
    axa.set_xticks(range(len(show)))
    axa.set_xticklabels(show.cell, rotation=55, ha="right", fontsize=6)
    axa.set_ylabel("ssGSEA score (group mean)")
    # per-cell significance of AL-low vs AL-high (Mann-Whitney, * p<0.05)
    for i, (_, r) in enumerate(show.iterrows()):
        if r.FDR < 0.05:
            ytop = max(r.mean_ALlow, r.mean_ALhigh)
            axa.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2],
                     [ytop + 0.015, ytop + 0.05, ytop + 0.05, ytop + 0.015],
                     color="black", lw=0.6)
            axa.text(i, ytop + 0.058, "*", ha="center", va="bottom", fontsize=7)
    # extra headroom so the top-right legend never overlaps the lollipops/asterisks
    axa.set_ylim(top=ymax * 1.40)
    axa.plot([], [], "o", color=R["blue"], label="AL-low")
    axa.plot([], [], "o", color=R["red"], label="AL-high")
    axa.legend(frameon=False, fontsize=6, loc="upper right", borderaxespad=0.5)
    pl(axa, "A")

    # (B) correlation with AL score
    axb = fig.add_subplot(gs[0, 1])
    if exists(PR("immune_AL_correlation.tsv")):
        ic = pd.read_csv(PR("immune_AL_correlation.tsv"), sep="\t").sort_values("rho")
        show = pd.concat([ic.head(8), ic.tail(8)])
        colors = [R["red"] if v > 0 else R["blue"] for v in show.rho]
        axb.barh(range(len(show)), show.rho, color=colors, height=0.6, edgecolor="white", lw=0.2)
        axb.set_yticks(range(len(show))); axb.set_yticklabels(show.cell, fontsize=6)
        axb.axvline(0, color=R["black"], lw=0.6)
        axb.set_xlabel("Spearman rho with AL score")
    pl(axb, "B")

    # (C)/(D) ESTIMATE-like stromal & immune scores: AL-low vs AL-high boxplots.
    # Each comparison group (AL-low, AL-high) is a single box; we fix boxplot
    # positions to [0, 1] so the significance bracket (drawn at x=0..1) sits
    # directly above BOTH boxes, with xtick labels aligned to the same positions.
    est = None
    if exists(PR("immune_estimate.tsv")):
        est = pd.read_csv(PR("immune_estimate.tsv"), sep="\t", index_col=0)
        al = pd.read_csv(PR("discovery_AL_score.tsv"), sep="\t", index_col=0)
        est = est.loc[[s for s in est.index if s in al.index]]
        grp = al.loc[est.index, "AL_group"]
    for col, sp, letter in [("Stromal", gs[1, 0], "C"), ("Immune", gs[1, 1], "D")]:
        sax = fig.add_subplot(sp)
        if est is None or col not in est.columns:
            sax.axis("off"); continue
        data = [est.loc[grp == "AL_low", col].dropna().values,
                est.loc[grp == "AL_high", col].dropna().values]
        if len(data[0]) < 5 or len(data[1]) < 5:
            sax.axis("off"); continue
        bp = sax.boxplot(data, positions=[0, 1], patch_artist=True, widths=0.5,
                         medianprops=dict(color=R["black"], lw=0.8),
                         boxprops=dict(linewidth=0.6),
                         whiskerprops=dict(linewidth=0.6),
                         capprops=dict(linewidth=0.6))
        for patch, c in zip(bp["boxes"], [R["blue"], R["red"]]):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        try:
            p = mannwhitneyu(data[0], data[1]).pvalue
        except Exception:
            p = None
        sax.set_title(f"{col} score (AL-low vs AL-high)", fontsize=7)
        try:
            allv = np.concatenate(data)
            ymaxv = np.nanmax(allv); yminv = np.nanmin(allv)
            rng = ymaxv - yminv
            sax.set_ylim(yminv - 0.05 * rng, ymaxv + 0.22 * rng)
            # bracket spans x=0..1, i.e. exactly the two boxes below it.
            # always=True so a non-significant comparison still shows a
            # bracket labelled 'ns' (Figure 5 must always display the test).
            sig_bracket(sax, 0, 1, ymaxv + 0.04 * rng, p,
                        h=max(0.03, 0.06 * rng), fs=7, always=True)
        except Exception:
            pass
        sax.set_xticks([0, 1]); sax.set_xticklabels(["AL-low", "AL-high"], fontsize=6.5)
        pl(sax, letter)
    save_composite(fig, 5)

# ============================================================ FIGURE 6
def figure6():
    print("Figure 6: ICG virtual cohort & CCA")
    if not exists(PR("discovery_icg_params.tsv")):
        return
    icg = pd.read_csv(PR("discovery_icg_params.tsv"), sep="\t", index_col=0)
    out = pd.read_csv(PR("discovery_al_outcome.tsv"), sep="\t", index_col=0)
    from gene_panel import ICG_PARAMETERS
    fig = plt.figure(figsize=(8.6, 4.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1],
                  wspace=0.5, left=0.08, right=0.98, top=0.90, bottom=0.31)
    # (A) ICG parameter medians (AL vs no-AL), sorted by effect size
    axa = fig.add_subplot(gs[0, 0])
    names = {p[0]: p[1] for p in ICG_PARAMETERS}
    cols = list(icg.columns[:12])
    diffs = []
    for c in cols:
        v_leak = icg.loc[out.AL == 1, c].dropna().values
        v_no = icg.loc[out.AL == 0, c].dropna().values
        diffs.append(abs(np.median(v_leak) - np.median(v_no)) if (len(v_leak) and len(v_no)) else 0)
    show = [cols[i] for i in np.argsort(diffs)[::-1]]
    axa.grid(axis="y", ls=":", lw=0.4, color="#cccccc", zorder=0)
    ymaxall = 0
    for i, c in enumerate(show):
        v_leak = icg.loc[out.AL == 1, c].dropna().values
        v_no = icg.loc[out.AL == 0, c].dropna().values
        med_leak = np.median(v_leak) if len(v_leak) else 0
        med_no = np.median(v_no) if len(v_no) else 0
        axa.plot([i - 0.18, i + 0.18], [med_leak, med_no], color=R["grey"], lw=0.6, zorder=1)
        axa.plot(i - 0.18, med_leak, "o", ms=4, color=R["red"], zorder=2)
        axa.plot(i + 0.18, med_no, "o", ms=4, color=R["blue"], zorder=2)
        ymaxall = max(ymaxall, med_leak, med_no)
        if len(v_leak) >= 3 and len(v_no) >= 3:
            try:
                p = mannwhitneyu(v_leak, v_no).pvalue
                rng = (np.nanmax(np.concatenate([v_leak, v_no]))
                       - np.nanmin(np.concatenate([v_leak, v_no])))
                sig_bracket(axa, i - 0.18, i + 0.18, max(med_leak, med_no) + 0.03 * rng, p,
                            h=max(0.02, 0.035 * rng), fs=6)
            except Exception:
                pass
    # Single-line short labels: at 45 deg the perpendicular gap between adjacent
    # ticks is smaller than the thickness of a 2-line label, so wrapped labels
    # collide. Keep every label on ONE line and let the generous bottom margin
    # (see GridSpec above) absorb the diagonal text extent.
    SHORT = {
        "T0": "Arrival time", "Tmax": "Time to max", "TTP": "Time to peak",
        "T1_2": "Half-decay time", "T90": "Time to 90% peak",
        "Fmax": "Max intensity", "Fmean": "Mean intensity",
        "Slope": "Inflow rate", "S1": "Initial slope",
        "Ingress": "Ingress rate", "Egress": "Egress rate",
        "PDR": "Perfusion decay", "RPI": "Relative perf. index",
        "ICG_ratio": "ICG ratio", "Tf": "Fluorescence onset",
        "Dur": "Fluorescence duration", "AUC": "Area under curve",
        "PI": "Perfusion index", "WoT": "Washout time",
        "MFR": "Max fluor. rate", "Tt": "Transit time",
        "F30": "Intensity at 30 s", "F60": "Intensity at 60 s",
        "PV": "Perfusion velocity", "TSI": "Tissue saturation",
        "ICGscore": "Composite ICG score",
    }
    axa.set_xticks(range(len(show)))
    axa.set_xticklabels([SHORT.get(c, c) for c in show],
                        rotation=45, ha="right", fontsize=5.5)
    axa.set_ylabel("ICG parameter value (median)")
    axa.set_ylim(top=ymaxall * 1.18)
    axa.plot([], [], "o", color=R["red"], label="AL", ms=4)
    axa.plot([], [], "o", color=R["blue"], label="no AL", ms=4)
    axa.legend(frameon=False, loc="upper right", fontsize=6)
    pl(axa, "A")

    # (B) CCA
    axb = fig.add_subplot(gs[0, 1])
    cca = pd.read_csv(PR("discovery_cca.tsv"), sep="\t")
    axb.bar(cca.component, cca.canon_corr, color=R["purple"], width=0.5, edgecolor="white", lw=0.2)
    axb.set_xticks(cca.component)
    axb.set_xlabel("Canonical variate")
    axb.set_ylabel("Canonical correlation")
    axb.set_ylim(0, 1)
    for _, r in cca.iterrows():
        axb.text(r.component, r.canon_corr + 0.03, f"{r.canon_corr:.2f}", ha="center", fontsize=7)
    pl(axb, "B")

    # (C) ICG risk vs molecular risk scatter
    axc = fig.add_subplot(gs[0, 2])
    for lab, c in [(1, R["red"]), (0, R["blue"])]:
        m = out.AL == lab
        axc.scatter(out.loc[m, "mol_z"], out.loc[m, "icg_risk_z"], s=4, alpha=0.5,
                    color=c, label="AL" if lab else "no AL", lw=0)
    rho = np.corrcoef(out.mol_z, out.icg_risk_z)[0, 1]
    axc.set_xlabel("Molecular AL-risk (z)")
    axc.set_ylabel("ICG composite risk (z)")
    axc.text(0.97, 0.05, f"r={rho:.2f}", transform=axc.transAxes, fontsize=7,
             va="bottom", ha="right")
    axc.legend(frameon=False, markerscale=2, fontsize=6, loc="upper right")
    pl(axc, "C")
    save_composite(fig, 6)

# ============================================================ FIGURE 7
def figure7():
    print("Figure 7: AI model comparison + SHAP")
    if not exists(P("model_performance.tsv")):
        return
    perf = pd.read_csv(P("model_performance.tsv"), sep="\t")
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.09, right=0.97, top=0.95, bottom=0.07)
    # (A) ROC
    axa = fig.add_subplot(gs[0, 0])
    if exists(P("roc_data.tsv")):
        roc = pd.read_csv(P("roc_data.tsv"), sep="\t")
        for mod, c in [("Discovery (OOF)", R["blue"])]:
            d = roc[roc.model == mod]
            if len(d):
                axa.plot(d.fpr, d.tpr, color=c, lw=1.4, label=f"{mod} AUC={d.AUC.iloc[0]:.3f}")
        axa.plot([0, 1], [0, 1], ls=":", color=R["grey"], lw=0.6)
        axa.set_xlabel("1 - Specificity"); axa.set_ylabel("Sensitivity")
        axa.legend(frameon=False, loc="lower right", fontsize=6)
    pl(axa, "A")

    # (B) AUC by modality x model
    axb = fig.add_subplot(gs[0, 1])
    mm = perf[perf.modality == "Multimodal (all)"]
    axb.bar(np.arange(len(mm)) - 0.18, mm.CV_AUC_mean, yerr=mm.CV_AUC_sd, width=0.35,
            color=R["blue"], edgecolor="white", lw=0.2, error_kw=dict(lw=0.6, capsize=2),
            label="All features")
    single = perf[perf.modality == "Molecular only"]
    axb.bar(np.arange(len(single)) + 0.18, single.CV_AUC_mean.values, yerr=single.CV_AUC_sd.values,
            width=0.35, color=R["orange"], edgecolor="white", lw=0.2,
            error_kw=dict(lw=0.6, capsize=2), label="Molecular only")
    axb.set_xticks(np.arange(len(mm))); axb.set_xticklabels(mm.model, fontsize=6, rotation=15)
    axb.set_ylabel("10-fold CV AUC"); axb.set_ylim(0.5, 1.0)
    axb.legend(frameon=False, fontsize=6)
    pl(axb, "B")

    # (C) SHAP summary
    axc = fig.add_subplot(gs[1, 0])
    if exists(P("shap_importance.tsv")):
        imp = pd.read_csv(P("shap_importance.tsv"), sep="\t").head(15)
        cmap = {"ICG": R["purple"], "Molecular": R["blue"], "Clinical": R["orange"]}
        axc.barh(range(len(imp))[::-1], imp.mean_abs_SHAP,
                 color=[cmap.get(m, R["grey"]) for m in imp.modality], height=0.6,
                 edgecolor="white", lw=0.2)
        axc.set_yticks(range(len(imp))[::-1])
        axc.set_yticklabels([wrap(f.replace("icg_", ""), 22) for f in imp.feature], fontsize=6)
        axc.set_xlabel("mean |SHAP|")
        handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in cmap.values()]
        axc.legend(handles, cmap.keys(), frameon=False, fontsize=6)
    pl(axc, "C")

    # (D) modality contribution pie
    axd = fig.add_subplot(gs[1, 1])
    if exists(P("modality_contribution.tsv")):
        mc = pd.read_csv(P("modality_contribution.tsv"), sep="\t", index_col=0)
        axd.pie(mc.contribution_pct, labels=mc.index, colors=cfg.PIE_COLORS[:len(mc)],
                autopct="%.1f%%", textprops=dict(fontsize=7), startangle=90,
                wedgeprops=dict(lw=0.5, edgecolor="white"))
        axd.set_title("Modality contribution", fontsize=8)
    pl(axd, "D")
    save_composite(fig, 7)

# ============================================================ FIGURE 8
def figure8():
    print("Figure 8: fusion strategies")
    if not exists(P("fusion_performance.tsv")):
        return
    fus = pd.read_csv(P("fusion_performance.tsv"), sep="\t")
    fig = plt.figure(figsize=(7.6, 3.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.0, 0.8],
                  wspace=0.5, left=0.08, right=0.97, top=0.88, bottom=0.22)
    # (A) early vs late
    axa = fig.add_subplot(gs[0, 0])
    short_learner = [r.learner for _, r in fus.iterrows()]
    colors = [R["blue"] if "Early" in s else R["green"] for s in fus.strategy]
    axa.bar(range(len(fus)), fus.CV_AUC, yerr=fus.CV_AUC_sd, color=colors, width=0.55,
            edgecolor="white", lw=0.2, error_kw=dict(lw=0.6, capsize=2))
    axa.set_xticks(range(len(fus))); axa.set_xticklabels(short_learner, fontsize=6, rotation=20, ha="right")
    axa.set_ylabel("CV AUC"); axa.set_ylim(0.5, 1.0)
    from matplotlib.patches import Patch
    axa.set_ylim(0.5, 1.03)
    axa.legend(handles=[Patch(facecolor=R["blue"], label="Early fusion"),
                        Patch(facecolor=R["green"], label="Late fusion (stacking)")],
               frameon=False, fontsize=6, loc="upper right")
    pl(axa, "A")

    # (B) ablation
    axb = fig.add_subplot(gs[0, 1])
    if exists(P("ablation.tsv")):
        abl = pd.read_csv(P("ablation.tsv"), sep="\t")
        colors = [R["green"] if r.removed == "none (full)" else R["orange"] for _, r in abl.iterrows()]
        axb.bar(range(len(abl)), abl.CV_AUC, color=colors, width=0.55, edgecolor="white", lw=0.2)
        axb.set_xticks(range(len(abl))); axb.set_xticklabels(abl.removed, fontsize=6, rotation=20)
        axb.set_ylabel("CV AUC (RF)"); axb.set_ylim(0.5, 1.0)
        for i, (_, r) in enumerate(abl.iterrows()):
            axb.text(i, r.CV_AUC + 0.008, f"{r.CV_AUC:.3f}", ha="center", fontsize=6)
    pl(axb, "B")

    # (C) late fusion weights
    axc = fig.add_subplot(gs[0, 2])
    if exists(P("late_fusion_weights.json")):
        w = json.load(open(P("late_fusion_weights.json")))["meta_weights"]
        keys = list(w.keys()); vals = [w[k] for k in keys]
        colors = [R["blue"] if v > 0 else R["red"] for v in vals]
        axc.bar(range(len(keys)), vals, color=colors, width=0.5, edgecolor="white", lw=0.2)
        axc.set_xticks(range(len(keys))); axc.set_xticklabels(keys, fontsize=6.5)
        axc.axhline(0, color=R["black"], lw=0.6)
        axc.set_ylabel("Meta-learner weight")
    pl(axc, "C")
    save_composite(fig, 8)

# ============================================================ FIGURE 9
def figure9():
    print("Figure 9: validation & clinical utility")
    if not exists(P("roc_data.tsv")):
        return
    roc = pd.read_csv(P("roc_data.tsv"), sep="\t")
    fig = plt.figure(figsize=(7.6, 3.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.45, left=0.08, right=0.97, top=0.9, bottom=0.14)
    # (A) ROC
    axa = fig.add_subplot(gs[0, 0])
    colors = {"Discovery (OOF)": R["blue"], "Validation": R["green"]}
    for mod, d in roc.groupby("model"):
        if len(d):
            axa.plot(d.fpr, d.tpr, lw=1.4, color=colors.get(mod, R["grey"]),
                     label=f"{mod} AUC={d.AUC.iloc[0]:.3f}")
    axa.plot([0, 1], [0, 1], ls=":", color=R["grey"], lw=0.6)
    axa.set_xlabel("1 - Specificity"); axa.set_ylabel("Sensitivity")
    axa.legend(frameon=False, loc="lower right", fontsize=6)
    pl(axa, "A")

    # (B) DCA
    axb = fig.add_subplot(gs[0, 1])
    if exists(P("dca_data.tsv")):
        dca = pd.read_csv(P("dca_data.tsv"), sep="\t")
        axb.plot(dca.threshold, dca.model_nb, color=R["blue"], lw=1.2, label="Multimodal model")
        axb.plot(dca.threshold, dca.treat_all_nb, color=R["grey"], lw=0.9, ls="--", label="Treat all")
        axb.plot(dca.threshold, dca.treat_none_nb, color=R["black"], lw=0.9, ls=":", label="Treat none")
        if "validation_model_nb" in dca:
            axb.plot(dca.threshold, dca.validation_model_nb, color=R["green"], lw=1.1, label="Validation model")
        axb.axhline(0, color=R["black"], lw=0.4)
        axb.set_xlabel("Threshold probability")
        axb.set_ylabel("Net benefit")
        axb.set_ylim(-0.05, max(0.12, dca.model_nb.max() * 1.15))
        axb.legend(frameon=False, fontsize=6)
    pl(axb, "B")

    # (C) calibration
    axc = fig.add_subplot(gs[0, 2])
    if exists(P("calibration_discovery.tsv")):
        cal = pd.read_csv(P("calibration_discovery.tsv"), sep="\t")
        axc.plot([0, 0.35], [0, 0.35], ls="--", color=R["grey"], lw=0.8, label="Perfect")
        axc.plot(cal.mean_predicted, cal.observed_rate, "o-", color=R["blue"], ms=3.5, lw=1, label="Discovery")
        if exists(P("calibration_validation.tsv")):
            cv = pd.read_csv(P("calibration_validation.tsv"), sep="\t")
            axc.plot(cv.mean_predicted, cv.observed_rate, "s-", color=R["green"], ms=3.5, lw=1, label="Validation")
        axc.set_xlabel("Predicted probability")
        axc.set_ylabel("Observed AL rate")
        axc.legend(frameon=False, fontsize=6)
    pl(axc, "C")
    save_composite(fig, 9)

# ============================================================ FIGURE 10
def figure10():
    """Figure 10 (combined): (a) nomogram, (b) nomogram ROC, (c) Kaplan-Meier
    overall survival by AL molecular subtype. Three panels, evenly spaced."""
    print("Figure 10: nomogram + ROC + survival (combined)")
    if not exists(P("nomogram_model.json")):
        return
    nm = json.load(open(P("nomogram_model.json")))
    coefs = nm["coefficients"]; b0 = nm["intercept"]
    # Three evenly-distributed panels: nomogram (A), ROC (B), survival (C).
    fig = plt.figure(figsize=(12.0, 4.3))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1],
                  wspace=0.34, left=0.06, right=0.985, top=0.90, bottom=0.13)
    # (A) nomogram -- clean axes with generous vertical spacing so the
    # "Points" scale, the variable axes and the "Total points / risk" scale
    # never overlap or look duplicated.
    axa = fig.add_subplot(gs[0, 0])
    total_c = sum(abs(v) for v in coefs.values())
    axa.set_xlim(-0.05, 1.05); axa.set_ylim(-0.34, 1.62); axa.set_axis_off()
    pt_ticks = [0, 20, 40, 60, 80, 100]
    py = 1.42                                   # top "Points" scale
    axa.plot([0, 1], [py, py], color=R["black"], lw=0.8)
    for v in pt_ticks:
        x = v / 100
        axa.plot([x, x], [py, py - 0.025], color=R["black"], lw=0.6)
        axa.text(x, py + 0.03, f"{v}", ha="center", fontsize=6.5)
    axa.text(0.0, py + 0.10, "Points", fontsize=8, ha="left")
    rows = list(coefs.items()); y_top, y_step = 1.05, 0.30
    for i, (name, c) in enumerate(rows):
        y = y_top - i * y_step
        axa.text(0.0, y + 0.05, name, fontsize=8, ha="left")
        x_full = abs(c) * 3 / total_c
        if x_full > 1.0:
            x_full = 1.0
        axa.plot([0, x_full], [y, y], color=R["blue"], lw=1.0)
        # tick marks only (no repeated |z| text) -> reads via the top Points scale
        for zv in [0, 1, 2]:
            xp = abs(c * zv) / total_c
            if 0 <= xp <= 1:
                axa.plot([xp, xp], [y, y - 0.022], color=R["black"], lw=0.5)
    yt = y_top - (len(rows) - 1) * y_step - 0.30   # bottom "Total points" scale
    axa.plot([0, 1], [yt, yt], color=R["black"], lw=0.8)
    for v in pt_ticks:
        x = v / 100
        axa.plot([x, x], [yt, yt + 0.025], color=R["black"], lw=0.6)
        axa.text(x, yt - 0.04, f"{v}", ha="center", fontsize=6.5)
    axa.text(0.0, yt + 0.10, "Total points", fontsize=8, ha="left")
    yr = yt - 0.28                               # "AL risk" probability axis
    axa.text(0.0, yr + 0.06, "AL risk", fontsize=8, ha="left", color=R["red"])
    for rp in [0.05, 0.10, 0.20, 0.40, 0.60]:
        lp = np.log(rp / (1 - rp)) - b0
        x = min(max(abs(lp) / total_c, 0.0), 1.0)
        axa.plot([x, x], [yr, yr + 0.025], color=R["red"], lw=0.6)
        axa.text(x, yr - 0.05, f"{rp*100:.0f}%", ha="center", fontsize=6.5, color=R["red"])
    axa.set_title("Nomogram for anastomotic leakage risk", fontsize=9)
    pl(axa, "A")

    # (B) nomogram ROC
    axb = fig.add_subplot(gs[0, 1])
    out = pd.read_csv(PR("discovery_al_outcome.tsv"), sep="\t", index_col=0)
    from sklearn.metrics import roc_curve
    z = {k: (out["mol_z"] if k == "mol_z" else out["clin_z"] if k == "clin_z" else out["icg_risk_z"])
         for k in coefs}
    lp = sum(coefs[k] * v for k, v in z.items()) + b0
    p = 1 / (1 + np.exp(-lp))
    fpr, tpr, _ = roc_curve(out.AL, p)
    axb.plot(fpr, tpr, color=R["orange"], lw=1.4, label=f"AUC={nm['nomogram_AUC']:.3f}")
    axb.plot([0, 1], [0, 1], ls=":", color=R["grey"], lw=0.6)
    axb.set_xlabel("1 - Specificity"); axb.set_ylabel("Sensitivity")
    axb.set_xlim(0, 1); axb.set_ylim(0, 1.02)
    axb.legend(frameon=False, fontsize=6, loc="lower right")
    pl(axb, "B")

    # (C) Kaplan-Meier overall survival by AL molecular subtype
    axc = fig.add_subplot(gs[0, 2])
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        from lifelines import CoxPHFitter
    except Exception as e:
        print("  lifelines unavailable:", e); axc = None
    if axc is not None:
        al = pd.read_csv(PR("discovery_AL_score.tsv"), sep="\t", index_col=0)
        clin_path = PR("discovery_clinical.tsv")
        if not exists(clin_path):
            clin_path = P("discovery_clinical.tsv")
        clin = pd.read_csv(clin_path, sep="\t", index_col=0)
        m = al.join(clin, how="inner")
        if {"OS_MONTHS", "OS_STATUS", "AL_group"}.issubset(m.columns):
            m = m.dropna(subset=["OS_MONTHS", "OS_STATUS", "AL_group"])
            m["event"] = (m["OS_STATUS"] == "1:DECEASED").astype(int)
            m["duration"] = m["OS_MONTHS"].astype(float)
            import re as _re
            _rv = {"I": 1, "V": 5, "X": 10}
            def _roman(tok):
                o = 0; p = 0
                for ch in reversed(tok):
                    v = _rv.get(ch, 0)
                    if v >= p:
                        o += v
                    else:
                        o -= v
                    p = v
                return o
            def stage_ord(s):
                if not isinstance(s, str):
                    return np.nan
                mm = _re.search(r"STAGE\s+([IVX]+)", s, _re.IGNORECASE)
                return min(4, _roman(mm.group(1))) if mm else np.nan
            if "AJCC_PATHOLOGIC_TUMOR_STAGE" in m.columns:
                m["stage_ord"] = m["AJCC_PATHOLOGIC_TUMOR_STAGE"].apply(stage_ord)
            else:
                m["stage_ord"] = np.nan
            m = m[(m["duration"] > 0)]
            groups = {"AL_low": R["blue"], "AL_high": R["red"]}
            kmfit = {}; med = {}
            for g, c in groups.items():
                sub = m[m.AL_group == g]
                if len(sub) < 10:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(sub["duration"], event_observed=sub["event"], label=g.replace("_", "-"))
                kmf.plot_survival_function(ax=axc, color=c, lw=1.4, ci_show=True, alpha=0.12)
                kmfit[g] = kmf
                mv = kmf.median_survival_time_
                med[g] = float(mv) if np.isfinite(mv) else None
            low = m[m.AL_group == "AL_low"]; high = m[m.AL_group == "AL_high"]
            lr = logrank_test(low["duration"], high["duration"],
                              event_observed_A=low["event"], event_observed_B=high["event"])
            p_surv = float(lr.p_value)
            axc.set_xlabel("Overall survival (months)")
            axc.set_ylabel("Survival probability")
            axc.set_ylim(0, 1.03); axc.set_xlim(0, max(120, m["duration"].max()))
            axc.legend(frameon=False, fontsize=7, loc="upper right")
            axc.text(0.5, 0.04, f"log-rank P = {p_surv:.1e}", transform=axc.transAxes,
                     ha="center", fontsize=7)
            pl(axc, "C")
            # Cox PH adjusted for age + stage
            hr = hr_lo = hr_hi = hr_p = None
            try:
                cov = m.dropna(subset=["stage_ord"]).copy()
                cov["AL_high"] = (cov["AL_group"] == "AL_high").astype(int)
                cov = cov[["AL_high", "AGE", "stage_ord", "duration", "event"]].dropna()
                if len(cov) > 20 and cov["AL_high"].nunique() == 2:
                    cox = CoxPHFitter()
                    cox.fit(cov, duration_col="duration", event_col="event")
                    if "AL_high" in cox.summary.index:
                        row = cox.summary.loc["AL_high"]
                        hr = float(np.exp(row["coef"]))
                        hr_lo = float(np.exp(row["coef lower 95%"]))
                        hr_hi = float(np.exp(row["coef upper 95%"]))
                        hr_p = float(row["p"])
            except Exception as e:
                print("  Cox failed:", e)
            summary = {
                "n_low": int((m.AL_group == "AL_low").sum()),
                "n_high": int((m.AL_group == "AL_high").sum()),
                "median_low": med.get("AL_low"),
                "median_high": med.get("AL_high"),
                "logrank_p": p_surv,
                "hr": hr, "hr_ci_lower": hr_lo, "hr_ci_upper": hr_hi, "hr_p": hr_p,
            }
            try:
                json.dump(summary, open(P("survival_summary.json"), "w"), indent=2, default=str)
            except Exception:
                pass
            print("  survival summary:", summary)
    save_composite(fig, 10)

# NOTE: The Kaplan-Meier survival panel formerly drawn as Figure 11 has been
# merged into Figure 10 (panel C) so Figures 10 and 11 are now a single,
# evenly-distributed three-panel composite (A = nomogram, B = ROC, C = survival).


def main():
    print("=== 11 figures (composite) ===")
    for i, fn in enumerate([figure1, figure2, figure3, figure4, figure5,
                            figure6, figure7, figure8, figure9, figure10],
                           start=1):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  figure{i} failed: {e}")
            traceback.print_exc()
    print("=== DONE ===")

if __name__ == "__main__":
    main()
