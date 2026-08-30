"""
cfg.py — shared configuration: paths, matplotlib journal style (300 DPI,
tif+svg dual output, Arial 7pt, colorblind-friendly palette).
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RAW  = os.path.join(ROOT, "data", "raw")
RESULTS = os.path.join(ROOT, "results")
FIGDIR  = os.path.join(ROOT, "figures")
FIGOUT  = os.path.join(ROOT, "figures_out")
TABLES  = os.path.join(RESULTS, "tables")
EXCEL   = os.path.join(RESULTS, "excel")
SUPP    = os.path.join(ROOT, "supplementary")
MANU    = os.path.join(ROOT, "manuscript")
for d in [RESULTS, FIGDIR, FIGOUT, TABLES, EXCEL, SUPP, MANU]:
    os.makedirs(d, exist_ok=True)

# ---- journal figure style --------------------------------------------------
# Use Arial if available, else DejaVu Sans (matplotlib default, widely accepted)
try:
    fm.findfont("Arial", fallback_to_default=True)
    FAMILY = "Arial"
except Exception:
    FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FAMILY,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# colorblind-friendly Okabe-Ito palette + extensions
PALETTE = {
    "red":    "#D55E00",
    "blue":   "#0072B2",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "black":  "#000000",
    "grey":   "#999999",
}
PIE_COLORS = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#999999","#F0E442"]

def savefig(fig, name, subdir=None, dpi=300):
    """Save a figure in BOTH tif and svg at 300 DPI. subdir groups panels."""
    d = os.path.join(FIGDIR, subdir) if subdir else FIGDIR
    os.makedirs(d, exist_ok=True)
    tif = os.path.join(d, f"{name}.tif")
    svg = os.path.join(d, f"{name}.svg")
    fig.savefig(tif, dpi=dpi, format="tiff", pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(svg, format="svg")
    plt.close(fig)
    return tif, svg

def load_expr(tag):
    """Load processed expression matrix (gene x sample) as DataFrame."""
    import pandas as pd
    return pd.read_csv(os.path.join(PROC, f"{tag}_expr.tsv"), sep="\t", index_col=0)

def load_clin(tag):
    import pandas as pd
    return pd.read_csv(os.path.join(PROC, f"{tag}_clinical.tsv"), sep="\t", index_col=0)
