"""
01_download.py — Download TCGA-COADREAD (PanCancer Atlas) discovery cohort and
an independent validation cohort (coad_silu_2022) from the cBioPortal REST API:
  * clinical data (patient + sample level)
  * sample metadata
  * mRNA expression for a curated candidate-gene panel
Saves expression matrices (parquet) + clinical tables (tsv) + gene map.

Uses only stdlib (urllib) so it is not blocked by missing packages.
"""
import sys, os, json, time, pickle
import urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_panel import build_candidate_panel, AL_SIGNATURE_UP, AL_SIGNATURE_DOWN

BASE = "https://www.cbioportal.org/api"
HDR = {"Accept": "application/json", "Content-Type": "application/json",
       "User-Agent": "Mozilla/5.0 (AL-paper-downloader)"}
RAW = r"D:/Jgq/AL_prediction_paper/data/raw"
PROC = r"D:/Jgq/AL_prediction_paper/data/processed"
os.makedirs(RAW, exist_ok=True); os.makedirs(PROC, exist_ok=True)

def _req(url, data=None, tries=4):
    last = None
    for t in range(tries):
        try:
            body = None if data is None else json.dumps(data).encode()
            r = urllib.request.Request(url, data=body, headers=HDR, method="POST" if data is not None else "GET")
            with urllib.request.urlopen(r, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last = e; time.sleep(2 + t*3)
    raise RuntimeError(f"request failed after {tries} tries: {url}\n{last}")

def get_json(url):
    return _req(url, data=None)

def post_json(url, body):
    return _req(url, data=body)

# ---------------------------------------------------------------- build gene map
def build_gene_map():
    cache = os.path.join(RAW, "gene_map.pkl")
    if os.path.exists(cache):
        with open(cache,"rb") as f: return pickle.load(f)
    gmap = {}
    page = 0
    while True:
        url = f"{BASE}/genes?pageSize=1000&pageNumber={page}&projection=DETAILED"
        d = get_json(url)
        if not isinstance(d, list) or not d:
            break
        for g in d:
            if g.get("entrezGeneId") and g.get("entrezGeneId") > 0:
                gmap[g["hugoGeneSymbol"].upper()] = g["entrezGeneId"]
        if len(d) < 1000:
            break
        page += 1
        if page % 10 == 0:
            print(f"  genes page {page}: total mapped {len(gmap)}")
    with open(cache,"wb") as f: pickle.dump(gmap, f)
    return gmap

# ---------------------------------------------------------------- clinical
def fetch_clinical(study_id):
    out = []
    for ctype in ["PATIENT", "SAMPLE"]:
        page = 0
        while True:
            url = f"{BASE}/studies/{study_id}/clinical-data?type={ctype}&pageSize=5000&pageNumber={page}&projection=DETAILED"
            d = get_json(url)
            if not isinstance(d, list) or not d:
                break
            out.extend(d)
            if len(d) < 5000:
                break
            page += 1
    return out

def fetch_samples(study_id):
    out = []; page = 0
    while True:
        url = f"{BASE}/samples?studyId={study_id}&pageSize=2000&pageNumber={page}&projection=DETAILED"
        d = get_json(url)
        if not isinstance(d, list) or not d: break
        out.extend(d)
        if len(d) < 2000: break
        page += 1
    return out

# ---------------------------------------------------------------- molecular data
def fetch_molecular_data(profile_id, sample_list_id, entrez_ids, batch=400):
    """POST /molecular-profiles/{profileId}/molecular-data/fetch in batches."""
    url = f"{BASE}/molecular-profiles/{profile_id}/molecular-data/fetch"
    rows = []
    n = len(entrez_ids)
    for i in range(0, n, batch):
        chunk = entrez_ids[i:i+batch]
        body = {"sampleListId": sample_list_id, "entrezGeneIds": chunk}
        d = post_json(url, body)
        if isinstance(d, list):
            rows.extend(d)
        print(f"    genes {i}-{i+len(chunk)-1}/{n}: rows so far {len(rows)}")
    return rows

def pivot_expression(rows, entrez_to_symbol):
    """rows -> {gene_symbol: {sampleId: value}}"""
    mat = {}
    for r in rows:
        v = r.get("value", None)
        if v is None or v == "NA": continue
        try: v = float(v)
        except: continue
        sym = entrez_to_symbol.get(r["entrezGeneId"])
        if sym is None: continue
        mat.setdefault(sym, {})[r["sampleId"]] = v
    return mat

# ---------------------------------------------------------------- main
def main():
    print("=== Building candidate panel & gene map ===")
    panel = build_candidate_panel()
    print("candidate panel size:", len(panel))
    gmap = build_gene_map()
    print("gene map size:", len(gmap))
    entrez_to_symbol = {v:k for k,v in gmap.items()}
    # map panel symbols
    panel_entrez = []
    missing = []
    for s in panel:
        if s in gmap:
            panel_entrez.append((gmap[s], s))
        else:
            missing.append(s)
    print(f"panel mapped: {len(panel_entrez)} / {len(panel)} (missing {len(missing)})")
    if missing:
        with open(os.path.join(RAW,"unmapped_symbols.txt"),"w") as f:
            f.write("\n".join(missing))

    studies = {
        "discovery": ("coadread_tcga_pan_can_atlas_2018",
                      "coadread_tcga_pan_can_atlas_2018_rna_seq_v2_mrna"),
        "validation": ("coad_silu_2022", "coad_silu_2022_rna_seq_mrna"),
    }

    for tag,(study_id, mrna_profile) in studies.items():
        print(f"\n=== {tag}: {study_id} ===")
        # clinical
        print("  fetching clinical ...")
        clin = fetch_clinical(study_id)
        print("  clinical rows:", len(clin))
        with open(os.path.join(RAW, f"{tag}_clinical.json"),"w") as f:
            json.dump(clin, f)
        # samples
        print("  fetching samples ...")
        samps = fetch_samples(study_id)
        print("  samples:", len(samps))
        with open(os.path.join(RAW, f"{tag}_samples.json"),"w") as f:
            json.dump(samps, f)
        # molecular data
        print("  fetching molecular data ...")
        entrez_ids = [e for e,_ in panel_entrez]
        rows = fetch_molecular_data(mrna_profile, mrna_profile, entrez_ids, batch=400)
        print("  molecular rows:", len(rows))
        with open(os.path.join(RAW, f"{tag}_molecdata.json"),"w") as f:
            json.dump(rows, f)
        mat = pivot_expression(rows, entrez_to_symbol)
        print(f"  genes with data: {len(mat)}; example gene MMP9 samples:",
              len(mat.get("MMP9",{})))
        # save as TSV (gene x sample)
        # build sample order
        all_s = sorted({sid for g in mat.values() for sid in g})
        with open(os.path.join(PROC, f"{tag}_expr.tsv"),"w") as f:
            f.write("Gene\t" + "\t".join(all_s) + "\n")
            for sym, vals in mat.items():
                row = "\t".join(f"{vals.get(s,float('nan')):.4f}" if s in vals else "NA" for s in all_s)
                f.write(sym + "\t" + row + "\n")
        print(f"  saved expr matrix: {len(mat)} genes x {len(all_s)} samples")
        # also save clinical as TSV (wide)
        save_clinical_wide(clin, tag, all_s)

    # save panel + signature entrez lists
    with open(os.path.join(PROC,"gene_lists.json"),"w") as f:
        json.dump({
            "panel_symbols": [s for _,s in panel_entrez],
            "AL_signature_up": [s for s in AL_SIGNATURE_UP if s in gmap],
            "AL_signature_down": [s for s in AL_SIGNATURE_DOWN if s in gmap],
            "panel_symbol_entrez": {s:e for e,s in panel_entrez},
        }, f, indent=0)
    print("\n=== DONE ===")

def save_clinical_wide(clin, tag, sample_order):
    # pivot clinical rows (clinicalAttributeId, sampleId/patientId, value)
    import collections
    by_sample = collections.defaultdict(dict)
    attrs = set()
    for r in clin:
        sid = r.get("sampleId") or r.get("patientId")
        if not sid: continue
        attr = r["clinicalAttributeId"]
        by_sample[sid][attr] = r.get("value","")
        attrs.add(attr)
    attr_list = sorted(attrs)
    with open(os.path.join(PROC, f"{tag}_clinical.tsv"),"w") as f:
        f.write("Sample\t" + "\t".join(attr_list) + "\n")
        for sid in sorted(by_sample):
            row = "\t".join(str(by_sample[sid].get(a,"")) for a in attr_list)
            f.write(sid + "\t" + row + "\n")

if __name__ == "__main__":
    main()
