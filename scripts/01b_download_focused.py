"""
01b_download_focused.py — Focused REAL molecular fetch from cBioPortal.
cBioPortal /molecular-data/fetch is rate-limited (~175 s per ~20-gene batch),
so we fetch a focused candidate panel (~360 genes) that covers:
  AL signature (up+down), immune-cell markers, key ECM/CRC/growth genes.
Resumable: each batch result is saved to data/raw/batches/<cohort>_<i>.json;
the script skips already-completed batches. Clinical data is also fetched.
"""
import os, sys, json, time, pickle
import urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_panel import AL_SIGNATURE_UP, AL_SIGNATURE_DOWN, PANEL_GROUPS

BASE = "https://www.cbioportal.org/api"
HDR = {"Accept": "application/json", "Content-Type": "application/json",
       "User-Agent": "Mozilla/5.0 (AL-paper)"}
ROOT = r"D:/Jgq/AL_prediction_paper"
RAW = os.path.join(ROOT, "data", "raw"); PROC = os.path.join(ROOT, "data", "processed")
BATCH = os.path.join(RAW, "batches"); os.makedirs(BATCH, exist_ok=True)

def focus_panel():
    s = set()
    for x in AL_SIGNATURE_UP + AL_SIGNATURE_DOWN: s.add(x.upper())
    for g in ["Immune_markers","ECM_collagen","Cytokines","CRC_hallmark"]:
        for x in PANEL_GROUPS[g]: s.add(x.upper())
    return sorted(s)

def _post(url, body, tries=3, timeout=200):
    last=None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers=HDR, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last=e; time.sleep(3+t*5)
    raise RuntimeError(f"post failed: {url} :: {last}")

def fetch_clinical(study_id, tag):
    if os.path.exists(os.path.join(PROC, f"{tag}_clinical.tsv")):
        return
    out=[]
    for ctype in ["PATIENT","SAMPLE"]:
        page=0
        while True:
            url=f"{BASE}/studies/{study_id}/clinical-data?type={ctype}&pageSize=5000&pageNumber={page}&projection=DETAILED"
            try:
                req=urllib.request.Request(url, headers=HDR)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    d=json.loads(resp.read().decode())
            except Exception:
                break
            if not isinstance(d,list) or not d: break
            out.extend(d)
            if len(d)<5000: break
            page+=1
    # wide
    import collections
    by=collections.defaultdict(dict); attrs=set()
    for r in out:
        sid=r.get("sampleId") or r.get("patientId")
        if not sid: continue
        by[sid][r["clinicalAttributeId"]]=r.get("value","")
        attrs.add(r["clinicalAttributeId"])
    al=sorted(attrs)
    with open(os.path.join(PROC, f"{tag}_clinical.tsv"),"w") as f:
        f.write("Sample\t"+"\t".join(al)+"\n")
        for sid in sorted(by):
            f.write(sid+"\t"+"\t".join(str(by[sid].get(a,"")) for a in al)+"\n")
    with open(os.path.join(RAW, f"{tag}_clinical.json"),"w") as f: json.dump(out,f)
    print(f"  {tag} clinical: {len(by)} samples, {len(attrs)} attrs", flush=True)

def fetch_molecular(study_id, tag, mrna_profile, symbols, entrez, batch=18, sample_ids=None):
    """Fetch expression in small batches; resumable.
    If sample_ids given, use sampleIds in the body (needed for some cohorts);
    otherwise use sampleListId=mrna_profile."""
    done_dir = os.path.join(BATCH, tag); os.makedirs(done_dir, exist_ok=True)
    eids = [entrez[s] for s in symbols if s in entrez]
    print(f"  {tag}: {len(eids)} genes to fetch (sample_ids={len(sample_ids) if sample_ids else 'list-id'})", flush=True)
    url = f"{BASE}/molecular-profiles/{mrna_profile}/molecular-data/fetch"
    nb = (len(eids)+batch-1)//batch
    for i in range(nb):
        fn = os.path.join(done_dir, f"b{i:03d}.json")
        if os.path.exists(fn) and os.path.getsize(fn)>5:
            continue
        chunk = eids[i*batch:(i+1)*batch]
        body = {"entrezGeneIds": chunk}
        if sample_ids:
            body["sampleIds"]=sample_ids
        else:
            body["sampleListId"]=mrna_profile
        d = _post(url, body)
        json.dump(d, open(fn,"w"))
        print(f"    batch {i+1}/{nb}: {len(d)} rows", flush=True)
        time.sleep(1)
    # reload all
    rows=[]
    for i in range(nb):
        fn=os.path.join(done_dir,f"b{i:03d}.json")
        if os.path.exists(fn):
            rows.extend(json.load(open(fn)))
    print(f"  {tag}: total rows {len(rows)}", flush=True)
    json.dump(rows, open(os.path.join(RAW, f"{tag}_molecdata.json"),"w"))
    # pivot
    entrez_to_sym={v:k for k,v in entrez.items()}
    mat={}
    for r in rows:
        v=r.get("value")
        if v is None or v=="NA": continue
        try: v=float(v)
        except: continue
        sym=entrez_to_sym.get(r["entrezGeneId"])
        if sym is None: continue
        mat.setdefault(sym,{})[r["sampleId"]]=v
    all_s=sorted({sid for g in mat.values() for sid in g})
    with open(os.path.join(PROC, f"{tag}_expr.tsv"),"w") as f:
        f.write("Gene\t"+"\t".join(all_s)+"\n")
        for sym,vals in mat.items():
            f.write(sym+"\t"+"\t".join(f"{vals.get(s,float('nan')):.4f}" if s in vals else "NA" for s in all_s)+"\n")
    print(f"  {tag}: expr {len(mat)} genes x {len(all_s)} samples", flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=["all","discovery","validation"])
    a=ap.parse_args()
    print("=== focused download ===", flush=True)
    gm = pickle.load(open(os.path.join(RAW,"gene_map.pkl"),"rb"))
    focus = focus_panel()
    print("focus panel:", len(focus), "mapped:", sum(1 for s in focus if s in gm), flush=True)
    studies = {
        "discovery":  ("coadread_tcga_pan_can_atlas_2018",
                       "coadread_tcga_pan_can_atlas_2018_rna_seq_v2_mrna"),
        "validation": ("coad_silu_2022", "coad_silu_2022_rna_seq_mrna"),
    }
    keys = ["discovery","validation"] if a.cohort=="all" else [a.cohort]
    for tag in keys:
        sid, prof = studies[tag]
        print(f"-- {tag}: {sid}", flush=True)
        fetch_clinical(sid, tag)
        # validation (silu) needs sampleIds; load from clinical json
        sids=None
        if tag=="validation":
            cj=os.path.join(RAW,f"{tag}_clinical.json")
            if os.path.exists(cj):
                cl=json.load(open(cj))
                sids=sorted(set([r.get("sampleId") for r in cl if r.get("sampleId")]))
        fetch_molecular(sid, tag, prof, focus, gm, batch=18, sample_ids=sids)
    # save gene lists
    json.dump({
        "focus_panel": focus,
        "AL_signature_up":[s for s in AL_SIGNATURE_UP if s in gm],
        "AL_signature_down":[s for s in AL_SIGNATURE_DOWN if s in gm],
    }, open(os.path.join(PROC,"gene_lists.json"),"w"), indent=0)
    print("=== DONE ===", flush=True)

if __name__=="__main__":
    main()
