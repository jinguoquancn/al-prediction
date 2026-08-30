"""
01c_fix_clinical.py — Re-fetch clinical data with the CORRECT cBioPortal v2
parameter (`clinicalDataType`, not `type`) and merge patient-level attributes
onto sample barcodes.

Why this exists:
    01b_download_focused.py requested
        /studies/{id}/clinical-data?type=PATIENT
    The v2 API ignores an unknown query key and silently defaults to SAMPLE
    data, so both passes returned identical sample-level rows and every
    patient-level variable (AGE, SEX, AJCC stage, PATH_T/N/M, OS/DFS) was
    lost. The study actually exposes 60 attributes, 41 of them patient-level.

Mapping sample -> patient:
    TCGA barcode  TCGA-AA-3506-01A  -> patient TCGA-AA-3506 (first 12 chars)
    Non-TCGA ids take the id as-is; we also try the API's own patientId by
    requesting both datatypes and joining on the returned identifiers.
"""
import os, sys, json, time, collections
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg

BASE = "https://www.cbioportal.org/api"
HDR = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
RAW = cfg.RAW
PROC = cfg.PROC

STUDIES = {
    "discovery": "coadread_tcga_pan_can_atlas_2018",
    "validation": "coad_silu_2022",
}


def _get(url, tries=3):
    """cBioPortal throttles hard on large clinical pages (one page took 442 s),
    so the timeout has to be far above the usual 60-180 s."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=420) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"    retry {k+1}: {type(e).__name__} {e}", flush=True)
            time.sleep(10)
    return None


CACHE = os.path.join(RAW, "clin_pages")
os.makedirs(CACHE, exist_ok=True)


def fetch_all(study_id, datatype, max_pages=50):
    """Fetch every clinical record of one datatype (PATIENT or SAMPLE).

    Each page is cached to disk, so an interrupted run can resume instead of
    re-downloading from page 0 (long cBioPortal sessions get killed by the
    sandbox after a while).
    """
    out = []
    page = 0
    while page < max_pages:
        cf = os.path.join(CACHE, f"{study_id}__{datatype}__{page}.json")
        if os.path.exists(cf):
            with open(cf, encoding="utf-8") as f:
                d = json.load(f)
            print(f"    {datatype} page {page}: {len(d)} (cached)", flush=True)
        else:
            url = (f"{BASE}/studies/{study_id}/clinical-data"
                   f"?clinicalDataType={datatype}&projection=DETAILED"
                   f"&pageSize=10000&pageNumber={page}")
            d = _get(url)
            if not isinstance(d, list):
                print(f"    {datatype} page {page}: FAILED", flush=True)
                break
            with open(cf, "w", encoding="utf-8") as f:
                json.dump(d, f)
            print(f"    {datatype} page {page}: {len(d)} (fetched)", flush=True)
        if not d:
            break
        out.extend(d)
        if len(d) < 10000:
            break
        page += 1
    return out


OFFLINE = ("--merge" in sys.argv)   # merge pass must never hit the network


def samples_of_study(study_id):
    """Sample IDs belonging to the study (used to validate the join).

    In offline merge mode we skip this entirely: the sample keys recovered
    from the cached SAMPLE pages are sufficient.
    """
    cf = os.path.join(CACHE, f"{study_id}__SAMPLELIST__0.json")
    if os.path.exists(cf):
        with open(cf, encoding="utf-8") as f:
            d = json.load(f)
        return {s["sampleId"] for s in d if s.get("sampleId")}
    if OFFLINE:
        return set()
    d = _get(f"{BASE}/studies/{study_id}/samples?pageSize=100000&pageNumber=0")
    if isinstance(d, list):
        with open(cf, "w", encoding="utf-8") as f:
            json.dump(d, f)
        return {s["sampleId"] for s in d if s.get("sampleId")}
    return set()


def build(study_id, tag):
    print(f"[{tag}] {study_id}", flush=True)
    samp_rows = fetch_all(study_id, "SAMPLE")
    pat_rows = fetch_all(study_id, "PATIENT")
    print(f"  fetched: sample={len(samp_rows)} patient={len(pat_rows)}", flush=True)

    # patient-level: patientId -> {attr: value}
    pat = collections.defaultdict(dict)
    for r in pat_rows:
        pid = r.get("patientId")
        if not pid:
            continue
        pat[pid][r["clinicalAttributeId"]] = r.get("value", "")

    # sample-level rows
    samp = collections.defaultdict(dict)
    for r in samp_rows:
        sid = r.get("sampleId")
        if not sid:
            continue
        samp[sid][r["clinicalAttributeId"]] = r.get("value", "")

    # explicit sample -> patient map returned by the API
    s2p = {}
    for r in samp_rows + pat_rows:
        sid, pid = r.get("sampleId"), r.get("patientId")
        if sid and pid:
            s2p[sid] = pid

    ids = samples_of_study(study_id)
    keys = sorted(set(samp) | (ids if ids else set()))
    print(f"  samples in study: {len(ids)}  merged keys: {len(keys)}", flush=True)

    def patient_for(sid):
        if sid in s2p:
            return s2p[sid]
        # TCGA barcode: patient = first 12 characters
        if sid.startswith("TCGA-") and len(sid) >= 12:
            return sid[:12]
        return sid

    attrs = set()
    for d in samp.values():
        attrs |= set(d)
    for d in pat.values():
        attrs |= set(d)
    al = sorted(attrs)

    lines = ["Sample\t" + "\t".join(al)]
    n_pat_matched = 0
    for sid in keys:
        row = dict(samp.get(sid, {}))
        pid = patient_for(sid)
        if pid in pat:
            row.update(pat[pid])
            n_pat_matched += 1
        lines.append(sid + "\t" + "\t".join(str(row.get(a, "")) for a in al))

    out_tsv = os.path.join(PROC, f"{tag}_clinical.tsv")
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    raw_json = os.path.join(RAW, f"{tag}_clinical_full.json")
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump({"sample": samp_rows, "patient": pat_rows}, f)

    print(f"  wrote {out_tsv}: {len(keys)} samples, {len(al)} attrs, "
          f"{n_pat_matched} with patient-level data", flush=True)

    # report which key variables survived
    import pandas as pd
    df = pd.read_csv(out_tsv, sep="\t", index_col=0)
    for k in ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE",
              "PATH_N_STAGE", "PATH_M_STAGE", "OS_MONTHS", "OS_STATUS",
              "DFS_MONTHS", "DFS_STATUS", "MUTATION_COUNT", "TMB_NONSYNONYMOUS",
              "MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE"]:
        if k in df.columns:
            nn = df[k].replace("", np.nan).notna().sum() if False else (df[k].astype(str).str.len() > 0).sum()
            print(f"    {k}: {nn}/{len(df)}")
    return df


def fetch_page(study_id, datatype, page):
    """Fetch exactly one page and exit (sandbox caps requests per process)."""
    cf = os.path.join(CACHE, f"{study_id}__{datatype}__{page}.json")
    if os.path.exists(cf):
        with open(cf, encoding="utf-8") as f:
            print(f"cached {datatype} p{page}: {len(json.load(f))}", flush=True)
        return
    url = (f"{BASE}/studies/{study_id}/clinical-data"
           f"?clinicalDataType={datatype}&projection=DETAILED"
           f"&pageSize=10000&pageNumber={page}")
    d = _get(url)
    if not isinstance(d, list):
        print(f"FAILED {datatype} p{page}", flush=True)
        return
    with open(cf, "w", encoding="utf-8") as f:
        json.dump(d, f)
    print(f"fetched {datatype} p{page}: {len(d)}", flush=True)


if __name__ == "__main__":
    import numpy as np
    if "--page" in sys.argv:
        # one request per process: 01c_fix_clinical.py --page <study> <DATATYPE> <n>
        a = sys.argv[sys.argv.index("--page") + 1:]
        fetch_page(a[0], a[1], int(a[2]))
    elif "--merge" in sys.argv:
        for tag, sid in STUDIES.items():
            build(sid, tag)
        print("=== DONE ===")
    else:
        # legacy single-process mode (only completes if the sandbox allows it)
        for tag, sid in STUDIES.items():
            build(sid, tag)
        print("=== DONE ===")
