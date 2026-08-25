#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def main():
    manifest=ROOT/"FILE_MANIFEST.csv"
    if not manifest.exists():
        raise SystemExit("FAIL: FILE_MANIFEST.csv missing")
    m=pd.read_csv(manifest)
    bad=[]
    for _,r in m.iterrows():
        p=ROOT/str(r.release_path)
        if not p.exists():
            bad.append((str(r.release_path),"MISSING"))
            continue
        if sha256(p)!=str(r.sha256):
            bad.append((str(r.release_path),"HASH_MISMATCH"))
    if bad:
        raise SystemExit("FAIL: "+json.dumps(bad))
    gates=pd.read_csv(ROOT/"audit"/"RELEASE_GATE.csv")
    vals=gates["value"].astype(str).str.lower().isin(["true","1","yes"])
    if not vals.all():
        raise SystemExit("FAIL: release gate contains false rows")
    print("PASS — publication release manifest and release gates verified")

if __name__=="__main__":
    main()
