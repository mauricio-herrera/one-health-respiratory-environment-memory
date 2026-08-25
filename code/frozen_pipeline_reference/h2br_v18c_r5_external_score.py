#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, math, re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

REV="V18C_R5_20260825T"
DEFAULT_PROJECT=Path("/PATH/TO/ONE_HEALTH_PROJECT")

R10R7_OUT="stage34C3b7b15h2br_prospective_full_history_repair_fit_OUTPUT_v18c_r4_r10_r7"
R10R7_PASS="PASS_PROSPECTIVE_FULL_HISTORY_REPAIR_FIT_M0_M2_FROZEN_DISTINCT_2025_PREDICTIONS_READY_FOR_EXTERNAL_RESCORING"
SADU_PASS="PASS_SADU_2025_SOURCE_TRUTH_CANONICAL_ROOT_FROZEN_4015_M0_M2_MAINLINE_READY"
PASS="PASS_EXTERNAL_2025_REPAIRED_M0_M2_SCORE_FROZEN_M3_REMAINS_DEFERRED"

PRED_FILE="FROZEN_REPAIR_2025_M0_M2_PREDICTIONS_PRE_OUTCOME_SCORE.parquet"
PRED_COLS={
    "M0":"M0_predicted_count_repair",
    "M1":"M1_predicted_count_repair",
    "M2":"M2_predicted_count_repair",
}
TARGET_ALIASES=(
    "urgent_respiratory_total",
    "urgent_respiratory",
    "sadu_urgent_respiratory_total",
)

def norm(x):
    s=unicodedata.normalize("NFKD",str(x))
    s="".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+","_",s).strip("_")

def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def read_table(p:Path):
    if p.suffix.lower()==".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower()==".csv":
        return pd.read_csv(p,low_memory=False)
    raise RuntimeError(f"unsupported table: {p}")

def normalize_keys(x:pd.DataFrame,name:str):
    if "date" not in x.columns or "unit" not in x.columns:
        raise RuntimeError(f"{name} missing date/unit")
    q=x.copy()
    q["date"]=pd.to_datetime(q["date"],errors="coerce").dt.normalize()
    q["unit"]=q["unit"].astype(str).str.strip()
    if q.date.isna().any():
        raise RuntimeError(f"{name} invalid date")
    if q.duplicated(["date","unit"]).any():
        raise RuntimeError(f"{name} duplicate date/unit")
    return q

def require_repaired_predictions(project:Path):
    summ=project/R10R7_OUT/"B7B15H2BR_V18C_R4_R10_R7_SUMMARY.json"
    pred=project/R10R7_OUT/PRED_FILE
    if not summ.exists():raise RuntimeError(f"R10_R7 summary missing: {summ}")
    obj=json.loads(summ.read_text())
    if obj.get("status")!=R10R7_PASS:
        raise RuntimeError(f"R10_R7 PASS missing: {obj.get('status')}")
    if obj.get("2025_health_read") is not False or obj.get("2025_outcome_scored") is not False:
        raise RuntimeError("R10_R7 taint guard failed")
    if not pred.exists():raise RuntimeError(f"repaired prediction file missing: {pred}")
    x=normalize_keys(pd.read_parquet(pred),"repaired predictions")
    req=set(PRED_COLS.values())
    if not req.issubset(x.columns):
        raise RuntimeError(f"repaired prediction columns missing: {sorted(req-set(x.columns))}")
    if len(x)!=4015 or x.date.nunique()!=365 or x.unit.nunique()!=11:
        raise RuntimeError(f"repaired prediction grid changed {len(x)}/{x.date.nunique()}/{x.unit.nunique()}")
    for c in PRED_COLS.values():
        v=pd.to_numeric(x[c],errors="coerce").to_numpy(float)
        if not (np.isfinite(v).all() and (v>0).all()):
            raise RuntimeError(f"invalid repaired predictions in {c}")
    return summ,obj,pred,x

def summary_jsons_with_pass(project:Path):
    rows=[]
    for d in sorted(project.iterdir()):
        if not d.is_dir() or "v18b_r10" not in d.name.lower():
            continue
        for p in d.rglob("*.json"):
            try:
                if p.stat().st_size>5_000_000:continue
                txt=p.read_text(errors="ignore")
            except Exception:
                continue
            if SADU_PASS in txt:
                rows.append((d.resolve(),p.resolve(),txt))
    return rows

def canonical_sadu_candidates(stage:Path):
    out=[]
    for p in sorted(stage.rglob("*")):
        try:
            if not p.is_file() or p.suffix.lower() not in {".csv",".parquet"}:
                continue
            if p.stat().st_size>500_000_000:continue
            x=read_table(p)
        except Exception:
            continue
        cmap={norm(c):c for c in x.columns}
        datec=cmap.get("date"); unitc=cmap.get("unit")
        tc=None
        for a in TARGET_ALIASES:
            if a in cmap:
                tc=cmap[a];break
        if datec is None or unitc is None or tc is None:
            continue
        q=pd.DataFrame({
            "date":pd.to_datetime(x[datec],errors="coerce").dt.normalize(),
            "unit":x[unitc].astype(str).str.strip(),
            "urgent_respiratory_total":pd.to_numeric(x[tc],errors="coerce"),
        })
        if not (
            len(q)==4015 and q.date.notna().all()
            and q.date.nunique()==365 and q.unit.nunique()==11
            and q.duplicated(["date","unit"]).sum()==0
            and q.urgent_respiratory_total.notna().all()
            and np.isfinite(q.urgent_respiratory_total.to_numpy(float)).all()
            and (q.urgent_respiratory_total>=0).all()
        ):
            continue
        q=q.sort_values(["date","unit"]).reset_index(drop=True)
        h=hashlib.sha256(pd.util.hash_pandas_object(q,index=False).values.tobytes()).hexdigest()
        out.append((p.resolve(),q,h))
    return out

def resolve_sadu(project:Path):
    hits=summary_jsons_with_pass(project)
    if len(hits)!=1:
        raise RuntimeError(
            f"V18B_R10 PASS stage not unique: pass_json_hits={len(hits)} "
            f"stages={[str(d) for d,_,_ in hits]}"
        )
    stage,summary,txt=hits[0]
    candidates=canonical_sadu_candidates(stage)
    if not candidates:
        raise RuntimeError(f"no canonical 4015 SADU table found under exact PASS stage {stage}")
    groups={}
    for p,q,h in candidates:
        groups.setdefault(h,[]).append((p,q))
    if len(groups)!=1:
        raise RuntimeError(
            f"SADU canonical root not logically unique under PASS stage: "
            f"physical={len(candidates)} logical={len(groups)}"
        )
    h,vals=next(iter(groups.items()))
    p,q=vals[0]
    return stage,summary,p,q,{
        "physical_candidates":len(candidates),
        "logical_candidates":1,
        "logical_hash":h,
        "equivalent_paths":[str(pp) for pp,_ in vals],
    }

def poisson_deviance_obs(y,mu):
    y=np.asarray(y,float);mu=np.asarray(mu,float)
    if not (np.isfinite(y).all() and np.isfinite(mu).all() and (y>=0).all() and (mu>0).all()):
        raise RuntimeError("invalid y/mu for Poisson deviance")
    out=np.empty_like(y,float)
    pos=y>0
    out[pos]=2.0*(y[pos]*np.log(y[pos]/mu[pos])-(y[pos]-mu[pos]))
    out[~pos]=2.0*mu[~pos]
    if not (np.isfinite(out).all() and (out>=-1e-10).all()):
        raise RuntimeError("invalid Poisson deviance contributions")
    return out

def observation_losses(df):
    out=df[["date","unit","urgent_respiratory_total"]].copy()
    y=df["urgent_respiratory_total"].to_numpy(float)
    for m,c in PRED_COLS.items():
        mu=df[c].to_numpy(float)
        err=mu-y
        out[f"{m}_predicted_count"]=mu
        out[f"{m}_error"]=err
        out[f"{m}_abs_error"]=np.abs(err)
        out[f"{m}_squared_error"]=err**2
        out[f"{m}_poisson_deviance"]=poisson_deviance_obs(y,mu)
    return out

def metrics_from_losses(loss,scope,unit=None):
    rows=[]
    g=loss if unit is None else loss[loss.unit==unit]
    if not len(g):raise RuntimeError(f"empty metric scope {scope}/{unit}")
    y=g.urgent_respiratory_total.to_numpy(float)
    for m in PRED_COLS:
        ae=g[f"{m}_abs_error"].to_numpy(float)
        se=g[f"{m}_squared_error"].to_numpy(float)
        dev=g[f"{m}_poisson_deviance"].to_numpy(float)
        err=g[f"{m}_error"].to_numpy(float)
        rows.append({
            "scope":scope,"unit":unit,
            "model":m,"n":len(g),
            "MAE":float(np.mean(ae)),
            "RMSE":float(np.sqrt(np.mean(se))),
            "mean_poisson_deviance":float(np.mean(dev)),
            "mean_error":float(np.mean(err)),
            "observed_mean":float(np.mean(y)),
            "predicted_mean":float(np.mean(g[f"{m}_predicted_count"])),
        })
    return rows

def pairwise_deltas(metrics):
    rows=[]
    pairs=[("M1","M0"),("M2","M0"),("M2","M1")]
    keys=["MAE","RMSE","mean_poisson_deviance","mean_error"]
    for (scope,unit),g in metrics.groupby(["scope","unit"],dropna=False):
        ix=g.set_index("model")
        for a,b in pairs:
            if a not in ix.index or b not in ix.index:
                raise RuntimeError(f"models missing in metric group {scope}/{unit}")
            rec={"scope":scope,"unit":unit,"comparison":f"{a}-{b}","n":int(ix.loc[a,"n"])}
            for k in keys:
                rec[f"delta_{k}"]=float(ix.loc[a,k]-ix.loc[b,k])
            rec["improves_MAE"]=bool(rec["delta_MAE"]<0)
            rec["improves_RMSE"]=bool(rec["delta_RMSE"]<0)
            rec["improves_deviance"]=bool(rec["delta_mean_poisson_deviance"]<0)
            rows.append(rec)
    return pd.DataFrame(rows)

def daily_paired_losses(loss):
    rows=[]
    for d,g in loss.groupby("date",sort=True):
        rec={"date":d,"n_units":g.unit.nunique(),"n_rows":len(g)}
        for m in PRED_COLS:
            rec[f"{m}_MAE_day"]=float(g[f"{m}_abs_error"].mean())
            rec[f"{m}_MSE_day"]=float(g[f"{m}_squared_error"].mean())
            rec[f"{m}_deviance_day"]=float(g[f"{m}_poisson_deviance"].mean())
        for a,b in [("M1","M0"),("M2","M0"),("M2","M1")]:
            rec[f"{a}_minus_{b}_MAE_day"]=rec[f"{a}_MAE_day"]-rec[f"{b}_MAE_day"]
            rec[f"{a}_minus_{b}_MSE_day"]=rec[f"{a}_MSE_day"]-rec[f"{b}_MSE_day"]
            rec[f"{a}_minus_{b}_deviance_day"]=rec[f"{a}_deviance_day"]-rec[f"{b}_deviance_day"]
        rows.append(rec)
    q=pd.DataFrame(rows)
    if len(q)!=365 or not q.n_units.eq(11).all() or not q.n_rows.eq(11).all():
        raise RuntimeError("daily paired-loss panel not 365 days × 11 units")
    return q

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=DEFAULT_PROJECT)
    ap.add_argument("--outdir","--output-dir",dest="outdir",type=Path,default=None)
    a=ap.parse_args()
    project=a.project_root.resolve()
    out=(a.outdir or project/"stage34C3b7b15h2br_external_2025_repaired_m0_m2_score_OUTPUT_v18c_r5").resolve()
    out.mkdir(parents=True,exist_ok=True)

    print("="*190)
    print("STAGE 34C.3b.7B15H2BR V18C_R5 — EXTERNAL 2025 REPAIRED M0–M2 SCORE")
    print(f"PACKAGE REVISION                         : {REV}")
    print("="*190)
    print("prediction source                         : R10_R7 frozen repaired health-blind M0–M2 predictions")
    print("outcome source                            : V18B_R10 frozen SADU 2025 canonical root")
    print("model refit / recalibration               : FORBIDDEN")
    print("bootstrap / inference                     : NOT RUN IN R5")
    print("M3                                        : DEFERRED — GRD 2025 not yet published")

    psumm,pobj,predp,pred=require_repaired_predictions(project)
    sadustage,sadusumm,sadup,sadu,smeta=resolve_sadu(project)

    pred=pred.sort_values(["date","unit"]).reset_index(drop=True)
    sadu=sadu.sort_values(["date","unit"]).reset_index(drop=True)
    if not pred[["date","unit"]].equals(sadu[["date","unit"]]):
        merged=pred.merge(sadu,on=["date","unit"],how="outer",indicator=True)
        counts=merged["_merge"].value_counts().to_dict()
        raise RuntimeError(f"repaired prediction/SADU key mismatch: {counts}")

    df=pred.merge(sadu,on=["date","unit"],how="left",validate="one_to_one")
    if len(df)!=4015:
        raise RuntimeError("scoring merge changed row count")

    loss=observation_losses(df)
    loss.to_parquet(out/"EXTERNAL_2025_REPAIRED_M0_M2_OBSERVATION_LOSSES.parquet",index=False)
    loss.to_csv(out/"EXTERNAL_2025_REPAIRED_M0_M2_OBSERVATION_LOSSES.csv",index=False)

    rows=[]
    rows.extend(metrics_from_losses(loss,"POOLED",None))
    for unit in sorted(loss.unit.unique(),key=str):
        rows.extend(metrics_from_losses(loss,"UNIT",unit))
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/"EXTERNAL_2025_REPAIRED_M0_M2_METRICS.csv",index=False)

    deltas=pairwise_deltas(metrics)
    deltas.to_csv(out/"EXTERNAL_2025_REPAIRED_M0_M2_PAIRWISE_DELTAS.csv",index=False)

    daily=daily_paired_losses(loss)
    daily.to_csv(out/"EXTERNAL_2025_REPAIRED_M0_M2_DAILY_PAIRED_LOSSES.csv",index=False)

    pooled=metrics[metrics.scope=="POOLED"].copy()
    pooled_delta=deltas[deltas.scope=="POOLED"].copy()
    regional=deltas[deltas.scope=="UNIT"].copy()

    # Regional improvement counts by pair.
    regional_summary=[]
    for comp,g in regional.groupby("comparison"):
        regional_summary.append({
            "comparison":comp,
            "regions_total":g.unit.nunique(),
            "regions_improve_MAE":int(g.improves_MAE.sum()),
            "regions_improve_RMSE":int(g.improves_RMSE.sum()),
            "regions_improve_deviance":int(g.improves_deviance.sum()),
            "regions_improve_all_three":int((g.improves_MAE & g.improves_RMSE & g.improves_deviance).sum()),
        })
    regional_summary=pd.DataFrame(regional_summary)
    regional_summary.to_csv(out/"EXTERNAL_2025_REPAIRED_M0_M2_REGIONAL_IMPROVEMENT_COUNTS.csv",index=False)

    print("repaired prediction rows/days/units".ljust(68)+f": {len(pred)}/{pred.date.nunique()}/{pred.unit.nunique()}")
    print("SADU outcome rows/days/units".ljust(68)+f": {len(sadu)}/{sadu.date.nunique()}/{sadu.unit.nunique()}")
    print("SADU logical canonical candidates".ljust(68)+f": {smeta['logical_candidates']}")
    print("\nPooled external 2025 metrics")
    print(pooled[["model","n","MAE","RMSE","mean_poisson_deviance","mean_error","observed_mean","predicted_mean"]].to_string(
        index=False,float_format=lambda x:f"{x:.12g}"
    ))
    print("\nPooled pairwise deltas (negative = lower loss for first model)")
    print(pooled_delta[[
        "comparison","delta_MAE","delta_RMSE","delta_mean_poisson_deviance","delta_mean_error",
        "improves_MAE","improves_RMSE","improves_deviance"
    ]].to_string(index=False,float_format=lambda x:f"{x:.12g}"))
    print("\nRegional improvement counts")
    print(regional_summary.to_string(index=False))

    # Pure descriptive classification, not inferential claim.
    m2m0=pooled_delta[pooled_delta.comparison=="M2-M0"].iloc[0]
    m2m1=pooled_delta[pooled_delta.comparison=="M2-M1"].iloc[0]
    if (
        m2m0.delta_MAE<0 and m2m0.delta_RMSE<0 and m2m0.delta_mean_poisson_deviance<0
        and m2m1.delta_MAE<0 and m2m1.delta_RMSE<0 and m2m1.delta_mean_poisson_deviance<0
    ):
        descriptive="M2_LOWER_POOLED_MAE_RMSE_DEVIANCE_THAN_M0_AND_M1"
    elif (
        m2m0.delta_MAE>0 and m2m0.delta_RMSE>0 and m2m0.delta_mean_poisson_deviance>0
        and m2m1.delta_MAE>0 and m2m1.delta_RMSE>0 and m2m1.delta_mean_poisson_deviance>0
    ):
        descriptive="M2_HIGHER_POOLED_MAE_RMSE_DEVIANCE_THAN_M0_AND_M1"
    else:
        descriptive="MIXED_POOLED_M0_M2_PATTERN"

    lineage={
        "R10_R7_summary":str(psumm),"R10_R7_summary_sha256":sha256_file(psumm),
        "repaired_prediction_file":str(predp),"repaired_prediction_sha256":sha256_file(predp),
        "V18B_R10_stage":str(sadustage),
        "V18B_R10_summary":str(sadusumm),"V18B_R10_summary_sha256":sha256_file(sadusumm),
        "SADU_canonical_file":str(sadup),"SADU_canonical_sha256":sha256_file(sadup),
        "SADU_logical_hash":smeta["logical_hash"],
        "SADU_equivalent_paths":smeta["equivalent_paths"],
    }
    (out/"EXTERNAL_2025_REPAIRED_SCORE_INPUT_LINEAGE_SHA256.json").write_text(
        json.dumps(lineage,indent=2,ensure_ascii=False)
    )

    summary={
        "status":PASS,
        "package_revision":REV,
        "descriptive_classification":descriptive,
        "pooled_metrics":pooled.to_dict("records"),
        "pooled_deltas":pooled_delta.to_dict("records"),
        "regional_improvement_counts":regional_summary.to_dict("records"),
        "model_refit":False,"recalibration":False,"model_selection":False,
        "bootstrap_run":False,
        "scientific_inference_authorized":False,
        "M3":"DEFERRED_GRD_2025_NOT_YET_PUBLISHED",
        "next_stage":"V18C_R6_PAIRED_MOVING_BLOCK_BOOTSTRAP_EXTERNAL_2025_M0_M2",
    }
    (out/"B7B15H2BR_V18C_R5_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)
    )

    print("\ndescriptive classification".ljust(68)+f": {descriptive}")
    print("scientific inference authorized".ljust(68)+": False — bootstrap not yet run")
    print("\n"+PASS)

if __name__=="__main__":
    main()
