#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd

REV="V18C_R6_20260825U"
DEFAULT_PROJECT=Path("/PATH/TO/ONE_HEALTH_PROJECT")

R5_OUT="stage34C3b7b15h2br_external_2025_repaired_m0_m2_score_OUTPUT_v18c_r5"
R5_PASS="PASS_EXTERNAL_2025_REPAIRED_M0_M2_SCORE_FROZEN_M3_REMAINS_DEFERRED"
PASS="PASS_EXTERNAL_2025_M0_M2_PAIRED_MOVING_BLOCK_BOOTSTRAP_FROZEN_M3_GATE_REMAINS_DEFERRED"

BLOCK_DAYS=7
B=5000
SEED=20260822
ALPHA=0.05
COMPARISONS=(("M1","M0"),("M2","M0"),("M2","M1"))
MODELS=("M0","M1","M2")

def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def require_r5(project:Path):
    summ=project/R5_OUT/"B7B15H2BR_V18C_R5_SUMMARY.json"
    daily=project/R5_OUT/"EXTERNAL_2025_REPAIRED_M0_M2_DAILY_PAIRED_LOSSES.csv"
    metrics=project/R5_OUT/"EXTERNAL_2025_REPAIRED_M0_M2_METRICS.csv"
    deltas=project/R5_OUT/"EXTERNAL_2025_REPAIRED_M0_M2_PAIRWISE_DELTAS.csv"
    obs=project/R5_OUT/"EXTERNAL_2025_REPAIRED_M0_M2_OBSERVATION_LOSSES.parquet"
    for p in (summ,daily,metrics,deltas,obs):
        if not p.exists():raise RuntimeError(f"R5 artifact missing: {p}")
    obj=json.loads(summ.read_text())
    if obj.get("status")!=R5_PASS:
        raise RuntimeError(f"R5 PASS missing: {obj.get('status')}")
    if obj.get("bootstrap_run") is not False:
        raise RuntimeError("R5 bootstrap taint guard changed")
    return summ,obj,daily,metrics,deltas,obs

def load_daily(p:Path):
    x=pd.read_csv(p,low_memory=False)
    if "date" not in x.columns:
        raise RuntimeError("daily losses missing date")
    x["date"]=pd.to_datetime(x.date,errors="coerce").dt.normalize()
    x=x.sort_values("date").reset_index(drop=True)
    if len(x)!=365 or x.date.nunique()!=365:
        raise RuntimeError(f"daily panel expected 365 unique days, got {len(x)}/{x.date.nunique()}")
    if "n_units" not in x.columns or not pd.to_numeric(x.n_units,errors="coerce").eq(11).all():
        raise RuntimeError("daily paired losses do not preserve 11 units/day")
    if "n_rows" not in x.columns or not pd.to_numeric(x.n_rows,errors="coerce").eq(11).all():
        raise RuntimeError("daily paired losses do not contain exactly 11 rows/day")
    need=set()
    for m in MODELS:
        need|={f"{m}_MAE_day",f"{m}_MSE_day",f"{m}_deviance_day"}
    missing=sorted(need-set(x.columns))
    if missing:raise RuntimeError(f"daily paired loss columns missing: {missing}")
    for c in need:
        v=pd.to_numeric(x[c],errors="coerce")
        if v.isna().any() or not np.isfinite(v.to_numpy(float)).all():
            raise RuntimeError(f"invalid daily loss column: {c}")
        x[c]=v.astype(float)
    return x

def load_r5_point_deltas(p:Path):
    x=pd.read_csv(p,low_memory=False)
    pooled=x[x.scope=="POOLED"].copy()
    if len(pooled)!=3:
        raise RuntimeError(f"R5 pooled delta rows expected 3, got {len(pooled)}")
    return pooled

def metric_from_day_indices(daily:pd.DataFrame,idx:np.ndarray):
    """
    Each calendar day contributes exactly 11 units, so equal day weighting is
    equivalent to equal observation weighting over the resampled 11-region panel.
    """
    out={}
    for m in MODELS:
        mae=float(np.mean(daily[f"{m}_MAE_day"].to_numpy(float)[idx]))
        mse=float(np.mean(daily[f"{m}_MSE_day"].to_numpy(float)[idx]))
        dev=float(np.mean(daily[f"{m}_deviance_day"].to_numpy(float)[idx]))
        out[m]={"MAE":mae,"RMSE":float(np.sqrt(mse)),"DEVIANCE":dev}
    return out

def point_metrics_from_daily(daily):
    idx=np.arange(len(daily),dtype=int)
    return metric_from_day_indices(daily,idx)

def point_delta_rows(point):
    rows=[]
    for a,b in COMPARISONS:
        rows.append({
            "comparison":f"{a}-{b}",
            "delta_MAE":point[a]["MAE"]-point[b]["MAE"],
            "delta_RMSE":point[a]["RMSE"]-point[b]["RMSE"],
            "delta_mean_poisson_deviance":point[a]["DEVIANCE"]-point[b]["DEVIANCE"],
        })
    return pd.DataFrame(rows)

def validate_point_against_r5(point_delta,r5_delta):
    rows=[]
    for comp in point_delta.comparison:
        a=point_delta[point_delta.comparison==comp].iloc[0]
        b=r5_delta[r5_delta.comparison==comp].iloc[0]
        for k in ("delta_MAE","delta_RMSE","delta_mean_poisson_deviance"):
            d=float(a[k]-b[k])
            tol=1e-10*max(1.0,abs(float(b[k])))
            rows.append({
                "comparison":comp,"metric":k,
                "reconstructed_from_daily":float(a[k]),
                "R5_frozen_value":float(b[k]),
                "difference":d,"tolerance":tol,
                "matches":bool(abs(d)<=tol),
            })
    audit=pd.DataFrame(rows)
    if not audit.matches.all():
        raise RuntimeError(f"R5 point estimates not exactly reconstructable from daily losses: {audit.to_dict('records')}")
    return audit

def moving_block_indices(n:int,block:int,rng:np.random.Generator):
    """
    Standard non-circular moving-block bootstrap.
    Sample block starts uniformly from 0..n-block inclusive, concatenate blocks,
    truncate to exactly n resampled days.
    """
    if block<1 or block>n:
        raise ValueError("invalid block length")
    nblocks=int(np.ceil(n/block))
    starts=rng.integers(0,n-block+1,size=nblocks)
    idx=np.concatenate([np.arange(s,s+block,dtype=int) for s in starts])[:n]
    return idx,starts

def bootstrap(daily):
    n=len(daily)
    rng=np.random.default_rng(SEED)
    records=[]
    starts_records=[]
    for b in range(B):
        idx,starts=moving_block_indices(n,BLOCK_DAYS,rng)
        met=metric_from_day_indices(daily,idx)
        rec={"replicate":b}
        for a,c in COMPARISONS:
            comp=f"{a}-{c}"
            rec[f"{comp}_delta_MAE"]=met[a]["MAE"]-met[c]["MAE"]
            rec[f"{comp}_delta_RMSE"]=met[a]["RMSE"]-met[c]["RMSE"]
            rec[f"{comp}_delta_deviance"]=met[a]["DEVIANCE"]-met[c]["DEVIANCE"]
        records.append(rec)
        starts_records.append({
            "replicate":b,
            "block_starts_zero_based":"|".join(str(int(s)) for s in starts),
            "n_blocks_drawn":len(starts),
            "resampled_days":len(idx),
        })
    return pd.DataFrame(records),pd.DataFrame(starts_records)

def percentile_ci(v):
    v=np.asarray(v,float)
    lo=float(np.quantile(v,ALPHA/2,method="linear"))
    hi=float(np.quantile(v,1-ALPHA/2,method="linear"))
    return lo,hi

def bootstrap_summary(boot,point_delta):
    rows=[]
    mapping={
        "MAE":("delta_MAE","delta_MAE"),
        "RMSE":("delta_RMSE","delta_RMSE"),
        "DEVIANCE":("delta_deviance","delta_mean_poisson_deviance"),
    }
    for comp in [f"{a}-{b}" for a,b in COMPARISONS]:
        prow=point_delta[point_delta.comparison==comp].iloc[0]
        for metric,(boot_suffix,point_col) in mapping.items():
            v=boot[f"{comp}_{boot_suffix}"].to_numpy(float)
            lo,hi=percentile_ci(v)
            point=float(prow[point_col])
            p_le_zero=float(np.mean(v<=0))
            p_ge_zero=float(np.mean(v>=0))
            # descriptive bootstrap sign probability, not a formal p-value.
            if hi<0:
                cls="CI_ENTIRELY_BELOW_ZERO"
            elif lo>0:
                cls="CI_ENTIRELY_ABOVE_ZERO"
            else:
                cls="CI_INCLUDES_ZERO"
            rows.append({
                "comparison":comp,
                "metric":metric,
                "point_delta":point,
                "bootstrap_mean":float(np.mean(v)),
                "bootstrap_sd":float(np.std(v,ddof=1)),
                "ci95_lower_percentile":lo,
                "ci95_upper_percentile":hi,
                "ci_classification":cls,
                "bootstrap_fraction_le_zero":p_le_zero,
                "bootstrap_fraction_ge_zero":p_ge_zero,
                "B":B,"block_days":BLOCK_DAYS,"seed":SEED,
            })
    return pd.DataFrame(rows)

def comparison_summary(ci):
    rows=[]
    for comp,g in ci.groupby("comparison"):
        by={r.metric:r for _,r in g.iterrows()}
        mae_below=(by["MAE"].ci_classification=="CI_ENTIRELY_BELOW_ZERO")
        rmse_below=(by["RMSE"].ci_classification=="CI_ENTIRELY_BELOW_ZERO")
        dev_below=(by["DEVIANCE"].ci_classification=="CI_ENTIRELY_BELOW_ZERO")
        rows.append({
            "comparison":comp,
            "MAE_CI_below_zero":mae_below,
            "RMSE_CI_below_zero":rmse_below,
            "deviance_CI_below_zero":dev_below,
            "all_three_CIs_below_zero":bool(mae_below and rmse_below and dev_below),
            "MAE_and_deviance_CIs_below_zero":bool(mae_below and dev_below),
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=DEFAULT_PROJECT)
    ap.add_argument("--outdir","--output-dir",dest="outdir",type=Path,default=None)
    a=ap.parse_args()
    project=a.project_root.resolve()
    out=(a.outdir or project/"stage34C3b7b15h2br_external_2025_m0_m2_block_bootstrap_OUTPUT_v18c_r6").resolve()
    out.mkdir(parents=True,exist_ok=True)

    print("="*190)
    print("STAGE 34C.3b.7B15H2BR V18C_R6 — PAIRED MOVING-BLOCK BOOTSTRAP, EXTERNAL 2025 M0–M2")
    print(f"PACKAGE REVISION                         : {REV}")
    print("="*190)
    print(f"block length                              : {BLOCK_DAYS} calendar days")
    print(f"bootstrap replicates                      : {B}")
    print(f"seed                                      : {SEED}")
    print("resampling unit                           : complete calendar day with all 11 regions preserved")
    print("bootstrap type                            : standard non-circular moving-block bootstrap")
    print("CI                                        : two-sided 95% percentile")
    print("M0–M2 role                                : supplemental external inference; original formal success gate belongs to deferred M3")

    summ,sobj,dailyp,metricsp,deltap,obsp=require_r5(project)
    daily=load_daily(dailyp)
    r5_delta=load_r5_point_deltas(deltap)

    point=point_metrics_from_daily(daily)
    point_delta=point_delta_rows(point)
    point_delta.to_csv(out/"EXTERNAL_2025_M0_M2_POINT_DELTAS_RECONSTRUCTED_FROM_DAILY_LOSSES.csv",index=False)

    point_audit=validate_point_against_r5(point_delta,r5_delta)
    point_audit.to_csv(out/"R5_POINT_ESTIMATE_RECONSTRUCTION_AUDIT.csv",index=False)

    boot,starts=bootstrap(daily)
    boot.to_parquet(out/"EXTERNAL_2025_M0_M2_BLOCK_BOOTSTRAP_REPLICATES.parquet",index=False)
    starts.to_csv(out/"EXTERNAL_2025_M0_M2_BLOCK_BOOTSTRAP_STARTS.csv",index=False)

    ci=bootstrap_summary(boot,point_delta)
    ci.to_csv(out/"EXTERNAL_2025_M0_M2_BLOCK_BOOTSTRAP_CI95.csv",index=False)
    comp=comparison_summary(ci)
    comp.to_csv(out/"EXTERNAL_2025_M0_M2_BLOCK_BOOTSTRAP_COMPARISON_SUMMARY.csv",index=False)

    print("R5 point estimates reconstructed from daily losses".ljust(72)+": PASS")
    print("\n95% moving-block-bootstrap CIs for paired deltas")
    print(ci[[
        "comparison","metric","point_delta","bootstrap_mean","bootstrap_sd",
        "ci95_lower_percentile","ci95_upper_percentile","ci_classification"
    ]].to_string(index=False,float_format=lambda x:f"{x:.12g}"))

    print("\nComparison-level CI summary")
    print(comp.to_string(index=False))

    # Conservative descriptive classification.
    m2m0=comp[comp.comparison=="M2-M0"].iloc[0]
    m2m1=comp[comp.comparison=="M2-M1"].iloc[0]
    if m2m0.all_three_CIs_below_zero and m2m1.all_three_CIs_below_zero:
        overall="M2_BOOTSTRAP_CIS_BELOW_ZERO_FOR_ALL_THREE_LOSSES_VS_M0_AND_M1"
    elif m2m0.MAE_and_deviance_CIs_below_zero:
        overall="M2_VS_M0_MAE_AND_DEVIANCE_CIS_BELOW_ZERO_OTHER_COMPARISONS_MIXED"
    elif m2m0.MAE_CI_below_zero or m2m0.deviance_CI_below_zero or m2m0.RMSE_CI_below_zero:
        overall="M2_VS_M0_AT_LEAST_ONE_LOSS_CI_BELOW_ZERO_BUT_NOT_ALL"
    else:
        overall="M2_VS_M0_CIS_DO_NOT_ESTABLISH_LOWER_LOSS"

    lineage={
        "R5_summary":str(summ),"R5_summary_sha256":sha256_file(summ),
        "R5_daily_paired_losses":str(dailyp),"R5_daily_paired_losses_sha256":sha256_file(dailyp),
        "R5_metrics":str(metricsp),"R5_metrics_sha256":sha256_file(metricsp),
        "R5_pairwise_deltas":str(deltap),"R5_pairwise_deltas_sha256":sha256_file(deltap),
        "R5_observation_losses":str(obsp),"R5_observation_losses_sha256":sha256_file(obsp),
    }
    (out/"EXTERNAL_2025_M0_M2_BOOTSTRAP_INPUT_LINEAGE_SHA256.json").write_text(
        json.dumps(lineage,indent=2,ensure_ascii=False)
    )

    summary={
        "status":PASS,
        "package_revision":REV,
        "block_days":BLOCK_DAYS,"B":B,"seed":SEED,
        "bootstrap_type":"standard non-circular moving-block bootstrap",
        "resampling_unit":"calendar day; all 11 regions preserved together",
        "ci":"two-sided 95% percentile",
        "overall_descriptive_bootstrap_classification":overall,
        "CI_results":ci.to_dict("records"),
        "comparison_summary":comp.to_dict("records"),
        "R5_point_reconstruction_pass":True,
        "model_refit":False,"prediction_recalculation":False,
        "M0_M2_inference_role":"SUPPLEMENTAL_EXTERNAL_INFERENCE_NOT_ORIGINAL_FORMAL_SUCCESS_GATE",
        "original_M3_gate_status":"DEFERRED_UNTIL_GRD_2025_AVAILABLE",
        "EWS_claim_authorized":False,
        "causal_claim_authorized":False,
        "next_stage":"V18C_R7_EXTERNAL_2025_M0_M2_FINAL_INTERPRETATION_FREEZE_AND_MAINLINE_WAIT_FOR_M3",
    }
    (out/"B7B15H2BR_V18C_R6_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)
    )

    print("\noverall bootstrap classification".ljust(72)+f": {overall}")
    print("original formal M3 success gate".ljust(72)+": DEFERRED")
    print("EWS / causal claim".ljust(72)+": NOT AUTHORIZED")
    print("\n"+PASS)

if __name__=="__main__":
    main()
