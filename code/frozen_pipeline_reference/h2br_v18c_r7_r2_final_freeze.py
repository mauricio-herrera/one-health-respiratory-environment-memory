#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import pandas as pd

REV="V18C_R7_R2_20260825W"
DEFAULT_PROJECT=Path("/PATH/TO/ONE_HEALTH_PROJECT")

R5_OUT="stage34C3b7b15h2br_external_2025_repaired_m0_m2_score_OUTPUT_v18c_r5"
R5_PASS="PASS_EXTERNAL_2025_REPAIRED_M0_M2_SCORE_FROZEN_M3_REMAINS_DEFERRED"
R6_OUT="stage34C3b7b15h2br_external_2025_m0_m2_block_bootstrap_OUTPUT_v18c_r6"
R6_PASS="PASS_EXTERNAL_2025_M0_M2_PAIRED_MOVING_BLOCK_BOOTSTRAP_FROZEN_M3_GATE_REMAINS_DEFERRED"
PASS="PASS_EXTERNAL_2025_M0_M2_SUPPLEMENTAL_RESULT_FROZEN_INFERENTIALLY_UNRESOLVED_M3_DEFERRED"

COMPS=("M1-M0","M2-M0","M2-M1")
METRICS=("MAE","RMSE","DEVIANCE")

def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def require_inputs(project:Path):
    r5s=project/R5_OUT/"B7B15H2BR_V18C_R5_SUMMARY.json"
    r5m=project/R5_OUT/"EXTERNAL_2025_REPAIRED_M0_M2_METRICS.csv"
    r5d=project/R5_OUT/"EXTERNAL_2025_REPAIRED_M0_M2_PAIRWISE_DELTAS.csv"
    r5r=project/R5_OUT/"EXTERNAL_2025_REPAIRED_M0_M2_REGIONAL_IMPROVEMENT_COUNTS.csv"

    r6s=project/R6_OUT/"B7B15H2BR_V18C_R6_SUMMARY.json"
    r6ci=project/R6_OUT/"EXTERNAL_2025_M0_M2_BLOCK_BOOTSTRAP_CI95.csv"
    r6c=project/R6_OUT/"EXTERNAL_2025_M0_M2_BLOCK_BOOTSTRAP_COMPARISON_SUMMARY.csv"

    for p in (r5s,r5m,r5d,r5r,r6s,r6ci,r6c):
        if not p.exists():
            raise RuntimeError(f"required upstream artifact missing: {p}")

    r5obj=json.loads(r5s.read_text())
    r6obj=json.loads(r6s.read_text())
    if r5obj.get("status")!=R5_PASS:
        raise RuntimeError(f"R5 PASS missing: {r5obj.get('status')}")
    if r6obj.get("status")!=R6_PASS:
        raise RuntimeError(f"R6 PASS missing: {r6obj.get('status')}")
    if r6obj.get("original_M3_gate_status")!="DEFERRED_UNTIL_GRD_2025_AVAILABLE":
        raise RuntimeError(f"R6 M3 gate state changed: {r6obj.get('original_M3_gate_status')}")

    return {
        "r5_summary":r5s,"r5_metrics":r5m,"r5_deltas":r5d,"r5_regional":r5r,
        "r6_summary":r6s,"r6_ci":r6ci,"r6_comparison":r6c,
        "r5_obj":r5obj,"r6_obj":r6obj,
    }

def validate_r5(inp):
    metrics=pd.read_csv(inp["r5_metrics"],low_memory=False)
    deltas=pd.read_csv(inp["r5_deltas"],low_memory=False)
    regional=pd.read_csv(inp["r5_regional"],low_memory=False)

    pooled=metrics[metrics.scope=="POOLED"].copy()
    if set(pooled.model)!={"M0","M1","M2"} or len(pooled)!=3:
        raise RuntimeError("R5 pooled metric model set changed")
    if not (pooled.n.astype(int)==4015).all():
        raise RuntimeError("R5 pooled n changed")

    pdlt=deltas[deltas.scope=="POOLED"].copy()
    if set(pdlt.comparison)!=set(COMPS) or len(pdlt)!=3:
        raise RuntimeError("R5 pooled comparisons changed")

    losscols=("delta_MAE","delta_RMSE","delta_mean_poisson_deviance")
    for c in losscols:
        if not (pd.to_numeric(pdlt[c],errors="raise")<0).all():
            raise RuntimeError(f"R5 descriptive pooled sign changed for {c}")

    r20=regional[regional.comparison=="M2-M0"]
    if len(r20)!=1:
        raise RuntimeError("R5 M2-M0 regional summary missing/not unique")
    rr=r20.iloc[0]
    if int(rr.regions_total)!=11 or int(rr.regions_improve_all_three)!=6:
        raise RuntimeError(
            f"R5 M2-M0 regional all-three count changed: "
            f"{rr.regions_improve_all_three}/{rr.regions_total}"
        )

    # FIX R7_R2: freeze a comparison-keyed view explicitly.
    pdlt_indexed=pdlt.set_index("comparison",drop=False)
    if not pdlt_indexed.index.is_unique:
        raise RuntimeError("R5 pooled comparison index not unique")
    return metrics,deltas,regional,pdlt_indexed

def validate_r6(inp):
    ci=pd.read_csv(inp["r6_ci"],low_memory=False)
    comp=pd.read_csv(inp["r6_comparison"],low_memory=False)

    if len(ci)!=9 or set(ci.comparison)!=set(COMPS) or set(ci.metric)!=set(METRICS):
        raise RuntimeError(
            f"R6 CI grid changed rows={len(ci)} comparisons={set(ci.comparison)} metrics={set(ci.metric)}"
        )

    if not ci.ci_classification.eq("CI_INCLUDES_ZERO").all():
        raise RuntimeError(
            "R6 inferential pattern changed: expected all nine frozen CIs to include zero"
        )
    lo=pd.to_numeric(ci.ci95_lower_percentile,errors="raise")
    hi=pd.to_numeric(ci.ci95_upper_percentile,errors="raise")
    if not ((lo<0)&(hi>0)).all():
        raise RuntimeError("R6 percentile endpoints do not straddle zero for all nine CIs")

    if len(comp)!=3 or set(comp.comparison)!=set(COMPS):
        raise RuntimeError("R6 comparison summary changed")
    boolcols=[
        "MAE_CI_below_zero","RMSE_CI_below_zero","deviance_CI_below_zero",
        "all_three_CIs_below_zero","MAE_and_deviance_CIs_below_zero",
    ]
    for c in boolcols:
        if c not in comp.columns:
            raise RuntimeError(f"R6 comparison boolean missing: {c}")
        vals=comp[c].astype(str).str.lower().isin({"true","1","yes"})
        if vals.any():
            raise RuntimeError(f"R6 comparison inferential flag unexpectedly true: {c}")

    return ci,comp

def evidence_table(metrics,deltas,regional,ci):
    pdlt=deltas[deltas.scope=="POOLED"].copy().set_index("comparison",drop=False)
    reg=regional.set_index("comparison",drop=False)
    rows=[]
    for comp in COMPS:
        a,b=comp.split("-")
        for metric,point_col in [
            ("MAE","delta_MAE"),
            ("RMSE","delta_RMSE"),
            ("DEVIANCE","delta_mean_poisson_deviance"),
        ]:
            rr=ci[(ci.comparison==comp)&(ci.metric==metric)].iloc[0]
            rows.append({
                "comparison":comp,
                "metric":metric,
                "model_first":a,
                "model_reference":b,
                "point_delta":float(pdlt.loc[comp,point_col]),
                "point_direction":"LOWER_LOSS_FIRST_MODEL" if float(pdlt.loc[comp,point_col])<0 else "HIGHER_LOSS_FIRST_MODEL",
                "ci95_lower":float(rr.ci95_lower_percentile),
                "ci95_upper":float(rr.ci95_upper_percentile),
                "ci_classification":rr.ci_classification,
                "inferential_conclusion":"NO_BOOTSTRAP_SUPPORTED_LOWER_LOSS",
                "regions_total":int(reg.loc[comp,"regions_total"]),
                "regions_improve_all_three":int(reg.loc[comp,"regions_improve_all_three"]),
            })
    return pd.DataFrame(rows)

def build_final_table(pdlt_indexed,regional,ci):
    reg=regional.set_index("comparison",drop=False)
    rows=[]
    for compname in COMPS:
        d=pdlt_indexed.loc[compname]
        rg=reg.loc[compname]
        cg=ci[ci.comparison==compname]
        mae=cg[cg.metric=="MAE"].iloc[0]
        rmse=cg[cg.metric=="RMSE"].iloc[0]
        dev=cg[cg.metric=="DEVIANCE"].iloc[0]
        rows.append({
            "comparison":compname,
            "delta_MAE":float(d.delta_MAE),
            "delta_RMSE":float(d.delta_RMSE),
            "delta_mean_poisson_deviance":float(d.delta_mean_poisson_deviance),
            "pooled_all_three_point_deltas_below_zero":bool(
                d.delta_MAE<0 and d.delta_RMSE<0 and d.delta_mean_poisson_deviance<0
            ),
            "regions_improve_all_three":int(rg.regions_improve_all_three),
            "regions_total":int(rg.regions_total),
            "MAE_CI95":f"[{mae.ci95_lower_percentile}, {mae.ci95_upper_percentile}]",
            "RMSE_CI95":f"[{rmse.ci95_lower_percentile}, {rmse.ci95_upper_percentile}]",
            "DEVIANCE_CI95":f"[{dev.ci95_lower_percentile}, {dev.ci95_upper_percentile}]",
            "all_three_CIs_include_zero":bool(cg.ci_classification.eq("CI_INCLUDES_ZERO").all()),
            "final_interpretation":"DESCRIPTIVE_LOWER_POOLED_LOSS_BUT_BOOTSTRAP_CIS_INCLUDE_ZERO",
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=DEFAULT_PROJECT)
    ap.add_argument("--outdir","--output-dir",dest="outdir",type=Path,default=None)
    a=ap.parse_args()
    project=a.project_root.resolve()
    out=(a.outdir or project/"stage34C3b7b15h2br_external_2025_m0_m2_final_interpretation_freeze_OUTPUT_v18c_r7_r2").resolve()
    out.mkdir(parents=True,exist_ok=True)

    print("="*190)
    print("STAGE 34C.3b.7B15H2BR V18C_R7_R2 — FINAL EXTERNAL 2025 M0–M2 INTERPRETATION FREEZE")
    print(f"PACKAGE REVISION                         : {REV}")
    print("="*190)
    print("patch                                     : comparison-index bug in R7 freeze harness only")
    print("new model fit / tuning                    : FORBIDDEN")
    print("new bootstrap / inferential test          : FORBIDDEN")
    print("upstream R5/R6 results                    : READ-ONLY / UNCHANGED")

    inp=require_inputs(project)
    metrics,deltas,regional,pdlt=validate_r5(inp)
    ci,comp=validate_r6(inp)

    evidence=evidence_table(metrics,deltas,regional,ci)
    evidence.to_csv(out/"FROZEN_EXTERNAL_2025_M0_M2_EVIDENCE_TABLE.csv",index=False)

    final=build_final_table(pdlt,regional,ci)
    final.to_csv(out/"FROZEN_EXTERNAL_2025_M0_M2_FINAL_COMPARISON_SUMMARY.csv",index=False)

    m2m0=final[final.comparison=="M2-M0"].iloc[0]
    if not (
        bool(m2m0.pooled_all_three_point_deltas_below_zero)
        and int(m2m0.regions_improve_all_three)==6
        and bool(m2m0.all_three_CIs_include_zero)
    ):
        raise RuntimeError("M2-M0 final freeze facts changed unexpectedly")

    interpretation={
        "M0_M2_external_status":"CLOSED_SUPPLEMENTAL",
        "primary_external_2025_message":(
            "M2 has lower pooled 2025 MAE, RMSE, and mean Poisson deviance than M0 and M1, "
            "but all paired 95% seven-day moving-block-bootstrap confidence intervals include zero. "
            "Therefore the pooled improvement is descriptive and uncertainty remains compatible with no loss difference."
        ),
        "M2_vs_M0":{
            "point_deltas":{
                "MAE":float(m2m0.delta_MAE),
                "RMSE":float(m2m0.delta_RMSE),
                "mean_poisson_deviance":float(m2m0.delta_mean_poisson_deviance),
            },
            "regions_improve_all_three":"6/11",
            "bootstrap_conclusion":"ALL_THREE_95PCT_CIS_INCLUDE_ZERO",
            "formal_external_superiority_claim":False,
        },
        "interpretive_boundary":{
            "M0_M2_result_role":"SUPPLEMENTAL_EXTERNAL_VALIDATION",
            "original_formal_success_gate":"M3_ONLY",
            "M3_gate_status":"DEFERRED_UNTIL_GRD_2025_AVAILABLE",
            "M0_M2_result_does_not_count_as_M3_gate_failure":True,
            "EWS_claim":False,
            "causal_claim":False,
            "weather_forecast_claim":False,
            "model_retuning_authorized":False,
            "post_hoc_model_selection_authorized":False,
        },
        "mainline_next_state":"WAIT_FOR_GRD_2025_THEN_EXECUTE_FROZEN_M3_SEQUENTIAL_EXTERNAL_PROTOCOL",
    }
    (out/"FROZEN_EXTERNAL_2025_M0_M2_INTERPRETATION.json").write_text(
        json.dumps(interpretation,indent=2,ensure_ascii=False)
    )

    md=f"""# Frozen external 2025 M0–M2 interpretation

## Status

**M0–M2 external validation: CLOSED / SUPPLEMENTAL.**  
**M3 external gate: DEFERRED until GRD 2025 becomes available.**

## Frozen result

The repaired prospective 2019–2024 fits generated distinct, health-blind 2025
predictions for M0, M1 and M2. In the frozen 2025 SADU external outcome, M2 had
lower pooled MAE, RMSE and mean Poisson deviance than both M0 and M1.

For M2 versus M0:

- ΔMAE = {float(m2m0.delta_MAE):.12g}
- ΔRMSE = {float(m2m0.delta_RMSE):.12g}
- Δmean Poisson deviance = {float(m2m0.delta_mean_poisson_deviance):.12g}
- regions improving all three losses = 6/11

However, the paired seven-day moving-block bootstrap (B=5000, seed=20260822)
produced 95% percentile confidence intervals that include zero for MAE, RMSE and
mean Poisson deviance. The same is true for all nine M1−M0, M2−M0 and M2−M1
model×loss comparisons.

## Interpretation

The external 2025 result supports a **descriptive pooled ranking** in which M2
has lower loss, but it does **not** establish bootstrap-supported superiority of
M2 over M0 or M1.

This does not count as failure of the original formal M3 external success gate.
That gate remains deferred until GRD 2025 becomes available.

No EWS, causal, weather-forecasting, post-hoc retuning, or post-hoc model-
selection claim is authorized.
"""
    (out/"FROZEN_EXTERNAL_2025_M0_M2_INTERPRETATION.md").write_text(md)

    lineage={}
    for k in ("r5_summary","r5_metrics","r5_deltas","r5_regional","r6_summary","r6_ci","r6_comparison"):
        p=inp[k]
        lineage[k]={"path":str(p),"sha256":sha256_file(p)}
    (out/"FINAL_EXTERNAL_2025_M0_M2_LINEAGE_SHA256.json").write_text(
        json.dumps(lineage,indent=2,ensure_ascii=False)
    )

    gate=pd.DataFrame([
        {"gate":"R5_PASS","value":True},
        {"gate":"R6_PASS","value":True},
        {"gate":"comparison_index_unique","value":bool(pdlt.index.is_unique)},
        {"gate":"comparison_index_exact","value":bool(set(pdlt.index)==set(COMPS))},
        {"gate":"M2_M0_point_MAE_below_zero","value":bool(m2m0.delta_MAE<0)},
        {"gate":"M2_M0_point_RMSE_below_zero","value":bool(m2m0.delta_RMSE<0)},
        {"gate":"M2_M0_point_deviance_below_zero","value":bool(m2m0.delta_mean_poisson_deviance<0)},
        {"gate":"M2_M0_regions_all_three_exactly_6_of_11","value":bool(int(m2m0.regions_improve_all_three)==6)},
        {"gate":"all_9_bootstrap_CIs_include_zero","value":bool(ci.ci_classification.eq("CI_INCLUDES_ZERO").all())},
        {"gate":"M3_formal_gate_remains_deferred","value":True},
        {"gate":"no_posthoc_tuning","value":True},
    ])
    gate.to_csv(out/"FINAL_EXTERNAL_2025_M0_M2_FREEZE_GATE_AUDIT.csv",index=False)
    if not gate.value.all():
        raise RuntimeError(f"final interpretation freeze gates failed: {gate[~gate.value].to_dict('records')}")

    summary={
        "status":PASS,
        "package_revision":REV,
        "patch":"R7 comparison-index bug only; upstream R5/R6 unchanged",
        "M0_M2_external_status":"CLOSED_SUPPLEMENTAL",
        "M2_vs_M0_point_delta_MAE":float(m2m0.delta_MAE),
        "M2_vs_M0_point_delta_RMSE":float(m2m0.delta_RMSE),
        "M2_vs_M0_point_delta_mean_poisson_deviance":float(m2m0.delta_mean_poisson_deviance),
        "M2_vs_M0_regions_improve_all_three":"6/11",
        "all_nine_95pct_block_bootstrap_CIs_include_zero":True,
        "external_M0_M2_superiority_established":False,
        "original_formal_success_gate":"M3",
        "M3_gate_status":"DEFERRED_UNTIL_GRD_2025_AVAILABLE",
        "EWS_claim_authorized":False,
        "causal_claim_authorized":False,
        "posthoc_tuning_authorized":False,
        "mainline_next_state":"WAIT_FOR_GRD_2025_THEN_EXECUTE_FROZEN_M3_SEQUENTIAL_EXTERNAL_PROTOCOL",
    }
    (out/"B7B15H2BR_V18C_R7_R2_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)
    )

    print("\nFrozen external M0–M2 interpretation")
    print(final.to_string(index=False))
    print("\nM0–M2 external status".ljust(72)+": CLOSED / SUPPLEMENTAL")
    print("M2 pooled descriptive ranking".ljust(72)+": LOWER MAE / RMSE / DEVIANCE")
    print("bootstrap-supported superiority".ljust(72)+": NOT ESTABLISHED — all 9 CIs include zero")
    print("M2-M0 regions improving all three losses".ljust(72)+": 6/11")
    print("original M3 formal success gate".ljust(72)+": DEFERRED UNTIL GRD 2025")
    print("post-hoc M0–M2 tuning".ljust(72)+": FORBIDDEN")
    print("\n"+PASS)

if __name__=="__main__":
    main()
