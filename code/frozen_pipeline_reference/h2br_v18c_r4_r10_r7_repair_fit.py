#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

REV="V18C_R4_R10_R7_20260825S"
DEFAULT_PROJECT=Path("/PATH/TO/ONE_HEALTH_PROJECT")

R10R6_OUT="stage34C3b7b15h2br_prospective_full_history_fit_contract_OUTPUT_v18c_r4_r10_r6"
R10R6_PASS="PASS_PROSPECTIVE_FULL_HISTORY_FIT_CONTRACT_FROZEN_FROM_ACCEPTED_H2BA_NO_FIT_2025_HEALTH_BLIND_NEXT_FIT_STAGE_AUTHORIZED"
R9_OUT="stage34C3b7b15h2br_v15e_historical_fit_degeneracy_audit_OUTPUT_v18c_r4_r9"
R9_PASS="PASS_V15E_HISTORICAL_FIT_DEGENERACY_MECHANISM_AUDITED_NO_REFIT"
R7_OUT="stage34C3b7b15h2br_v15e_exact_norefit_replay_OUTPUT_v18c_r4_r7"
PASS="PASS_PROSPECTIVE_FULL_HISTORY_REPAIR_FIT_M0_M2_FROZEN_DISTINCT_2025_PREDICTIONS_READY_FOR_EXTERNAL_RESCORING"

V15E_DIRNAME="STAGE34C3B7B15H2BR_FINAL_PREUNSEAL_MODEL_AND_M3_SEQUENTIAL_FREEZE_V15E"
TARGET="urgent_respiratory_total"
EXPOSURE="population"

MODELS=("M0","M1","M2")
EXPECTED_COUNTS={"M0":27,"M1":34,"M2":54}

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

def require_contract(project:Path):
    p=project/R10R6_OUT/"B7B15H2BR_V18C_R4_R10_R6_SUMMARY.json"
    if not p.exists():raise RuntimeError(f"R10_R6 summary missing: {p}")
    obj=json.loads(p.read_text())
    if obj.get("status")!=R10R6_PASS:
        raise RuntimeError(f"R10_R6 PASS missing: {obj.get('status')}")
    contract=project/R10R6_OUT/"FROZEN_PROSPECTIVE_FULL_HISTORY_FIT_CONTRACT.json"
    order=project/R10R6_OUT/"FROZEN_PROSPECTIVE_FULL_HISTORY_M0_M3_FEATURE_ORDER.csv"
    if not contract.exists() or not order.exists():
        raise RuntimeError("R10_R6 frozen contract artifacts missing")
    return p,obj,contract,order

def require_r9_inputs(project:Path):
    p=project/R9_OUT/"B7B15H2BR_V18C_R4_R9_SUMMARY.json"
    if not p.exists():raise RuntimeError(f"R9 summary missing: {p}")
    obj=json.loads(p.read_text())
    if obj.get("status")!=R9_PASS:
        raise RuntimeError(f"R9 PASS missing: {obj.get('status')}")
    roles=obj.get("historical_inputs") or {}
    h2w=Path(roles.get("H2W_canonical_48_design",""))
    g3=Path(roles.get("G3_canonical_2019_2024_health_and_exposure",""))
    if not h2w.exists() or not g3.exists():
        raise RuntimeError(f"historical source paths missing h2w={h2w} g3={g3}")
    return p,obj,h2w.resolve(),g3.resolve()

def require_v14i_design(project:Path):
    p=project/R7_OUT/"B7B15H2BR_V18C_R4_R7_SUMMARY.json"
    if not p.exists():raise RuntimeError(f"R7 summary missing: {p}")
    obj=json.loads(p.read_text())
    meta=obj.get("v14i_design") or {}
    paths=[Path(s) for s in meta.get("physical_paths",[]) if Path(s).exists()]
    if not paths:
        raise RuntimeError("R7 summary has no accessible V14I design physical path")
    # R7 already proved logical equality. Use deterministic lexical first physical copy.
    paths=sorted({p.resolve() for p in paths},key=str)
    return p,obj,paths[0],paths

def load_feature_order(orderp:Path):
    x=pd.read_csv(orderp,low_memory=False)
    req={"model","feature_order","feature_n"}
    if not req.issubset(x.columns):
        raise RuntimeError(f"feature-order schema changed: {list(x.columns)}")
    out={}
    for m,nexp in EXPECTED_COUNTS.items():
        g=x[x.model==m].copy()
        g["feature_order"]=pd.to_numeric(g.feature_order,errors="raise")
        g=g.sort_values("feature_order")
        fs=g.feature_n.astype(str).map(norm).tolist()
        if len(fs)!=nexp or len(set(fs))!=nexp:
            raise RuntimeError(f"{m} feature order count invalid {len(fs)}/{len(set(fs))}")
        out[m]=fs
    return out,x

def normalize_keys(x:pd.DataFrame,name:str):
    if "date" not in x.columns or "unit" not in x.columns:
        raise RuntimeError(f"{name} missing date/unit")
    q=x.copy()
    q["date"]=pd.to_datetime(q["date"],errors="coerce").dt.normalize()
    q["unit"]=q["unit"].astype(str).str.strip()
    if q.date.isna().any():
        raise RuntimeError(f"{name} invalid dates")
    if q.duplicated(["date","unit"]).any():
        raise RuntimeError(f"{name} duplicate date/unit")
    return q

def historical_training(h2w:Path,g3:Path,feature_order):
    h=normalize_keys(read_table(h2w),"H2W")
    g=normalize_keys(read_table(g3),"G3")
    required_science=set()
    nuisance_prefixes=("dow_","unit_")
    for m in MODELS:
        required_science.update(
            f for f in feature_order[m]
            if not f.startswith(nuisance_prefixes)
        )

    cmap={norm(c):c for c in h.columns}
    missing=sorted(required_science-set(cmap))
    if missing:
        raise RuntimeError(f"H2W missing accepted scientific columns: {missing}")
    if TARGET not in g.columns or EXPOSURE not in g.columns:
        raise RuntimeError(f"G3 missing {TARGET}/{EXPOSURE}")

    hs=h[["date","unit"]+[cmap[f] for f in sorted(required_science)]].copy()
    hs=hs.rename(columns={cmap[f]:f for f in sorted(required_science)})
    gg=g[["date","unit",TARGET,EXPOSURE]].copy()
    train=hs.merge(gg,on=["date","unit"],how="inner",validate="one_to_one")
    train=train.sort_values(["date","unit"]).reset_index(drop=True)
    if len(train)!=35072 or train.date.nunique()!=2192 or train.unit.nunique()!=16:
        raise RuntimeError(
            f"historical grid changed rows/days/units={len(train)}/{train.date.nunique()}/{train.unit.nunique()}"
        )
    y=pd.to_numeric(train[TARGET],errors="coerce")
    e=pd.to_numeric(train[EXPOSURE],errors="coerce")
    if y.isna().any() or e.isna().any() or (y<0).any() or (e<=0).any():
        raise RuntimeError("historical target/exposure invalid")
    train[TARGET]=y.astype(float)
    train[EXPOSURE]=e.astype(float)

    for f in sorted(required_science):
        train[f]=pd.to_numeric(train[f],errors="coerce")
        if not np.isfinite(train[f].to_numpy(float)).all():
            raise RuntimeError(f"historical scientific feature nonfinite: {f}")
    return train,sorted(required_science)

def nuisance_design(train:pd.DataFrame,test:pd.DataFrame):
    tr_dow=train["date"].dt.dayofweek.astype(str)
    te_dow=test["date"].dt.dayofweek.astype(str)
    tr_unit=train["unit"].astype(str)
    te_unit=test["unit"].astype(str)

    A=pd.get_dummies(tr_dow,prefix="dow",drop_first=True,dtype=float)
    B=pd.get_dummies(te_dow,prefix="dow",drop_first=True,dtype=float).reindex(columns=A.columns,fill_value=0.0)
    C=pd.get_dummies(tr_unit,prefix="unit",drop_first=True,dtype=float)
    D=pd.get_dummies(te_unit,prefix="unit",drop_first=True,dtype=float).reindex(columns=C.columns,fill_value=0.0)

    Xn_tr=pd.concat([A.reset_index(drop=True),C.reset_index(drop=True)],axis=1)
    Xn_te=pd.concat([B.reset_index(drop=True),D.reset_index(drop=True)],axis=1)
    Xn_tr.columns=[norm(c) for c in Xn_tr.columns]
    Xn_te.columns=[norm(c) for c in Xn_te.columns]
    if list(Xn_tr.columns)!=list(Xn_te.columns):
        raise RuntimeError("train/test nuisance columns differ after accepted reindex rule")
    return Xn_tr,Xn_te

def load_2025_design(v14i:Path,feature_order):
    x=normalize_keys(read_table(v14i),"V14I")
    if len(x)!=4015 or x.date.nunique()!=365 or x.unit.nunique()!=11:
        raise RuntimeError(f"V14I grid changed {len(x)}/{x.date.nunique()}/{x.unit.nunique()}")

    forbidden=[
        c for c in x.columns
        if any(t in norm(c) for t in (
            "urgent_respiratory","respiratory_hospital","respiratory_death",
            "health_outcome","observed_outcome"
        ))
    ]
    if forbidden:
        raise RuntimeError(f"V14I is not health-blind; forbidden columns found: {forbidden}")

    required_science=set()
    for m in MODELS:
        required_science.update(
            f for f in feature_order[m]
            if not f.startswith(("dow_","unit_"))
        )
    cmap={norm(c):c for c in x.columns}
    missing=sorted(required_science-set(cmap))
    if missing:
        raise RuntimeError(f"V14I missing accepted scientific columns: {missing}")

    out=x[["date","unit"]+[cmap[f] for f in sorted(required_science)]].copy()
    out=out.rename(columns={cmap[f]:f for f in sorted(required_science)})
    for f in sorted(required_science):
        out[f]=pd.to_numeric(out[f],errors="coerce")
        if not np.isfinite(out[f].to_numpy(float)).all():
            raise RuntimeError(f"V14I feature nonfinite: {f}")
    return out.sort_values(["date","unit"]).reset_index(drop=True)

def load_2025_population(project:Path,domain:pd.DataFrame):
    p=project/V15E_DIRNAME/"FROZEN_2025_DAILY_EXPOSURE_POPULATION_4015.parquet"
    if not p.exists():raise RuntimeError(f"frozen 2025 population missing: {p}")
    x=normalize_keys(pd.read_parquet(p),"V15E population")
    if "population" not in x.columns:
        raise RuntimeError("frozen population column missing")
    q=x[["date","unit","population"]].copy()
    q["population"]=pd.to_numeric(q.population,errors="coerce")
    if len(q)!=4015 or q.population.isna().any() or (q.population<=0).any():
        raise RuntimeError("frozen 2025 population invalid")
    keys=domain[["date","unit"]].copy()
    q=keys.merge(q,on=["date","unit"],how="left",validate="one_to_one")
    if q.population.isna().any():
        raise RuntimeError("2025 population does not cover V14I domain")
    return q,p

def scientific_block(df:pd.DataFrame,features:list[str]):
    cols=[f for f in features if not f.startswith(("dow_","unit_"))]
    return df[cols].reset_index(drop=True).copy()

def build_exact_design(train,test,feature_order):
    Xn_tr,Xn_te=nuisance_design(train,test)
    designs={}
    audit=[]
    for m in MODELS:
        fs=feature_order[m]
        sci=[f for f in fs if not f.startswith(("dow_","unit_"))]
        Xs_tr=scientific_block(train,sci)
        Xs_te=scientific_block(test,sci)
        Xtr=pd.concat([Xs_tr,Xn_tr],axis=1)
        Xte=pd.concat([Xs_te,Xn_te],axis=1)

        # Accepted source constructs scientific block first, nuisance second, but
        # final order is source-frozen explicitly by R10_R6. Reorder to that exact sequence.
        missing_tr=[f for f in fs if f not in Xtr.columns]
        missing_te=[f for f in fs if f not in Xte.columns]
        if missing_tr or missing_te:
            raise RuntimeError(f"{m} exact design missing train={missing_tr} test={missing_te}")
        Xtr=Xtr[fs].copy()
        Xte=Xte[fs].copy()
        if list(Xtr.columns)!=fs or list(Xte.columns)!=fs:
            raise RuntimeError(f"{m} exact feature order not enforced")
        if not np.isfinite(Xtr.to_numpy(float)).all():
            raise RuntimeError(f"{m} historical design nonfinite")
        if not np.isfinite(Xte.to_numpy(float)).all():
            raise RuntimeError(f"{m} 2025 design nonfinite")
        designs[m]=(Xtr,Xte)
        audit.append({
            "model":m,"train_rows":len(Xtr),"test_rows":len(Xte),
            "n_design_columns":Xtr.shape[1],
            "feature_order_hash":hashlib.sha256(json.dumps(fs).encode()).hexdigest(),
            "feature_order":"|".join(fs),
        })
    return designs,pd.DataFrame(audit)

def fit_and_project(train,test,pop,designs):
    y=train[TARGET].to_numpy(float)
    e=train[EXPOSURE].to_numpy(float)
    rate=y/e
    params=[]
    fit_audit=[]
    predictions=test[["date","unit"]].copy()

    for m in MODELS:
        Xtr,Xte=designs[m]
        finite=(
            np.isfinite(Xtr.to_numpy(float)).all(axis=1)
            & np.isfinite(rate) & np.isfinite(e)
            & np.isfinite(y) & (y>=0) & (e>0)
        )
        if int(finite.sum())!=len(train):
            raise RuntimeError(f"{m} full-history fit would drop rows: {finite.sum()}/{len(train)}")

        model=PoissonRegressor(
            alpha=1e-5,max_iter=800,fit_intercept=True,
            tol=1e-10,solver="lbfgs"
        )
        model.fit(Xtr.loc[finite].to_numpy(float),rate[finite],sample_weight=e[finite])

        coef=np.asarray(model.coef_,float)
        intercept=float(model.intercept_)
        if len(coef)!=EXPECTED_COUNTS[m] or not np.isfinite(coef).all() or not np.isfinite(intercept):
            raise RuntimeError(f"{m} fitted parameter state invalid")
        if int(getattr(model,"n_iter_",800))>=800:
            raise RuntimeError(f"{m} optimizer reached max_iter; n_iter_={getattr(model,'n_iter_',None)}")

        pred_rate=np.asarray(model.predict(Xte.to_numpy(float)),float)
        pred_count=pred_rate*pop["population"].to_numpy(float)
        if not (np.isfinite(pred_rate).all() and np.isfinite(pred_count).all()
                and (pred_rate>0).all() and (pred_count>0).all()):
            raise RuntimeError(f"{m} 2025 prediction invalid")

        predictions[f"{m}_predicted_rate_repair"]=pred_rate
        predictions[f"{m}_predicted_count_repair"]=pred_count

        for i,(f,b) in enumerate(zip(Xtr.columns,coef)):
            params.append({
                "model":m,"feature_order":i,"design_column":f,
                "coefficient":float(b),"intercept":intercept
            })

        fit_audit.append({
            "model":m,
            "train_rows":int(finite.sum()),
            "design_columns":len(coef),
            "n_iter":int(getattr(model,"n_iter_", -1)),
            "intercept":intercept,
            "nonzero_slope_count_exact":int(np.count_nonzero(coef)),
            "max_abs_slope":float(np.max(np.abs(coef))),
            "l2_slope_norm":float(np.linalg.norm(coef)),
            "mean_predicted_rate_2025":float(np.mean(pred_rate)),
            "mean_predicted_count_2025":float(np.mean(pred_count)),
        })
    return pd.DataFrame(params),pd.DataFrame(fit_audit),predictions

def distinctness(pred):
    rows=[]
    for a,b in [("M1","M0"),("M2","M0"),("M2","M1")]:
        x=pred[f"{a}_predicted_count_repair"].to_numpy(float)
        y=pred[f"{b}_predicted_count_repair"].to_numpy(float)
        d=x-y
        rows.append({
            "pair":f"{a}-{b}",
            "exact_equal":bool(np.array_equal(x,y)),
            "allclose_1e_12":bool(np.allclose(x,y,rtol=0,atol=1e-12)),
            "max_abs_diff":float(np.max(np.abs(d))),
            "mean_abs_diff":float(np.mean(np.abs(d))),
            "nonzero_count_exact":int(np.count_nonzero(d)),
            "correlation":float(np.corrcoef(x,y)[0,1]),
        })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=DEFAULT_PROJECT)
    ap.add_argument("--outdir","--output-dir",dest="outdir",type=Path,default=None)
    a=ap.parse_args()
    project=a.project_root.resolve()
    out=(a.outdir or project/"stage34C3b7b15h2br_prospective_full_history_repair_fit_OUTPUT_v18c_r4_r10_r7").resolve()
    out.mkdir(parents=True,exist_ok=True)

    print("="*190)
    print("STAGE 34C.3b.7B15H2BR V18C_R4_R10_R7 — PROSPECTIVE FULL-HISTORY REPAIR FIT M0–M2")
    print(f"PACKAGE REVISION                         : {REV}")
    print("="*190)
    print("fit scope                                : full 2019–2024 only")
    print("estimator                                : frozen H2BA PoissonRegressor(alpha=1e-5,tol=1e-10,max_iter=800,lbfgs)")
    print("model/tolerance selection                : FORBIDDEN")
    print("2025 health outcome                      : NOT READ")
    print("objective                                : obtain one prospective non-degenerate M0–M2 parameter vector and freeze health-blind 2025 predictions")

    r6p,r6,contractp,orderp=require_contract(project)
    r9p,r9,h2w,g3=require_r9_inputs(project)
    r7p,r7,v14i,v14i_copies=require_v14i_design(project)
    feature_order,_=load_feature_order(orderp)

    train,science_cols=historical_training(h2w,g3,feature_order)
    test=load_2025_design(v14i,feature_order)
    pop,popp=load_2025_population(project,test)

    designs,design_audit=build_exact_design(train,test,feature_order)
    design_audit.to_csv(out/"FULL_HISTORY_AND_2025_EXACT_DESIGN_AUDIT.csv",index=False)

    params,fit_audit,pred=fit_and_project(train,test,pop,designs)
    distinct=distinctness(pred)

    params.to_csv(out/"FROZEN_REPAIR_FULL_HISTORY_2019_2024_M0_M2_PARAMETERS.csv",index=False)
    params.to_parquet(out/"FROZEN_REPAIR_FULL_HISTORY_2019_2024_M0_M2_PARAMETERS.parquet",index=False)
    fit_audit.to_csv(out/"FULL_HISTORY_REPAIR_FIT_CONVERGENCE_AUDIT.csv",index=False)
    pred.to_csv(out/"FROZEN_REPAIR_2025_M0_M2_PREDICTIONS_PRE_OUTCOME_SCORE.csv",index=False)
    pred.to_parquet(out/"FROZEN_REPAIR_2025_M0_M2_PREDICTIONS_PRE_OUTCOME_SCORE.parquet",index=False)
    distinct.to_csv(out/"REPAIR_2025_M0_M2_POINTWISE_DISTINCTNESS.csv",index=False)

    all_distinct=bool((distinct.nonzero_count_exact>0).all())
    all_nonzero=bool((fit_audit.nonzero_slope_count_exact>0).all())
    all_converged=bool((fit_audit.n_iter<800).all())

    print("historical rows/days/units".ljust(70)+f": {len(train)}/{train.date.nunique()}/{train.unit.nunique()}")
    print("2025 design rows/days/units".ljust(70)+f": {len(test)}/{test.date.nunique()}/{test.unit.nunique()}")
    print("frozen 2025 population rows".ljust(70)+f": {len(pop)}")
    print("\nFull-history repair fit convergence")
    print(fit_audit.to_string(index=False,float_format=lambda x:f"{x:.12g}"))
    print("\n2025 repaired prediction distinctness")
    print(distinct.to_string(index=False,float_format=lambda x:f"{x:.12g}"))
    print("\nall M0–M2 fits have nonzero slopes".ljust(70)+f": {all_nonzero}")
    print("all M0–M2 optimizers converged before max_iter".ljust(70)+f": {all_converged}")
    print("all 2025 M0/M1/M2 prediction pairs distinct".ljust(70)+f": {all_distinct}")

    if not (all_nonzero and all_converged and all_distinct):
        raise RuntimeError(
            f"repair fit validation failed nonzero={all_nonzero} converged={all_converged} distinct={all_distinct}"
        )

    # Freeze lineage/hashes.
    lineage={
        "package_revision":REV,
        "R10_R6_contract_summary":str(r6p),
        "R10_R6_contract":str(contractp),
        "R10_R6_contract_sha256":sha256_file(contractp),
        "R10_R6_feature_order":str(orderp),
        "R10_R6_feature_order_sha256":sha256_file(orderp),
        "R9_historical_H2W":str(h2w),"R9_historical_H2W_sha256":sha256_file(h2w),
        "R9_historical_G3":str(g3),"R9_historical_G3_sha256":sha256_file(g3),
        "R7_V14I_design_selected_physical_copy":str(v14i),
        "R7_V14I_design_sha256":sha256_file(v14i),
        "R7_V14I_equivalent_physical_copies":[str(p) for p in v14i_copies],
        "V15E_frozen_2025_population":str(popp),
        "V15E_frozen_2025_population_sha256":sha256_file(popp),
        "2025_health_read":False,
        "outcome_scoring_performed":False,
    }
    (out/"REPAIR_FIT_INPUT_LINEAGE_SHA256.json").write_text(
        json.dumps(lineage,indent=2,ensure_ascii=False)
    )

    summary={
        "status":PASS,
        "package_revision":REV,
        "historical_grid":{"rows":len(train),"days":train.date.nunique(),"units":train.unit.nunique()},
        "projection_grid":{"rows":len(test),"days":test.date.nunique(),"units":test.unit.nunique()},
        "fit_audit":fit_audit.to_dict("records"),
        "prediction_distinctness":distinct.to_dict("records"),
        "all_nonzero_slopes":all_nonzero,
        "all_converged_before_max_iter":all_converged,
        "all_prediction_pairs_distinct":all_distinct,
        "fit_recipe":{
            "class":"PoissonRegressor","alpha":1e-5,"tol":1e-10,
            "max_iter":800,"fit_intercept":True,"solver":"lbfgs",
            "target":"urgent_respiratory_total / population",
            "sample_weight":"population",
            "prediction_count":"model.predict(X_2025) * frozen_2025_population"
        },
        "model_selection":False,"tolerance_selection":False,
        "historical_science_reopened":False,
        "2025_health_read":False,"2025_outcome_scored":False,
        "M3":"DEFERRED_GRD_2025_NOT_YET_PUBLISHED",
        "scientific_interpretation_authorized":False,
        "next_stage":"V18C_R5_EXTERNAL_2025_RESCORING_WITH_REPAIRED_FULL_HISTORY_M0_M2",
    }
    (out/"B7B15H2BR_V18C_R4_R10_R7_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)
    )

    print("\n"+PASS)

if __name__=="__main__":
    main()
