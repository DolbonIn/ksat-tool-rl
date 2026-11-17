#!/usr/bin/env python3
"""Offline strict gold-tier TEXT exporter. Not a crawler/generator/trainer.
Requires jsonschema. Approval references are checked for presence, not authenticity.
Semantic/visual deduplication and actual mathematical verification remain external.
"""
from __future__ import annotations
import argparse, hashlib, json, unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[1]
def load(path: Path) -> list[dict[str, Any]]:
    rows=[]
    for i,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: r=json.loads(line)
        except json.JSONDecodeError as e: raise ValueError(f"{path}:{i}: {e}") from e
        if not isinstance(r,dict): raise ValueError(f"{path}:{i}: object required")
        rows.append(r)
    return rows
def save(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
def approved(c: dict[str, Any]) -> bool:
    return c.get("status")=="pass" and bool(c.get("report_ref"))
def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC",s).split())
def hashval(x: Any) -> str:
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def same_answer(a: Any,b: Any,kind: str) -> bool:
    if a is None or b is None: return False
    if kind=="choice": return str(a).strip()==str(b).strip()
    if kind=="numeric":
        try: return Fraction(str(a).strip())==Fraction(str(b).strip())
        except (ValueError,ZeroDivisionError,OverflowError): return False
    return False
def effective_split(p: dict[str, Any]) -> str:
    return p["split"] if p["split"]!="quarantine" else p["intended_split"]
def integrity(problems: list[dict[str, Any]],traces: list[dict[str, Any]]) -> None:
    errors=[]
    for rows,key,schema in [(problems,"problem_id","problem"),(traces,"trace_id","trace")]:
        v=Draft202012Validator(json.loads((ROOT/"schemas"/(schema+".schema.json")).read_text()))
        ids=[]
        for row in rows:
            errors.extend(f"{schema}: {e.message}" for e in v.iter_errors(row))
            ids.append(row.get(key))
        if len(ids)!=len(set(ids)): errors.append(f"duplicate {key}")
    if errors: raise ValueError("\n".join(errors[:30]))
    index={p["problem_id"]:p for p in problems}; groups=defaultdict(set)
    for p in problems:
        split=effective_split(p)
        groups["family:"+p["family_id"]].add(split)
        for passage in p["input"]["passages"]:
            groups["passage_id:"+passage["id"]].add(split)
            if passage["text"].strip(): groups["passage_text:"+hashval(norm(passage["text"]))].add(split)
        if p["input"]["question"] and p["input"]["modality"]=="text":
            fingerprint=[norm(p["input"]["question"]),[norm(s["text"]) for s in p["input"]["passages"]],
                         sorted(norm(c["text"]) for c in p["input"]["choices"])]
            groups["input:"+hashval(fingerprint)].add(split)
        for parent in p.get("parent_problem_ids",[]):
            if parent not in index: errors.append(f"unresolved parent {parent}")
            elif effective_split(index[parent])!=split: errors.append(f"parent split mismatch {parent}")
    for k,splits in groups.items():
        if len(splits-{"quarantine"})>1: errors.append(f"cross-split leakage: {k}: {sorted(splits)}")
    for t in traces:
        if t["problem_id"] not in index: errors.append(f"orphan trace {t['trace_id']}")
    if errors: raise ValueError("\n".join(errors[:30]))
def problem_blocks(p: dict[str, Any]) -> list[str]:
    b=[]
    if p["demo_only"]: b.append("demo_only")
    if p["split"]!="train" or p["intended_split"]!="train": b.append("not_train")
    year=p["source"].get("administration_year")
    if year is not None and year not in [2021,2022,2023]: b.append("temporal_holdout")
    if not p["input"]["question"] or not p["input"]["question"].strip(): b.append("missing_question")
    if p["input"]["modality"]!="text" or p["input"]["assets"]: b.append("multimodal_adapter_required")
    for k in ["source","parse","grouping","gold","human_review"]:
        if not approved(p["checks"][k]): b.append("problem_"+k+"_not_approved")
    if p["rights"]["status"]!="approved" or not p["rights"]["evidence_ref"]: b.append("rights_not_approved")
    if not p["gold"]["source_ref"]: b.append("missing_gold_provenance")
    kind=p["gold"]["answer_type"]
    if kind=="choice":
        ids=[c["id"] for c in p["input"]["choices"]]
        if len(ids)<2 or len(set(ids))!=len(ids) or p["gold"]["choice_id"] not in ids: b.append("invalid_choices")
    elif kind=="numeric":
        if not same_answer(p["gold"]["value"],p["gold"]["value"],kind): b.append("symbolic_checker_required")
    else: b.append("unsupported_answer_type")
    return b
def trace_blocks(p: dict[str, Any],t: dict[str, Any]) -> list[str]:
    b=[]
    if t["task"]!="solve": b.append("non_solve_adapter_required")
    if t["generation"]["gold_visible"] is not False: b.append("not_confirmed_blind")
    if t["generation"]["finish_reason"] not in ["stop","eos"]: b.append("incomplete_or_unknown_finish")
    if t["generation"]["mode"]!="no_tool": b.append("not_no_tool")
    if not t["solution"]["rationale"].strip() or not t["solution"]["final"].strip(): b.append("empty_solution")
    for k in ["answer","rationale","format","human_review"]:
        if not approved(t["checks"][k]): b.append("trace_"+k+"_not_approved")
    kind=p["gold"]["answer_type"]
    expected=p["gold"]["choice_id"] if kind=="choice" else p["gold"]["value"]
    actual=t["solution"]["choice_id"] if kind=="choice" else t["solution"]["answer_value"]
    if not same_answer(expected,actual,kind): b.append("answer_mismatch_or_unsupported")
    return b
def prompt(p: dict[str, Any]) -> list[dict[str, Any]]:
    # Gold, source metadata, teacher labels and legacy explanations are never serialized.
    parts=[f"지문 [{s['id']}]\n{s['text']}" for s in p["input"]["passages"]]
    parts.append("문제\n"+p["input"]["question"])
    if p["input"]["choices"]: parts.append("선지\n"+"\n".join(f"{c['id']}. {c['text']}" for c in p["input"]["choices"]))
    return [{"role":"user","content":"\n\n".join(parts)}]
def completion(t: dict[str, Any],style: str) -> list[dict[str, Any]]:
    r=t["solution"]["rationale"];f=t["solution"]["final"]
    if style=="native_fields": return [{"role":"assistant","reasoning_content":r,"content":f}]
    text="<think>\n"+r+"\n</think>\n"+f if style=="think_tags" else r+"\n\n"+f
    return [{"role":"assistant","content":text}]
def audit(problems: list[dict[str, Any]],traces: list[dict[str, Any]],style: str="plain") -> tuple:
    integrity(problems,traces)
    index={p["problem_id"]:p for p in problems}; blocks={k:problem_blocks(p) for k,p in index.items()}
    sft=[];rl=[];rejected=[]
    for p in problems:
        if blocks[p["problem_id"]]: continue
        kind=p["gold"]["answer_type"]
        answer=p["gold"]["choice_id"] if kind=="choice" else str(p["gold"]["value"])
        rl.append({"prompt":prompt(p),"problem_id":p["problem_id"],"answer":answer,"answer_type":kind,
                   "checker":"exact_choice_v1" if kind=="choice" else "exact_rational_v1"})
    for t in traces:
        p=index[t["problem_id"]]; reasons=blocks[t["problem_id"]]+trace_blocks(p,t)
        if reasons: rejected.append({"trace_id":t["trace_id"],"reasons":reasons})
        else: sft.append({"prompt":prompt(p),"completion":completion(t,style),"problem_id":p["problem_id"],"trace_id":t["trace_id"]})
    report={"problems":len(problems),"traces":len(traces),"accepted_sft":len(sft),"accepted_rl_prompts":len(rl),
            "blocked_traces":len(rejected),"block_reason_counts":dict(Counter(x for r in rejected for x in set(r["reasons"]))),
            "blocked":rejected,"template_roundtrip_tested":False,"loss_mask_tested":False,
            "scope":"strict human-reviewed text-only solve exporter; no generation/training",
            "not_implemented":["semantic/visual deduplication","evidence validity checking","human/rights decisions",
                               "multimodal export","verify/repair/preference export","tokenization/training"]}
    return report,sft,rl
def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems",type=Path,required=True);ap.add_argument("--traces",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--style",choices=["plain","think_tags","native_fields"],default="plain")
    a=ap.parse_args()
    try: report,sft,rl=audit(load(a.problems),load(a.traces),a.style)
    except (ValueError,OSError) as e: ap.exit(2,str(e)+"\n")
    a.out.mkdir(parents=True,exist_ok=True);save(a.out/"sft.jsonl",sft);save(a.out/"rl.jsonl",rl)
    (a.out/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["problems","traces","accepted_sft","accepted_rl_prompts","blocked_traces"]},ensure_ascii=False))
if __name__=="__main__": main()
