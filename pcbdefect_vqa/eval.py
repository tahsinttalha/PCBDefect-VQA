import re
from collections import Counter
import numpy as np
import pandas as pd

CANONICAL_CLASSES = [
    "clean", "bent pin", "burnt component", "damaged component",
    "defective solder joint", "excessive solder", "extra component",
    "insufficient solder", "melted solder", "misaligned component",
    "missing component", "missing hole", "mousebite", "open",
    "pin-hole", "short", "solder spike", "spur", "spurious copper",
    "surface contamination", "surface scratch"
]
CANONICAL_SET = set(CANONICAL_CLASSES)

SYNONYM_MAP = {
    "no defect": "clean", "no defects": "clean", "normal": "clean", "defect-free": "clean", "good": "clean", "none": "clean",
    "open circuit": "open", "broken trace": "open", "trace break": "open", "disconnected trace": "open", "cut trace": "open",
    "short circuit": "short", "solder bridge": "short", "bridging": "short", "trace bridge": "short",
    "mouse bite": "mousebite", "mouse-bite": "mousebite", "edge notch": "mousebite", "nibble": "mousebite",
    "pinhole": "pin-hole", "pin hole": "pin-hole", "void": "pin-hole", "solder void": "pin-hole",
    "copper spur": "spur", "protrusion": "spur", "copper protrusion": "spur",
    "spurious": "spurious copper", "extra copper": "spurious copper", "residual copper": "spurious copper", "copper residue": "spurious copper",
    "missing drill": "missing hole", "un-drilled hole": "missing hole", "undrilled hole": "missing hole", "absent hole": "missing hole",
    "scratch": "surface scratch", "scratches": "surface scratch", "trace scratch": "surface scratch",
    "contamination": "surface contamination", "dirt": "surface contamination", "stain": "surface contamination", "foreign material": "surface contamination", "flux residue": "surface contamination",
    "missing": "missing component", "absent component": "missing component", "missing ic": "missing component", "missing resistor": "missing component", "missing capacitor": "missing component",
    "extra": "extra component", "additional component": "extra component", "foreign component": "extra component",
    "misaligned": "misaligned component", "shifted component": "misaligned component", "offset component": "misaligned component", "skewed component": "misaligned component", "tombstone": "misaligned component",
    "damaged": "damaged component", "cracked component": "damaged component", "broken component": "damaged component", "chipped component": "damaged component",
    "burnt": "burnt component", "burned component": "burnt component", "charred component": "burnt component", "scorched component": "burnt component",
    "bent": "bent pin", "bent lead": "bent pin", "deformed pin": "bent pin", "damaged pin": "bent pin",
    "defective solder": "defective solder joint", "bad solder": "defective solder joint", "poor solder joint": "defective solder joint", "cold solder": "defective solder joint", "cold joint": "defective solder joint",
    "excess solder": "excessive solder", "over-soldered": "excessive solder", "too much solder": "excessive solder",
    "insufficient": "insufficient solder", "starved solder": "insufficient solder", "solder starvation": "insufficient solder", "under-soldered": "insufficient solder",
    "melted": "melted solder", "reflow defect": "melted solder", "solder splash": "melted solder", "solder ball": "melted solder"
}

VALID_REGIONS = {
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right"
}

REGION_SYNONYMS = {
    "centre": "center", "middle-center": "center", "middle center": "center", "middle-centre": "center",
    "centre-left": "middle-left", "centre left": "middle-left", "center-left": "middle-left", "center left": "middle-left",
    "centre-right": "middle-right", "centre right": "middle-right", "center-right": "middle-right", "center right": "middle-right",
    "top-middle": "top-center", "top middle": "top-center", "top center": "top-center", "top-centre": "top-center",
    "bottom-middle": "bottom-center", "bottom middle": "bottom-center", "bottom center": "bottom-center", "bottom-centre": "bottom-center",
    "top left": "top-left", "top right": "top-right",
    "bottom left": "bottom-left", "bottom right": "bottom-right",
    "middle left": "middle-left", "middle right": "middle-right"
}


def normalize_label(raw_tok):
    t = re.sub(r"\s+", " ", str(raw_tok).strip().lower().strip(".,;:!?\"'()[]"))
    if t in CANONICAL_SET:
        return t, True
    if t in SYNONYM_MAP:
        return SYNONYM_MAP[t], True
    return t, False


def normalize_region(raw_reg):
    r = re.sub(r"\s+", " ", str(raw_reg).strip().lower().strip(".,;:!?\"'()[]"))
    if r in REGION_SYNONYMS:
        return REGION_SYNONYMS[r], True
    r_dash = r.replace(" ", "-")
    if r_dash in REGION_SYNONYMS:
        return REGION_SYNONYMS[r_dash], True
    if r_dash in VALID_REGIONS:
        return r_dash, True
    return r_dash, False


def parse_prediction(tier, raw_text):
    txt = str(raw_text).strip()
    if not txt or txt.lower() == "nan":
        return None, False

    if tier == 1:
        first = re.split(r"[\s.,;:!?]+", txt.lower())[0]
        if first in ("yes", "defective", "defect", "true"):
            return "Yes", True
        if first in ("no", "clean", "normal", "false"):
            return "No", True
        return None, False

    if tier == 2:
        parts = [p.strip() for p in re.split(r"[,;\n]+", txt) if p.strip()]
        mapped, all_ok = set(), True
        for p in parts:
            c, ok = normalize_label(p)
            if ok:
                mapped.add(c)
            else:
                all_ok = False
        if not mapped:
            return None, False
        if "clean" in mapped and len(mapped) > 1:
            mapped.discard("clean")
        return sorted(mapped), all_ok

    if tier == 3:
        low = txt.lower().strip(" .")
        if low in ("none", "clean", "no defects", "no defect", "n/a", "no defect detected", "no defects detected"):
            return [], True
        items, all_ok = set(), True
        for chunk in re.split(r"[;\n]+|,(?![^()]*\))", txt):
            chunk = chunk.strip()
            if not chunk:
                continue
            reg_only, ok_reg_only = normalize_region(chunk)
            if ok_reg_only:
                items.add(reg_only)
                continue
            m = re.match(r"^(.+?)\s*(?::|\(|(?:\bat|\bin)\s+)\s*([^()]+?)\)?$", chunk, flags=re.IGNORECASE)
            if m:
                c, ok_c = normalize_label(m.group(1))
                reg, ok_r = normalize_region(m.group(2))
                if ok_c and c != "clean" and ok_r:
                    items.add(f"{c}:{reg}")
                    continue
            all_ok = False
        return sorted(items), (all_ok and len(items) > 0)

    if tier == 4:
        low = txt.lower().strip(" .")
        if low.startswith(("clean", "no defect", "the board is clean", "this board is clean")):
            return ["clean"], True
        found_classes, all_ok = set(), True
        segments = re.split(r"[\n;]+|(?<=\.)\s+(?=[A-Za-z][A-Za-z0-9 \-]+:)", txt)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            lbl_part = seg.split(":", 1)[0] if ":" in seg else seg
            c, ok = normalize_label(lbl_part)
            if ok:
                found_classes.add(c)
            else:
                all_ok = False
        return sorted(found_classes), (len(found_classes) > 0 and all_ok)

    raise ValueError(f"Unknown tier: {tier}")


def multilabel_f1(gt_sets, pr_sets, label_list=None):
    tp = fp = fn = 0
    for g, p in zip(gt_sets, pr_sets):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    micro = 2.0 * tp / max(1, 2 * tp + fp + fn)
    if label_list is None:
        return micro, None
    per_c = []
    for c in label_list:
        tpc = sum(1 for g, p in zip(gt_sets, pr_sets) if c in g and c in p)
        fpc = sum(1 for g, p in zip(gt_sets, pr_sets) if c not in g and c in p)
        fnc = sum(1 for g, p in zip(gt_sets, pr_sets) if c in g and c not in p)
        denom = 2 * tpc + fpc + fnc
        per_c.append(2.0 * tpc / denom if denom > 0 else 0.0)
    return micro, float(np.mean(per_c))


def clustered_bootstrap_ci(records_sub, gt_sets, pr_sets, label_list=None, metric="macro", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.array([r["split_group_id"] for r in records_sub])
    uniq_g = np.unique(groups)
    g_idx = [np.where(groups == g)[0] for g in uniq_g]
    n_g = len(uniq_g)
    vals = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([g_idx[i] for i in draw])
        g_s = [gt_sets[i] for i in idx]
        p_s = [pr_sets[i] for i in idx]
        mi, ma = multilabel_f1(g_s, p_s, label_list if metric == "macro" else None)
        vals.append(ma if metric == "macro" else mi)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def compute_blind_floors(train_real_records, eval_records):
    maj_t1 = {
        s: Counter(r["qa_pairs"][0]["answer"] for r in train_real_records if r["source_dataset"] == s).most_common(1)[0][0]
        for s in {"Amazon-VisA", "SolDef_AI"}
    }
    maj_t2 = {
        s: Counter(r["qa_pairs"][1]["answer"] for r in train_real_records if r["source_dataset"] == s).most_common(1)[0][0]
        for s in set(r["source_dataset"] for r in train_real_records)
    }
    maj_t3 = {
        s: Counter(r["qa_pairs"][2]["answer"] for r in train_real_records if r["source_dataset"] == s).most_common(1)[0][0]
        for s in set(r["source_dataset"] for r in train_real_records)
    }

    subsets = [
        ("All Eval Test", lambda s: True, False),
        ("Without VisA", lambda s: s != "Amazon-VisA", False),
        ("Amazon-VisA [T1 Primary]", lambda s: s == "Amazon-VisA", True),
        ("SolDef_AI [T1 Caveat]", lambda s: s == "SolDef_AI", True),
        ("PKU-Market-PCB", lambda s: s == "PKU-Market-PCB", False),
        ("DeepPCB Non-Temp", lambda s: s == "DeepPCB", False),
    ]

    rows = []
    for subset_name, src_filter, t1_valid in subsets:
        sub_te = [r for r in eval_records if src_filter(r["source_dataset"])]
        t1_acc = (
            round(float(np.mean([maj_t1[r["source_dataset"]] == r["qa_pairs"][0]["answer"] for r in sub_te])), 4)
            if t1_valid and sub_te else None
        )
        tp2 = fp2 = fn2 = tp3 = fp3 = fn3 = 0
        for r in sub_te:
            g2 = set(parse_prediction(2, r["qa_pairs"][1]["answer"])[0] or [])
            p2 = set(parse_prediction(2, maj_t2[r["source_dataset"]])[0] or [])
            tp2 += len(g2 & p2); fp2 += len(p2 - g2); fn2 += len(g2 - p2)
            g3 = set(parse_prediction(3, r["qa_pairs"][2]["answer"])[0] or [])
            p3 = set(parse_prediction(3, maj_t3[r["source_dataset"]])[0] or [])
            tp3 += len(g3 & p3); fp3 += len(p3 - g3); fn3 += len(g3 - p3)
        rows.append({
            "Subset": subset_name,
            "N": len(sub_te),
            "T1_BlindFloor_Acc": t1_acc,
            "T2_BlindFloor_MicroF1": round(2 * tp2 / max(1, 2 * tp2 + fp2 + fn2), 4),
            "T3_BlindFloor_MicroF1": round(2 * tp3 / max(1, 2 * tp3 + fp3 + fn3), 4),
        })
    return pd.DataFrame(rows)


def evaluate_predictions(train_real_records, eval_records, preds_by_path, model_name="model", run_bootstrap=True):
    tr_cls_counts = Counter(r["primary_class"] for r in train_real_records)
    rare_8 = sorted([c for c in CANONICAL_CLASSES if tr_cls_counts.get(c, 0) < 40])
    common_13 = sorted([c for c in CANONICAL_CLASSES if c not in set(rare_8)])

    subsets = [
        ("All Eval Test", lambda s: True, False),
        ("Without VisA", lambda s: s != "Amazon-VisA", False),
        ("Amazon-VisA [T1 Primary]", lambda s: s == "Amazon-VisA", True),
        ("SolDef_AI [T1 Caveat]", lambda s: s == "SolDef_AI", True),
        ("PKU-Market-PCB", lambda s: s == "PKU-Market-PCB", False),
        ("DeepPCB Non-Temp", lambda s: s == "DeepPCB", False),
    ]

    rows = []
    for sub_name, fn, t1_valid in subsets:
        sub = [r for r in eval_records if fn(r["source_dataset"])]
        if not sub:
            continue

        t1_gt, t1_pr, t1_ok = [], [], 0
        t2_gt, t2_pr, t2_ok = [], [], 0
        t3_gt, t3_pr, t3_ok = [], [], 0
        t4_gt, t4_pr, t4_ok = [], [], 0

        for r in sub:
            p_dict = preds_by_path[r["portable_path"]]

            g1, _ = parse_prediction(1, r["qa_pairs"][0]["answer"])
            p1, ok1 = parse_prediction(1, p_dict.get("tier1", ""))
            t1_gt.append(g1); t1_pr.append(p1); t1_ok += int(ok1)

            g2, _ = parse_prediction(2, r["qa_pairs"][1]["answer"])
            p2, ok2 = parse_prediction(2, p_dict.get("tier2", ""))
            t2_gt.append(set(g2 or [])); t2_pr.append(set(p2 or [])); t2_ok += int(ok2)

            g3, _ = parse_prediction(3, r["qa_pairs"][2]["answer"])
            p3, ok3 = parse_prediction(3, p_dict.get("tier3", ""))
            t3_gt.append(set(g3 or [])); t3_pr.append(set(p3 or [])); t3_ok += int(ok3)

            g4, _ = parse_prediction(4, r["qa_pairs"][3]["answer"])
            p4, ok4 = parse_prediction(4, p_dict.get("tier4", ""))
            t4_gt.append(set(g4 or [])); t4_pr.append(set(p4 or [])); t4_ok += int(ok4)

        n = len(sub)
        t1_acc = round(float(np.mean([g == p for g, p in zip(t1_gt, t1_pr)])), 4) if t1_valid else None
        present_cls = sorted(set().union(*t2_gt))
        t2_micro, t2_macro = multilabel_f1(t2_gt, t2_pr, present_cls)
        _, t2_rare8 = multilabel_f1(t2_gt, t2_pr, rare_8) if sub_name == "All Eval Test" else (None, None)
        _, t2_comm13 = multilabel_f1(t2_gt, t2_pr, common_13) if sub_name == "All Eval Test" else (None, None)
        t3_micro, _ = multilabel_f1(t3_gt, t3_pr)
        t4_micro, t4_macro = multilabel_f1(t4_gt, t4_pr, present_cls)
        ci_t2 = (
            clustered_bootstrap_ci(sub, t2_gt, t2_pr, present_cls, "macro", n_boot=2000)
            if (run_bootstrap and sub_name in ("All Eval Test", "Without VisA"))
            else None
        )

        rows.append({
            "Model": model_name,
            "Subset": sub_name,
            "N": n,
            "T1_Acc": t1_acc,
            "T2_MicroF1": round(t2_micro, 4),
            "T2_MacroF1": round(t2_macro, 4),
            "T2_Rare8_MF1": round(t2_rare8, 4) if t2_rare8 is not None else None,
            "T2_Comm13_MF1": round(t2_comm13, 4) if t2_comm13 is not None else None,
            "T2_MacroF1_CI95": ci_t2,
            "T3_MicroF1": round(t3_micro, 4),
            "T4_ClassImpact_MF1": round(t4_macro, 4),
            "ParseRate_T1_T4": f"{t1_ok/n:.2f}/{t2_ok/n:.2f}/{t3_ok/n:.2f}/{t4_ok/n:.2f}",
        })

    return pd.DataFrame(rows)