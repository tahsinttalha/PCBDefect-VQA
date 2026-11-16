# MuQA-VQA

**MuQA-VQA: A Multi-Stage, Multi-Domain Visual Question Answering Benchmark for Printed Circuit Board Defect Inspection**

`MuQA-VQA` harmonises four complementary optical inspection datasets across three stages of PCB manufacturing into a unified, leak-free VQA benchmark:
1. **Stage 1 — Bare Copper Boards:** `DeepPCB` (monochrome optical scans) & `PKU-Market-PCB` (RGB captures)
2. **Stage 2 — Populated Assembled PCBs:** `Amazon-VisA` (`pcb1`–`pcb4`)
3. **Stage 3 — Macro Solder Joints:** `SolDef_AI`

## Benchmark Overview
- **Total Images:** 5,864 (`3,945` untouched real factory captures + `1,919` coordinate-aware `Train`-only optical augmentations)
- **Strict Split Integrity:** `4,682` Train / `591` Val (**100% real**) / `591` Test (**100% real**) — zero synthetic leakage
- **Total VQA Pairs:** 23,456 across 4 cognitive tiers (Screening, Multi-Label Classification, Spatial Localisation, Diagnostic Reasoning)
- **Canonical Categories:** 21 (`clean` + 20 physical defect classes, `1.39x` max-to-min defect imbalance ratio)

## Repository Structure
- `data/muqa_vqa_release.json`: Complete 5,864-image / 23,456-QA benchmark with portable relative paths and provenance flags.
- `data/muqa_vqa_summary_table.csv`: Exact raw, real, and augmented split counts across all 21 categories.
- `data/muqa_vqa_datasheet.json`: Machine-readable datasheet and split provenance (`random_seed = 42`).
- `data/plot_csvs/`: Cached summary CSVs for rapid figure regeneration.
- `audit/muqa_vqa_human_audit_100.csv`: 100-sample double-blind human quality verification template.
- `scripts/verify_local_dataset.py`: Cross-platform path verification utility.
- `figures/`: High-resolution (`300 DPI`) publication figures.
- `notebooks/`: Clean, reproducible dataset construction pipeline.
