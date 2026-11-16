import os
import json
from collections import Counter

DATASET_ROOT = "."
JSON_FILE = os.path.join(DATASET_ROOT, "data", "muqa_vqa_release.json")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

found = 0
missing = Counter()
for rec in records:
    full_path = os.path.join(DATASET_ROOT, rec["portable_path"].replace("/", os.sep))
    if os.path.exists(full_path):
        found += 1
    else:
        missing[rec["source_dataset"]] += 1

print(f"Verified {found} / {len(records)} images on disk.")
if missing:
    print("Missing by source:", dict(missing))
