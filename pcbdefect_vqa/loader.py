import os
import json
import zipfile
import subprocess
from PIL import Image
from huggingface_hub import hf_hub_download
import kagglehub

REPO_ID = "tahsinttalha/PCBDefect-VQA-v3.0"
DEFAULT_REVISION = "d5f6a6e9613c1fa1a17a14825fb36c2fe593b6da"

KAGGLE_SLUGS = {
    "Amazon-VisA": "tensura3607/amazon-visa-anomaly",
    "PKU-Market-PCB": "norbertelter/pcb-defect-dataset",
    "SolDef_AI": "mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection",
}
DEEPPCB_GIT = "https://github.com/tangsanli5201/DeepPCB.git"


def _find_dir_containing(root, marker_subpath):
    for dp, _, _ in os.walk(root):
        if os.path.exists(os.path.join(dp, marker_subpath)):
            return dp
    return None


def _index_images(root):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[fn.lower()] = os.path.join(dp, fn)
    return idx


def load_benchmark(
    work_dir="./pcbdefect_v3_cache",
    revision=DEFAULT_REVISION,
    hf_token=None,
    verify=True,
):
    os.makedirs(work_dir, exist_ok=True)
    token = hf_token or os.environ.get("HF_TOKEN")

    release_fp = hf_hub_download(
        REPO_ID,
        "pcbdefect_vqa_v3_release.json",
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=work_dir,
    )
    datasheet_fp = hf_hub_download(
        REPO_ID,
        "pcbdefect_vqa_v3_datasheet.json",
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=work_dir,
    )
    aug_zip_fp = hf_hub_download(
        REPO_ID,
        "augmented_train_v3.zip",
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=work_dir,
    )

    aug_root = os.path.join(work_dir, "augmented_train_v3")
    if not os.path.exists(aug_root):
        with zipfile.ZipFile(aug_zip_fp, "r") as zf:
            zf.extractall(work_dir)

    deeppcb_dir = os.path.join(work_dir, "DeepPCB")
    if not os.path.exists(os.path.join(deeppcb_dir, "PCBData")):
        subprocess.run(
            ["git", "clone", "--depth", "1", DEEPPCB_GIT, deeppcb_dir],
            check=True,
        )

    visa_dl = kagglehub.dataset_download(KAGGLE_SLUGS["Amazon-VisA"])
    pku_dl = kagglehub.dataset_download(KAGGLE_SLUGS["PKU-Market-PCB"])
    soldef_dl = kagglehub.dataset_download(KAGGLE_SLUGS["SolDef_AI"])

    pku_root = _find_dir_containing(pku_dl, os.path.join("train", "images")) or pku_dl
    visa_root = (
        _find_dir_containing(visa_dl, os.path.join("pcb1", "image_anno.csv"))
        or _find_dir_containing(visa_dl, "pcb1")
        or visa_dl
    )
    soldef_index = _index_images(soldef_dl)
    pku_fallback = _index_images(pku_dl)

    with open(release_fp, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(datasheet_fp, "r", encoding="utf-8") as f:
        datasheet = json.load(f)

    def resolve_image_path(rec):
        p = rec["portable_path"]
        if p.startswith("augmented_train_v3/"):
            return os.path.join(aug_root, os.path.basename(p))
        if p.startswith("DeepPCB/"):
            return os.path.join(work_dir, p)
        if p.startswith("PKU-Market-PCB/"):
            cand = os.path.join(pku_root, p[len("PKU-Market-PCB/"):])
            if os.path.exists(cand):
                return cand
            return pku_fallback.get(os.path.basename(p).lower(), cand)
        if p.startswith("Amazon-VisA/"):
            return os.path.join(visa_root, p[len("Amazon-VisA/"):])
        if p.startswith("SolDef_AI/"):
            hit = soldef_index.get(os.path.basename(p).lower())
            if hit:
                return hit
        raise FileNotFoundError(f"Cannot resolve image path: {p}")

    def load_clean_image(rec):
        with Image.open(resolve_image_path(rec)) as im:
            img = im.convert("RGB").copy()
        img.info.clear()
        return img

    if verify:
        missing = [
            r["portable_path"]
            for r in records
            if not os.path.exists(resolve_image_path(r))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)}/{len(records)} images. Sample: {missing[:3]}"
            )

    splits = {
        "train_real": [
            r for r in records
            if r["split"] == "train" and not r["is_augmented"] and not r["eval_exclude"]
        ],
        "train_full": [
            r for r in records
            if r["split"] == "train" and not r["eval_exclude"]
        ],
        "val": [
            r for r in records
            if r["split"] == "val" and not r["eval_exclude"]
        ],
        "test": [
            r for r in records
            if r["split"] == "test" and not r["eval_exclude"]
        ],
    }

    print(
        f"PCBDefect-VQA-v3.0 ready | Commit: {revision} | "
        f"Records: {len(records):,} | Eval Test: {len(splits['test'])}"
    )
    return {
        "records": records,
        "splits": splits,
        "datasheet": datasheet,
        "commit_sha": revision,
        "work_dir": work_dir,
        "resolve_image_path": resolve_image_path,
        "load_clean_image": load_clean_image,
    }