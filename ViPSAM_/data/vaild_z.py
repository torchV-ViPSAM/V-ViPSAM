import os
import csv
import argparse
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ViPSAM_.data.prepare_data import load_volume, nonempty_slices


def compute_and_save(csv_path: str, resampled_root: str, out_csv: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        cid = row.get("case_id", "").strip()
        if not cid:
            continue

        label_path = row.get("label_out", "").strip()
        if not label_path:
            label_path = os.path.join(resampled_root, cid, f"{cid}_label_1mm.nii.gz")
        label_path = label_path.replace(".nii.gz", ".npy") if os.path.exists(label_path.replace(".nii.gz", ".npy")) else label_path

        if not os.path.isfile(label_path):
            print(f"  [SKIP] {cid}: label not found")
            continue

        mk_vol  = load_volume(label_path)
        binary  = ((mk_vol == 1) | (mk_vol == 2)).astype(np.float32)
        valid_z = np.sort(nonempty_slices(binary))
        del mk_vol, binary

        if valid_z.size == 0:
            print(f"  [SKIP] {cid}: no valid slices")
            continue

        results.append({
            "case_id": cid,
            "valid_z": ",".join(str(z) for z in valid_z.tolist()),
        })
        print(f"  [{cid}] valid_z: {len(valid_z)} slices ({valid_z[0]}~{valid_z[-1]})")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "valid_z"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSave done: {out_csv}  ({len(results)} cases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv",   type=str, required=True)
    parser.add_argument("--test_csv",  type=str, required=True)
    parser.add_argument("--out_dir",   type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split, path in [("train", args.train_csv),
                        ("val",   args.val_csv),
                        ("test",  args.test_csv)]:
        print(f"\n=== {split} ===")
        compute_and_save(
            csv_path=path,
            resampled_root=args.root,
            out_csv=os.path.join(args.out_dir, f"{split}_valid_z.csv"),
        )


# python ViPSAM_/data/vaild_z.py \
#   --root     /data2/2025_san/data/HECKTOR2025_Task1_resampled_1mm \
#   --train_csv /data2/2025_san/data/HECKTOR2025_splits_random/train.csv \
#   --val_csv   /data2/2025_san/data/HECKTOR2025_splits_random/val.csv \
#   --test_csv  /data2/2025_san/data/HECKTOR2025_splits_random/test.csv \
#   --out_dir   /data2/2025_san/data/HECKTOR2025_splits_random/valid_z