import os
import csv
import argparse
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ViPSAM_.data.prepare_data import load_volume


def process(valid_z_csv, split_csv, resampled_root, out_dir):
    valid_z_map = {}
    with open(valid_z_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["case_id"].strip()
            valid_z_map[cid] = np.array(
                [int(z) for z in row["valid_z"].split(",")], dtype=np.int64)

    with open(split_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    os.makedirs(out_dir, exist_ok=True)

    for row in rows:
        cid = row.get("case_id", "").strip()
        if not cid or cid not in valid_z_map:
            continue

        ct_path    = row.get("ct_out",    "").strip() or \
                     os.path.join(resampled_root, cid, f"{cid}__CT_1mm.nii.gz")
        pt_path    = row.get("pt_out",    "").strip() or \
                     os.path.join(resampled_root, cid, f"{cid}__PT_1mm.nii.gz")
        label_path = row.get("label_out", "").strip() or \
                     os.path.join(resampled_root, cid, f"{cid}_label_1mm.nii.gz")

        for p in [ct_path, pt_path, label_path]:
            npy = p.replace(".nii.gz", ".npy")
            if os.path.exists(npy):
                p = npy

        ct_path    = ct_path.replace(".nii.gz",    ".npy") if os.path.exists(ct_path.replace(".nii.gz",    ".npy")) else ct_path
        pt_path    = pt_path.replace(".nii.gz",    ".npy") if os.path.exists(pt_path.replace(".nii.gz",    ".npy")) else pt_path
        label_path = label_path.replace(".nii.gz", ".npy") if os.path.exists(label_path.replace(".nii.gz", ".npy")) else label_path

        out_ct    = os.path.join(out_dir, f"{cid}_CT_valid.npy")
        out_pt    = os.path.join(out_dir, f"{cid}_PT_valid.npy")
        out_label = os.path.join(out_dir, f"{cid}_label_valid.npy")

        if os.path.exists(out_ct) and os.path.exists(out_pt) and os.path.exists(out_label):
            print(f"  [SKIP] {cid}: already exists")
            continue

        valid_z = valid_z_map[cid]

        ct_vol    = load_volume(ct_path)
        ct_valid  = ct_vol[:, :, valid_z].copy()
        del ct_vol
        np.save(out_ct, ct_valid)
        del ct_valid

        pt_vol    = load_volume(pt_path)
        pt_valid  = pt_vol[:, :, valid_z].copy()
        del pt_vol
        np.save(out_pt, pt_valid)
        del pt_valid

        mk_vol    = load_volume(label_path)
        mk_valid  = mk_vol[:, :, valid_z].copy()
        del mk_vol
        np.save(out_label, mk_valid)
        del mk_valid

        print(f"  [{cid}] saved: {len(valid_z)} slices, shape {ct_valid.shape if False else '?'}")

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       type=str, required=True)
    parser.add_argument("--train_csv",  type=str, required=True)
    parser.add_argument("--val_csv",    type=str, required=True)
    parser.add_argument("--test_csv",   type=str, required=True)
    parser.add_argument("--vz_dir",     type=str, required=True)
    parser.add_argument("--out_dir",    type=str, required=True)
    args = parser.parse_args()

    for split, csv_path in [("train", args.train_csv),
                             ("val",   args.val_csv),
                             ("test",  args.test_csv)]:
        print(f"\n=== {split} ===")
        process(
            valid_z_csv   = os.path.join(args.vz_dir, f"{split}_valid_z.csv"),
            split_csv     = csv_path,
            resampled_root= args.root,
            out_dir       = args.out_dir,
        )

# python vaild_num.py \
#   --root     /data2/2025_san/data/HECKTOR2025_Task1_resampled_1mm \
#   --train_csv /data2/2025_san/data/HECKTOR2025_splits_random/train.csv \
#   --val_csv   /data2/2025_san/data/HECKTOR2025_splits_random/val.csv \
#   --test_csv  /data2/2025_san/data/HECKTOR2025_splits_random/test.csv \
#   --vz_dir    /data2/2025_san/data/HECKTOR2025_splits_random/valid_z \
#   --out_dir   /data2/2025_san/data/HECKTOR2025_Task1_resampled_1mm/valid_slices