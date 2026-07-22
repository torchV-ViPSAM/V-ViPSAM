import os
import csv
import numpy as np
from torch.utils.data import DataLoader

from .hecktor_dataset import HECKTORDataset

def scan_hecktor_cases_from_csv(csv_path: str, resampled_root: str,
                                 valid_z_csv: str = None,
                                 valid_slices_dir: str = None) -> list:
    cases = []
    skipped_missing = 0

    valid_z_map = {}
    if valid_z_csv and os.path.isfile(valid_z_csv):
        with open(valid_z_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row["case_id"].strip()
                valid_z_map[cid] = np.array([int(z) for z in row["valid_z"].split(",")],
                                             dtype=np.int64)
        print(f"[valid_z] {len(valid_z_map)} cases loaded from {os.path.basename(valid_z_csv)}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        cid = row.get("case_id", "").strip()
        if not cid:
            continue

        ct_path    = row.get("ct_out",    "").strip()
        pt_path    = row.get("pt_out",    "").strip()
        label_path = row.get("label_out", "").strip()

        if not ct_path:
            ct_path    = os.path.join(resampled_root, cid, f"{cid}__CT_1mm.nii.gz")
        if not pt_path:
            pt_path    = os.path.join(resampled_root, cid, f"{cid}__PT_1mm.nii.gz")
        if not label_path:
            label_path = os.path.join(resampled_root, cid, f"{cid}_label_1mm.nii.gz")

        ct_path    = ct_path.replace(".nii.gz", ".npy") if os.path.exists(ct_path.replace(".nii.gz", ".npy")) else ct_path
        pt_path    = pt_path.replace(".nii.gz", ".npy") if os.path.exists(pt_path.replace(".nii.gz", ".npy")) else pt_path
        label_path = label_path.replace(".nii.gz", ".npy") if os.path.exists(label_path.replace(".nii.gz", ".npy")) else label_path

        if not (os.path.isfile(ct_path) and os.path.isfile(pt_path) and os.path.isfile(label_path)):
            skipped_missing += 1
            continue

        case = {
            "case_id":    f"{cid}_tumor",
            "ct":         ct_path,
            "mr":         pt_path,
            "label":      label_path,
            "class_name": "tumor",
        }
        if cid in valid_z_map:
            case["valid_z"] = valid_z_map[cid]

        case_dir_vs = os.path.join(resampled_root, cid)
        ct_v = os.path.join(case_dir_vs, f"{cid}_CT_valid.npy")
        pt_v = os.path.join(case_dir_vs, f"{cid}_PT_valid.npy")
        lb_v = os.path.join(case_dir_vs, f"{cid}_label_valid.npy")
        if os.path.exists(ct_v) and os.path.exists(pt_v) and os.path.exists(lb_v):
            case["ct_valid"]    = ct_v
            case["pet_valid"]   = pt_v
            case["label_valid"] = lb_v

        cases.append(case)

    print(
        f"[HECKTOR] {os.path.basename(csv_path)}: "
        f"{len(cases)} patients → {len(cases)} cases, "
        f"skip_missing={skipped_missing}"
    )
    return cases


def get_hecktor_train_val_loaders(j: dict, cfg):
    root      = j["hecktor_resampled_root"]
    train_csv = j["hecktor_train_csv"]
    val_csv   = j["hecktor_val_csv"]

    vz_dir        = j.get("valid_z_dir", None)
    vs_dir        = j.get("valid_slices_dir", None)
    train_vz_csv  = os.path.join(vz_dir, "train_valid_z.csv") if vz_dir else None
    val_vz_csv    = os.path.join(vz_dir, "val_valid_z.csv")   if vz_dir else None

    train_cases = scan_hecktor_cases_from_csv(train_csv, root, train_vz_csv, vs_dir)
    val_cases   = scan_hecktor_cases_from_csv(val_csv,   root, val_vz_csv,   vs_dir)

    train_ds = HECKTORDataset(train_cases, image_size=cfg.image_size,
                               use_augmentation=True,
                               num_frames=cfg.num_frames, overlap=cfg.overlap)
    val_ds   = HECKTORDataset(val_cases,   image_size=cfg.image_size,
                               use_augmentation=False,
                               num_frames=cfg.num_frames, overlap=cfg.overlap)

    nw_tr = int(j.get("num_workers_train", 4))
    nw_va = int(j.get("num_workers_val",   2))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                               shuffle=True,  num_workers=nw_tr,
                               pin_memory=True, drop_last=True,persistent_workers=nw_tr > 0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                               shuffle=False, num_workers=nw_va,
                               pin_memory=True,persistent_workers=nw_va > 0)
    return train_loader, val_loader


def get_hecktor_test_loader(j: dict, cfg):
    root     = j["hecktor_resampled_root"]
    test_csv = j["hecktor_test_csv"]

    vz_dir   = j.get("valid_z_dir", None)
    vs_dir        = j.get("valid_slices_dir", None)
    test_vz_csv = os.path.join(vz_dir, "test_valid_z.csv") if vz_dir else None

    test_cases = scan_hecktor_cases_from_csv(test_csv, root, test_vz_csv,vs_dir)
    test_ds    = HECKTORDataset(test_cases, image_size=cfg.image_size,
                                 use_augmentation=False,
                                 num_frames=cfg.num_frames, overlap=cfg.overlap)
    nw = int(j.get("num_workers_val", 2))
    return DataLoader(test_ds, batch_size=1,
                      shuffle=False, num_workers=nw, pin_memory=True)