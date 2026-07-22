import os
import re
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader

from .apis_dataset import APISDataset


def _find_first_nii(case_dir, keywords):
    files = sorted(os.listdir(case_dir))
    for f in files:
        fl = f.lower()
        if not (fl.endswith(".nii.gz") or fl.endswith(".nii")):
            continue
        if all(k in fl for k in keywords):
            return os.path.join(case_dir, f)
    return None


def _scan_case_dir(case_dir):
    ct_path = (
        _find_first_nii(case_dir, ["ncct"])
    )
    mr_path = (
        _find_first_nii(case_dir, ["adc"])
    )
    
    masks_dir = os.path.join(case_dir, "masks")
    label_path = (
        _find_first_nii(masks_dir, ["mask"])
        or _find_first_nii(masks_dir, ["label"])
        or _find_first_nii(masks_dir, ["seg"])
    )
    return ct_path, mr_path, label_path


def _is_nonempty_mask(mask_path):
    try:
        m = nib.load(mask_path).get_fdata()
        return bool(np.any(m > 0))
    except Exception:
        return False


def _case_num(case_name):
    # train_000 -> 0
    m = re.search(r"(\d+)$", case_name)
    return int(m.group(1)) if m else -1


def scan_apis_cases(split_root):
    """
    split_root: /data2/.../APIS/train or /data2/.../APIS/test
    """
    cases = []
    skipped_missing = 0
    skipped_empty = 0

    if not os.path.isdir(split_root):
        print(f"[APIS] split root not found: {split_root}")
        return cases

    case_names = [
        d for d in sorted(os.listdir(split_root))
        if os.path.isdir(os.path.join(split_root, d)) and d.startswith("train_")
    ]

    for case_name in case_names:
        case_dir = os.path.join(split_root, case_name)
        ct_path, mr_path, label_path = _scan_case_dir(case_dir)

        if ct_path is None or mr_path is None or label_path is None:
            skipped_missing += 1
            continue

        if not _is_nonempty_mask(label_path):
            skipped_empty += 1
            continue

        cases.append({
            "case_id": case_name,
            "ct": ct_path,
            "mr": mr_path,
            "label": label_path,
            "class_name": "lesion",
        })

    print(
        f"[APIS] root={split_root}, valid={len(cases)}, "
        f"missing={skipped_missing}, empty={skipped_empty}"
    )
    return cases


def split_train_val_apis(train_cases, val_split=0.1, seed=42):
    # train_000 ~ train_052 내에서만 split
    cases = sorted(train_cases, key=lambda x: _case_num(x["case_id"]))
    rng = np.random.RandomState(seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)
    n_val = max(1, int(np.floor(len(cases) * float(val_split))))
    val_idx = set(idx[:n_val].tolist())

    train_out, val_out = [], []
    for i, c in enumerate(cases):
        if i in val_idx:
            val_out.append(c)
        else:
            train_out.append(c)
    return train_out, val_out


def get_apis_train_val_loaders(j: dict, cfg):
    apis_root = j["apis_root"]
    train_root = os.path.join(apis_root, "train")

    all_train_cases = scan_apis_cases(train_root)
    train_cases, val_cases = split_train_val_apis(
        all_train_cases, val_split=cfg.val_split, seed=cfg.seed
    )

    print(f"[APIS] train={len(train_cases)}, val={len(val_cases)}")

    train_ds = APISDataset(
        train_cases,
        image_size=cfg.image_size,
        use_augmentation=True,
        num_frames=cfg.num_frames,
        overlap=cfg.overlap,
    )
    val_ds = APISDataset(
        val_cases,
        image_size=cfg.image_size,
        use_augmentation=False,
        num_frames=cfg.num_frames,
        overlap=cfg.overlap,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=int(j.get("num_workers_train", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(j.get("num_workers_val", 2)),
        pin_memory=True,
    )
    return train_loader, val_loader


def get_apis_test_loader(j: dict, cfg):
    apis_root = j["apis_root"]
    test_root = os.path.join(apis_root, "test")

    test_cases = scan_apis_cases(test_root)
    print(f"[APIS] test={len(test_cases)}")

    test_ds = APISDataset(
        test_cases,
        image_size=cfg.image_size,
        use_augmentation=False,
        num_frames=cfg.num_frames,
        overlap=cfg.overlap,
    )

    return DataLoader(
        test_ds,
        batch_size=int(j.get("batch_size", cfg.batch_size)),
        shuffle=False,
        num_workers=int(j.get("num_workers", 4)),
        pin_memory=True,
    )