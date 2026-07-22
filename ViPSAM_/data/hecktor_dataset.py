import numpy as np
import nibabel as nib
import torch
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .prepare_data import load_volume, to_3c,nonempty_slices, apply_augmentation, norm_ct_hecktor, norm_pet, resize3d_cv2

class HECKTORDataset(Dataset):
    def __init__(self, data_list, image_size=512,
                 use_augmentation=False, num_frames=4, overlap=2):
        self.data_list        = data_list
        self.image_size       = int(image_size)
        self.use_augmentation = use_augmentation
        self.num_frames       = int(num_frames)
        self.overlap          = int(overlap)
        self.stride           = max(1, self.num_frames - self.overlap)
        self._case_meta: dict = {}  
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for case_idx, case in enumerate(self.data_list):

            if "valid_z" in case:
                valid_z = np.sort(case["valid_z"])
            else:
                mk_vol  = load_volume(case["label"])
                binary  = ((mk_vol == 1) | (mk_vol == 2)).astype(np.float32)
                del mk_vol
                valid_z = nonempty_slices(binary)
                del binary
                if valid_z.size == 0:
                    continue
                valid_z = np.sort(valid_z)

            L = len(valid_z)
            if L == 0:
                continue

            self._case_meta[case_idx] = {
                "valid_z": valid_z,
            }

            if L >= self.num_frames:
                for start_i in range(0, L - self.num_frames + 1, self.stride):
                    samples.append((case_idx, start_i))
            else:
                samples.append((case_idx, 0))

        print(f"[HECKTORDataset] {len(self._case_meta)} cases, "
            f"{len(samples)} samples total")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, start_i = self.samples[idx]
        case  = self.data_list[case_idx]
        meta  = self._case_meta[case_idx]

        valid_z = meta["valid_z"]
        L       = len(valid_z)

        if L >= self.num_frames:
            rel_sel = slice(start_i, start_i + self.num_frames)
            z_indices = list(valid_z[start_i:start_i + self.num_frames])
        else:
            rel_idx = list(range(L)) + [L-1] * (self.num_frames - L)
            rel_sel = rel_idx
            z_indices = list(valid_z[rel_idx])

        while len(z_indices) < self.num_frames:
            z_indices.append(z_indices[-1])

        if "ct_valid" in case:
            ct2d  = norm_ct_hecktor(
                np.load(case["ct_valid"],    mmap_mode='r')[:, :, rel_sel].copy())
            pet2d = norm_pet(
                np.load(case["pet_valid"],   mmap_mode='r')[:, :, rel_sel].copy())
            mk_raw = np.load(case["label_valid"], mmap_mode='r')[:, :, rel_sel].copy()
            mk2d  = ((mk_raw == 1) | (mk_raw == 2)).astype(np.float32)
            del mk_raw
        else:
            ct_vol  = load_volume(case["ct"])
            pet_vol = load_volume(case["mr"])
            mk_vol  = load_volume(case["label"])
            ct2d  = norm_ct_hecktor(ct_vol[:, :, valid_z[rel_sel]])
            pet2d = norm_pet(pet_vol[:, :, valid_z[rel_sel]])
            mk_raw = mk_vol[:, :, valid_z[rel_sel]]
            mk2d  = ((mk_raw == 1) | (mk_raw == 2)).astype(np.float32)
            del ct_vol, pet_vol, mk_vol, mk_raw

        # 이하 기존과 동일
        ct2d  = resize3d_cv2(ct2d,  self.image_size, self.num_frames, is_mask=False)
        pet2d = resize3d_cv2(pet2d, self.image_size, self.num_frames, is_mask=False)
        mk2d  = resize3d_cv2(mk2d,  self.image_size, self.num_frames, is_mask=True)

        ct3   = to_3c(ct2d)
        pet3  = to_3c(pet2d)
        mask  = (mk2d > 0.5).astype(np.float32)

        ct_stack   = np.transpose(ct3,  (3, 0, 1, 2))
        pet_stack  = np.transpose(pet3, (3, 0, 1, 2))
        mask_stack = np.transpose(mask, (2, 0, 1))

        ct_stack, pet_stack, mask_stack = apply_augmentation(
                ct_stack, pet_stack, mask_stack,
                is_lesion=(case.get("class_name", "") == "tumor"),
            )

        return {
            "case_id":     case["case_id"],
            "class_name":  case.get("class_name", "unknown"),
            "ct":          torch.from_numpy(ct_stack),
            "mr":          torch.from_numpy(pet_stack),
            "mask":        torch.from_numpy(mask_stack),
            "spacing":     torch.tensor((1.0, 1.0), dtype=torch.float32),
            "start_i":     start_i,
            "num_valid_z": L,
            "spacing_z":   1.0,
            "z_indices":   torch.tensor(z_indices, dtype=torch.long),
        }