import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .prepare_data import (
    load_volume, resize3d, norm_ct, norm_mr, to_3c,
    nonempty_slices, crop_bbox_from_ct, apply_crop, apply_augmentation
)

class APISDataset(Dataset):
    def __init__(
        self,
        data_list,
        image_size=1024,
        crop_thr=-950.0,
        crop_margin=0,
        use_augmentation=False,
        num_frames=4,
        overlap=2,
    ):
        self.data_list = data_list
        self.image_size = int(image_size)
        self.crop_thr = float(crop_thr)
        self.crop_margin = int(crop_margin)
        self.num_frames = int(num_frames)
        self.overlap = int(overlap)
        self.stride = max(1, self.num_frames - self.overlap)
        self.use_augmentation = bool(use_augmentation)
        self.case_bbox = {}
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for case_idx, case in enumerate(self.data_list):
            mask_vol = load_volume(case["label"])
            valid_z = nonempty_slices(mask_vol)

            if valid_z.size == 0:
                continue

            valid_z = np.sort(valid_z)
            L = len(valid_z)

            ct_vol = load_volume(case["ct"])
            bbox = crop_bbox_from_ct(ct_vol, valid_z, self.crop_thr, self.crop_margin)
            self.case_bbox[case_idx] = bbox

            if L >= self.num_frames:
                starts = list(range(0, L - self.num_frames + 1, self.stride))
                last_start = L - self.num_frames
                if len(starts) == 0 or starts[-1] != last_start:
                    starts.append(last_start)  
                for start_i in starts:
                    samples.append((case_idx, start_i))
            else:
                samples.append((case_idx, 0))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, start_i = self.samples[idx]
        case = self.data_list[case_idx]
        bbox = self.case_bbox[case_idx]

        ct_nii = nib.load(case["ct"])
        sx, sy, sz = ct_nii.header.get_zooms()[:3]

        ct_vol = load_volume(case["ct"])
        mr_vol = load_volume(case["mr"])
        mk_vol = load_volume(case["label"])

        valid_z = np.sort(nonempty_slices(mk_vol))
        L = len(valid_z)

        if L >= self.num_frames:
            z_indices = list(valid_z[start_i:start_i + self.num_frames])
        else:
            z_indices = list(valid_z[:])

        while len(z_indices) < self.num_frames:
            z_indices.append(z_indices[-1])

        ct2d = ct_vol[:, :, z_indices]
        mr2d = mr_vol[:, :, z_indices]
        mk2d = (mk_vol[:, :, z_indices] > 0).astype(np.float32)

        ct2d = apply_crop(ct2d, bbox)
        mr2d = apply_crop(mr2d, bbox)
        mk2d = apply_crop(mk2d, bbox)
        crop_h, crop_w = ct2d.shape[0], ct2d.shape[1]

        ct2d = resize3d(ct2d, self.image_size, self.num_frames, is_mask=False)
        mr2d = resize3d(mr2d, self.image_size, self.num_frames, is_mask=False)
        mk2d = resize3d(mk2d, self.image_size, self.num_frames, is_mask=True)

        # 별도 norm 함수 없이 기존 재사용
        ct3 = to_3c(norm_ct(ct2d))
        mr3 = to_3c(norm_mr(mr2d))
        mask = (mk2d > 0.5).astype(np.float32)

        ct_stack = np.transpose(ct3, (3, 0, 1, 2))
        mr_stack = np.transpose(mr3, (3, 0, 1, 2))
        mask_stack = np.transpose(mask, (2, 0, 1))

        if self.use_augmentation:
            ct_stack, mr_stack, mask_stack = apply_augmentation(
                ct_stack, mr_stack, mask_stack, is_lesion=True
            )

        metric_size = self.image_size
        spacing = (float(sx) * (crop_w / metric_size), float(sy) * (crop_h / metric_size))

        return {
            "case_id": case["case_id"],
            "class_name": "lesion",  # APIS는 단일 클래스
            "ct": torch.from_numpy(ct_stack),
            "mr": torch.from_numpy(mr_stack),
            "mask": torch.from_numpy(mask_stack),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "start_i": start_i,
            "num_valid_z": L,
            "spacing_z": sz,
            "z_indices": torch.tensor(z_indices, dtype=torch.long),

        }