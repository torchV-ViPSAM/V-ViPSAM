# basic dataset 3dvipsam
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .prepare_data import load_volume,resize3d,norm_ct,norm_mr,to_3c,nonempty_slices,crop_bbox_from_ct,apply_crop, apply_augmentation

class CTMRDataset(Dataset):
    def __init__(self, data_list, image_size=1024, crop_thr=-950.0, crop_margin=0, use_augmentation=False, num_frames=4, overlap=2):
        self.data_list = data_list
        self.image_size = int(image_size)
        self.crop_thr = float(crop_thr)
        self.crop_margin = int(crop_margin)
        self.num_frames = int(num_frames)
        self.overlap = int(overlap)
        self.stride = max(1, self.num_frames - self.overlap)
        self.use_augmentation = use_augmentation
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
            # if L >= self.num_frames:
            #     starts = list(range(0, L - self.num_frames + 1, self.stride))
            #     last_start = L - self.num_frames
            #     if len(starts) == 0 or starts[-1] != last_start:
            #         starts.append(last_start)  
            #     for start_i in starts:
            #         samples.append((case_idx, start_i))
            if L >= self.num_frames:
                for start_i in range(0, L - self.num_frames + 1, self.stride):
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
        valid_z = nonempty_slices(mk_vol)
        valid_z = np.sort(valid_z)
        L = len(valid_z)

        if L >= self.num_frames:
            z_indices = list(valid_z[start_i : start_i + self.num_frames])
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

        ct3 = to_3c(norm_ct(ct2d)) # (3, H, W, T)
        mr3 = to_3c(norm_mr(mr2d))
        mask = (mk2d > 0.5).astype(np.float32)

        ct_stack = np.transpose(ct3, (3, 0, 1, 2)) # (T, 3, H, W)
        mr_stack = np.transpose(mr3, (3, 0, 1, 2)) 
        mask_stack= np.transpose(mask, (2, 0, 1)) # (T, H, W)
        
        # if self.use_augmentation:
        #     for t in range(self.num_frames):
        #         ct_stack[t], mr_stack[t], mask_stack[t] = apply_augmentation(ct_stack[t], mr_stack[t], mask_stack[t])

        if self.use_augmentation:
            is_lesion = (case.get("class_name", "") == "lesion")
            if not getattr(self, "_aug_logged", False):
                self._aug_logged = True
                print("[DEBUG] apply_augmentation called (augmentation ON)")
            if is_lesion:
                ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)
            else:
                ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)

        metric_size = self.image_size
        spacing = (float(sx) * (crop_w / metric_size), float(sy) * (crop_h / metric_size))

        return {
            "case_id": case["case_id"],
            "class_name": case.get("class_name", "unknown"),
            "ct": torch.from_numpy(ct_stack),
            "mr": torch.from_numpy(mr_stack),
            "mask": torch.from_numpy(mask_stack),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "start_i": start_i,
            "num_valid_z": L,
            "spacing_z": sz,
            "z_indices": torch.tensor(z_indices, dtype=torch.long),

        }

# # patch oversampling
# import numpy as np
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset

# from .prepare_data import load_volume,resize3d,norm_ct,norm_mr,to_3c,nonempty_slices,crop_bbox_from_ct,apply_crop, apply_augmentation,random_patch_crop

# class CTMRDataset(Dataset):
#     def __init__(self, data_list, image_size=1024, crop_thr=-950.0, crop_margin=0, use_augmentation=False, num_frames=4, overlap=2, is_train=False):
#         self.data_list = data_list
#         self.image_size = int(image_size)
#         self.crop_thr = float(crop_thr)
#         self.crop_margin = int(crop_margin)
#         self.num_frames = int(num_frames)
#         self.overlap = int(overlap)
#         self.stride = max(1, self.num_frames - self.overlap)
#         self.use_augmentation = use_augmentation
#         self.is_train = bool(is_train)
#         self.case_bbox = {}
#         self.samples = self._build_samples()

#     def _build_samples(self):
#         samples = []
#         for case_idx, case in enumerate(self.data_list):
#             mask_vol = load_volume(case["label"])
#             valid_z = nonempty_slices(mask_vol)
#             if valid_z.size == 0:
#                 continue
#             valid_z = np.sort(valid_z)
#             L = len(valid_z)
#             ct_vol = load_volume(case["ct"])
#             bbox = crop_bbox_from_ct(ct_vol, valid_z, self.crop_thr, self.crop_margin)
#             self.case_bbox[case_idx] = bbox
#             if L >= self.num_frames:
#                 for start_i in range(0, L - self.num_frames + 1, self.stride):
#                     samples.append((case_idx, start_i))
#             else:
#                 samples.append((case_idx, 0))
#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         case_idx, start_i = self.samples[idx]
#         case = self.data_list[case_idx]
#         bbox = self.case_bbox[case_idx]

#         ct_nii = nib.load(case["ct"])
#         sx, sy, sz = ct_nii.header.get_zooms()[:3]

#         ct_vol = load_volume(case["ct"])
#         mr_vol = load_volume(case["mr"])
#         mk_vol = load_volume(case["label"])
#         valid_z = nonempty_slices(mk_vol)
#         valid_z = np.sort(valid_z)
#         L = len(valid_z)

#         if L >= self.num_frames:
#             z_indices = list(valid_z[start_i : start_i + self.num_frames])
#         else:
#             z_indices = list(valid_z[:])
#         while len(z_indices) < self.num_frames:
#             z_indices.append(z_indices[-1])

#         ct2d = ct_vol[:, :, z_indices]
#         mr2d = mr_vol[:, :, z_indices]
#         mk2d = (mk_vol[:, :, z_indices] > 0).astype(np.float32)

#         ct2d = apply_crop(ct2d, bbox)
#         mr2d = apply_crop(mr2d, bbox)
#         mk2d = apply_crop(mk2d, bbox)

#         if self.is_train and self.use_augmentation:
#             is_lesion = (case.get("class_name", "") == "lesion")
#             ct2d, mr2d, mk2d = random_patch_crop(
#                 ct2d, mr2d, mk2d,
#                 scale_range=(0.7, 1.0) if is_lesion else (0.85,1.0),
#                 is_lesion=is_lesion,
#             )

#         crop_h, crop_w = ct2d.shape[0], ct2d.shape[1]

#         ct2d = resize3d(ct2d, self.image_size, self.num_frames, is_mask=False)
#         mr2d = resize3d(mr2d, self.image_size, self.num_frames, is_mask=False)
#         mk2d = resize3d(mk2d, self.image_size, self.num_frames, is_mask=True)

#         ct3 = to_3c(norm_ct(ct2d)) # (3, H, W, T)
#         mr3 = to_3c(norm_mr(mr2d))
#         mask = (mk2d > 0.5).astype(np.float32)

#         ct_stack = np.transpose(ct3, (3, 0, 1, 2)) # (T, 3, H, W)
#         mr_stack = np.transpose(mr3, (3, 0, 1, 2)) 
#         mask_stack= np.transpose(mask, (2, 0, 1)) # (T, H, W)
        
#         # if self.use_augmentation:
#         #     for t in range(self.num_frames):
#         #         ct_stack[t], mr_stack[t], mask_stack[t] = apply_augmentation(ct_stack[t], mr_stack[t], mask_stack[t])

#         if self.use_augmentation:
#             is_lesion = (case.get("class_name", "") == "lesion")
#             if not getattr(self, "_aug_logged", False):
#                 self._aug_logged = True
#                 print("[DEBUG] apply_augmentation called (augmentation ON)")
#             if is_lesion:
#                 ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)
#             else:
#                 ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)


#         metric_size = 256
#         spacing = (float(sx) * (crop_w / metric_size), float(sy) * (crop_h / metric_size))

#         return {
#             "case_id": case["case_id"],
#             "class_name": case.get("class_name", "unknown"),
#             "ct": torch.from_numpy(ct_stack),
#             "mr": torch.from_numpy(mr_stack),
#             "mask": torch.from_numpy(mask_stack),
#             "spacing": torch.tensor(spacing, dtype=torch.float32),
#             "start_i": start_i,
#             "num_valid_z": L,
#             "spacing_z": sz,
#         }

# # contain target 0 dataset
# import numpy as np
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset

# from .prepare_data import load_volume,resize3d,norm_ct,norm_mr,to_3c,nonempty_slices,crop_bbox_from_ct,apply_crop, apply_augmentation


# def _extend_valid_z(valid_z: np.ndarray, z_depth: int) -> np.ndarray:
#     valid_z = np.sort(np.asarray(valid_z, dtype=int).ravel())
#     if valid_z.size == 0:
#         return valid_z
#     out = []
#     lo = int(valid_z[0]) - 1
#     if lo >= 0:
#         out.append(lo)
#     out.extend(int(x) for x in valid_z.tolist())
#     hi = int(valid_z[-1]) + 1
#     if hi < int(z_depth):
#         out.append(hi)
#     return np.array(out, dtype=int)


# class CTMRDataset(Dataset):
#     def __init__(self, data_list, image_size=1024, crop_thr=-950.0, crop_margin=0, use_augmentation=False, num_frames=4, overlap=2, is_train=False):
#         self.data_list = data_list
#         self.image_size = int(image_size)
#         self.crop_thr = float(crop_thr)
#         self.crop_margin = int(crop_margin)
#         self.num_frames = int(num_frames)
#         self.overlap = int(overlap)
#         self.stride = max(1, self.num_frames - self.overlap)
#         self.use_augmentation = use_augmentation
#         self.is_train = bool(is_train)
#         self.case_bbox = {}
#         self.samples = self._build_samples()

#     def _build_samples(self):
#         samples = []
#         for case_idx, case in enumerate(self.data_list):
#             mask_vol = load_volume(case["label"])
#             valid_z = nonempty_slices(mask_vol)
#             if valid_z.size == 0:
#                 continue
#             valid_z = np.sort(valid_z)
#             if self.is_train:
#                 z_depth = int(mask_vol.shape[2])
#                 extended_z = _extend_valid_z(valid_z, z_depth)
#                 L = len(extended_z)
#                 ct_vol = load_volume(case["ct"])
#                 bbox = crop_bbox_from_ct(ct_vol, extended_z, self.crop_thr, self.crop_margin)
#             else:
#                 L = len(valid_z)
#                 ct_vol = load_volume(case["ct"])
#                 bbox = crop_bbox_from_ct(ct_vol, valid_z, self.crop_thr, self.crop_margin)
#             self.case_bbox[case_idx] = bbox
#             if L >= self.num_frames:
#                 for start_i in range(0, L - self.num_frames + 1, self.stride):
#                     samples.append((case_idx, start_i))
#             else:
#                 samples.append((case_idx, 0))
#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         case_idx, start_i = self.samples[idx]
#         case = self.data_list[case_idx]
#         bbox = self.case_bbox[case_idx]

#         ct_nii = nib.load(case["ct"])
#         sx, sy, sz = ct_nii.header.get_zooms()[:3]

#         ct_vol = load_volume(case["ct"])
#         mr_vol = load_volume(case["mr"])
#         mk_vol = load_volume(case["label"])
#         valid_z = nonempty_slices(mk_vol)
#         valid_z = np.sort(valid_z)
#         if self.is_train:
#             z_depth = int(mk_vol.shape[2])
#             pool_z = _extend_valid_z(valid_z, z_depth)
#             L = len(pool_z)
#         else:
#             pool_z = valid_z
#             L = len(valid_z)

#         if L >= self.num_frames:
#             z_indices = list(pool_z[start_i : start_i + self.num_frames])
#         else:
#             z_indices = list(pool_z[:])
#         while len(z_indices) < self.num_frames:
#             z_indices.append(z_indices[-1])

#         ct2d = ct_vol[:, :, z_indices]
#         mr2d = mr_vol[:, :, z_indices]
#         mk2d = (mk_vol[:, :, z_indices] > 0).astype(np.float32)

#         ct2d = apply_crop(ct2d, bbox)
#         mr2d = apply_crop(mr2d, bbox)
#         mk2d = apply_crop(mk2d, bbox)
#         crop_h, crop_w = ct2d.shape[0], ct2d.shape[1]

#         ct2d = resize3d(ct2d, self.image_size, self.num_frames, is_mask=False)
#         mr2d = resize3d(mr2d, self.image_size, self.num_frames, is_mask=False)
#         mk2d = resize3d(mk2d, self.image_size, self.num_frames, is_mask=True)

#         ct3 = to_3c(norm_ct(ct2d))
#         mr3 = to_3c(norm_mr(mr2d))
#         mask = (mk2d > 0.5).astype(np.float32)

#         ct_stack = np.transpose(ct3, (3, 0, 1, 2))
#         mr_stack = np.transpose(mr3, (3, 0, 1, 2))
#         mask_stack = np.transpose(mask, (2, 0, 1))

#         if self.use_augmentation:
#             is_lesion = (case.get("class_name", "") == "lesion")
#             if not getattr(self, "_aug_logged", False):
#                 self._aug_logged = True
#                 print("[DEBUG] apply_augmentation called (augmentation ON)")
#             if is_lesion:
#                 ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)
#             else:
#                 ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)

#         metric_size = 256
#         spacing = (float(sx) * (crop_w / metric_size), float(sy) * (crop_h / metric_size))

#         return {
#             "case_id": case["case_id"],
#             "class_name": case.get("class_name", "unknown"),
#             "ct": torch.from_numpy(ct_stack),
#             "mr": torch.from_numpy(mr_stack),
#             "mask": torch.from_numpy(mask_stack),
#             "spacing": torch.tensor(spacing, dtype=torch.float32),
#             "start_i": start_i,
#             "num_valid_z": L,
#             "num_nonempty_z": int(len(valid_z)),
#             "spacing_z": sz,
#         }

# import numpy as np
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset

# from .prepare_data import load_volume,resize3d,norm_ct,norm_mr,to_3c,nonempty_slices,crop_bbox_from_ct,apply_crop, apply_augmentation,random_patch_crop

# class CTMRDataset(Dataset):
#     def __init__(self, data_list, image_size=1024, crop_thr=-950.0, crop_margin=0, use_augmentation=False, num_frames=4, overlap=2, is_train=False):
#         self.data_list = data_list
#         self.image_size = int(image_size)
#         self.crop_thr = float(crop_thr)
#         self.crop_margin = int(crop_margin)
#         self.num_frames = int(num_frames)
#         self.overlap = int(overlap)
#         self.stride = max(1, self.num_frames - self.overlap)
#         self.use_augmentation = use_augmentation
#         self.is_train = bool(is_train)
#         self.case_bbox = {}
#         self.samples = self._build_samples()

#     def _build_samples(self):
#         samples = []
#         for case_idx, case in enumerate(self.data_list):
#             mask_vol = load_volume(case["label"])
#             valid_z = nonempty_slices(mask_vol)
#             if valid_z.size == 0:
#                 continue
#             valid_z = np.sort(valid_z)
#             L = len(valid_z)
#             ct_vol = load_volume(case["ct"])
#             bbox = crop_bbox_from_ct(ct_vol, valid_z, self.crop_thr, self.crop_margin)
#             self.case_bbox[case_idx] = bbox
#             if L >= self.num_frames:
#                 for start_i in range(0, L - self.num_frames + 1, self.stride):
#                     samples.append((case_idx, start_i))
#             else:
#                 samples.append((case_idx, 0))
#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         case_idx, start_i = self.samples[idx]
#         case = self.data_list[case_idx]
#         bbox = self.case_bbox[case_idx]

#         ct_nii = nib.load(case["ct"])
#         sx, sy, sz = ct_nii.header.get_zooms()[:3]

#         ct_vol = load_volume(case["ct"])
#         mr_vol = load_volume(case["mr"])
#         mk_vol = load_volume(case["label"])
#         valid_z = nonempty_slices(mk_vol)
#         valid_z = np.sort(valid_z)
#         L = len(valid_z)

#         if L >= self.num_frames:
#             z_indices = list(valid_z[start_i : start_i + self.num_frames])
#         else:
#             z_indices = list(valid_z[:])
#         while len(z_indices) < self.num_frames:
#             z_indices.append(z_indices[-1])

#         ct2d = ct_vol[:, :, z_indices]
#         mr2d = mr_vol[:, :, z_indices]
#         mk2d = (mk_vol[:, :, z_indices] > 0).astype(np.float32)

#         ct2d = apply_crop(ct2d, bbox)
#         mr2d = apply_crop(mr2d, bbox)
#         mk2d = apply_crop(mk2d, bbox)

#         if self.is_train and self.use_augmentation:
#             is_lesion = (case.get("class_name", "") == "lesion")
#             ct2d, mr2d, mk2d = random_patch_crop(
#                 ct2d, mr2d, mk2d,
#                 scale_range=(0.7, 1.0) if is_lesion else (0.85,1.0),
#                 is_lesion=is_lesion,
#             )

#         crop_h, crop_w = ct2d.shape[0], ct2d.shape[1]

#         ct2d = resize3d(ct2d, self.image_size, self.num_frames, is_mask=False)
#         mr2d = resize3d(mr2d, self.image_size, self.num_frames, is_mask=False)
#         mk2d = resize3d(mk2d, self.image_size, self.num_frames, is_mask=True)

#         ct3 = to_3c(norm_ct(ct2d)) # (3, H, W, T)
#         mr3 = to_3c(norm_mr(mr2d))
#         mask = (mk2d > 0.5).astype(np.float32)

#         ct_stack = np.transpose(ct3, (3, 0, 1, 2)) # (T, 3, H, W)
#         mr_stack = np.transpose(mr3, (3, 0, 1, 2)) 
#         mask_stack= np.transpose(mask, (2, 0, 1)) # (T, H, W)
        
#         # if self.use_augmentation:
#         #     for t in range(self.num_frames):
#         #         ct_stack[t], mr_stack[t], mask_stack[t] = apply_augmentation(ct_stack[t], mr_stack[t], mask_stack[t])

#         if self.use_augmentation:
#             is_lesion = (case.get("class_name", "") == "lesion")
#             if not getattr(self, "_aug_logged", False):
#                 self._aug_logged = True
#                 print("[DEBUG] apply_augmentation called (augmentation ON)")
#             if is_lesion:
#                 ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)
#             else:
#                 ct_stack, mr_stack, mask_stack = apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=is_lesion)


#         metric_size = 256
#         spacing = (float(sx) * (crop_w / metric_size), float(sy) * (crop_h / metric_size))

#         return {
#             "case_id": case["case_id"],
#             "class_name": case.get("class_name", "unknown"),
#             "ct": torch.from_numpy(ct_stack),
#             "mr": torch.from_numpy(mr_stack),
#             "mask": torch.from_numpy(mask_stack),
#             "spacing": torch.tensor(spacing, dtype=torch.float32),
#             "start_i": start_i,
#             "num_valid_z": L,
#             "spacing_z": sz,
#         }

# class CTMRDataset(Dataset):
#     def __init__(self, data_list, image_size=1024, crop_thr=-950.0, crop_margin=0, use_augmentation=False, num_frames=4, overlap=2, is_training=False):
#         self.data_list      = data_list
#         self.image_size     = int(image_size)
#         self.crop_thr       = float(crop_thr)
#         self.crop_margin    = int(crop_margin)
#         self.num_frames     = int(num_frames)
#         self.overlap        = int(overlap)
#         self.stride         = max(1, self.num_frames - self.overlap)
#         self.use_augmentation = use_augmentation
#         self.is_training    = is_training
#         self.case_bbox      = {}
#         self._case_valid_z  = {}
#         self._build_cache()
#         self.samples = self._build_samples()

#     def _build_cache(self):
#         for case_idx, case in enumerate(self.data_list):
#             mask_vol = load_volume(case["label"])
#             valid_z  = nonempty_slices(mask_vol)
#             if valid_z.size == 0:
#                 continue
#             valid_z = np.sort(valid_z)
#             self._case_valid_z[case_idx] = valid_z
#             ct_vol = load_volume(case["ct"])
#             bbox   = crop_bbox_from_ct(ct_vol, valid_z, self.crop_thr, self.crop_margin)
#             self.case_bbox[case_idx] = bbox

#     def _build_samples(self):
#         samples = []
#         for case_idx in range(len(self.data_list)):
#             if case_idx not in self._case_valid_z:
#                 continue
#             valid_z = self._case_valid_z[case_idx]
#             L = len(valid_z)

#             if self.is_training:
#                 k = max(1, int(np.floor(L / 2)))
#                 for _ in range(k):
#                     samples.append((case_idx,))
#             else:
#                 if L >= self.num_frames:
#                     for start_i in range(0, L - self.num_frames + 1, self.stride):
#                         z_indices = list(valid_z[start_i : start_i + self.num_frames])
#                         samples.append((case_idx, z_indices))
#                 else:
#                     z_indices = self._center_pad(valid_z, L, self.num_frames)
#                     samples.append((case_idx, z_indices))
#         return samples

#     def _sample_random_window(self, valid_z, L):
#         T = self.num_frames
#         if L >= T:
#             c = np.random.randint(0, L)   # 중심 인덱스
#             left = T // 2 - 1             # T=4면 left=1
#             start = c - left
#             start = max(0, min(start, L - T))
#             z_indices = list(valid_z[start:start + T])
#         else:
#             z_indices = self._center_pad(valid_z, L, T)

#         return z_indices

#     def _center_pad(self, valid_z, L, T):
#         need = T - L
#         pad_front = need // 2
#         pad_back  = need - pad_front

#         z_indices = (
#             [valid_z[0]] * pad_front +
#             list(valid_z) +
#             [valid_z[-1]] * pad_back
#         )
#         return z_indices

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample   = self.samples[idx]
#         case_idx = sample[0]
#         case     = self.data_list[case_idx]
#         bbox     = self.case_bbox[case_idx]
#         valid_z  = self._case_valid_z[case_idx]
#         L        = len(valid_z)

#         if self.is_training:
#             z_indices = self._sample_random_window(valid_z, L)
#         else:
#             z_indices = sample[1]   # 고정 window

#         ct_nii = nib.load(case["ct"])
#         sx, sy, sz = ct_nii.header.get_zooms()[:3]

#         ct_vol = load_volume(case["ct"])
#         mr_vol = load_volume(case["mr"])
#         mk_vol = load_volume(case["label"])

#         ct2d = ct_vol[:, :, z_indices]
#         mr2d = mr_vol[:, :, z_indices]
#         mk2d = (mk_vol[:, :, z_indices] > 0).astype(np.float32)

#         ct2d = apply_crop(ct2d, bbox)
#         mr2d = apply_crop(mr2d, bbox)
#         mk2d = apply_crop(mk2d, bbox)
#         crop_h, crop_w = ct2d.shape[0], ct2d.shape[1]

#         ct2d  = resize3d(ct2d,  self.image_size, self.num_frames, is_mask=False)
#         mr2d  = resize3d(mr2d,  self.image_size, self.num_frames, is_mask=False)
#         mk2d  = resize3d(mk2d,  self.image_size, self.num_frames, is_mask=True)

#         ct3   = to_3c(norm_ct(ct2d))
#         mr3   = to_3c(norm_mr(mr2d))
#         mask  = (mk2d > 0.5).astype(np.float32)

#         ct_stack   = np.transpose(ct3,  (3, 0, 1, 2))
#         mr_stack   = np.transpose(mr3,  (3, 0, 1, 2))
#         mask_stack = np.transpose(mask, (2, 0, 1))

#         # if self.use_augmentation:
#         #     ct_stack, mr_stack, mask_stack = apply_augmentation(
#         #         ct_stack, mr_stack, mask_stack
#         #     )

#         if self.use_augmentation:
#             if not getattr(self, "_aug_logged", False):
#                 self._aug_logged = True
#                 print("[DEBUG] apply_augmentation called (augmentation ON)")
#             is_lesion = (case.get("class_name", "") == "lesion")
#             ct_stack, mr_stack, mask_stack = apply_augmentation(
#                 ct_stack, mr_stack, mask_stack, is_lesion=is_lesion
#             )

#         metric_size = 256
#         spacing = (float(sx) * (crop_w / metric_size),
#                    float(sy) * (crop_h / metric_size))

#         return {
#             "case_id":    case["case_id"],
#             "class_name": case.get("class_name", "unknown"),
#             "ct":         torch.from_numpy(ct_stack),
#             "mr":         torch.from_numpy(mr_stack),
#             "mask":       torch.from_numpy(mask_stack),
#             "spacing":    torch.tensor(spacing, dtype=torch.float32),
#             "num_valid_z": L,
#             "spacing_z":  sz,
#             "z_indices": torch.tensor(z_indices, dtype=torch.long),
#         }