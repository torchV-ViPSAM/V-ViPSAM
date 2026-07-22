import random
import os
import numpy as np
import nibabel as nib
from skimage import transform

def load_volume(path):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode='r')
    # print("==================================")
    # print(f"[nii.file]{os.path.basename(path)}")
    # print("==================================")
    return nib.load(path).get_fdata().astype(np.float32)

def resize3d(vol, size, num_frames, is_mask=False):
    return transform.resize(
        vol, (size, size, num_frames), 
        order=0 if is_mask else 3, 
        preserve_range=True, 
        anti_aliasing=False if is_mask else True,
        ).astype(np.float32)

def norm_ct(vol):
    vol = np.clip(vol, -100, 200)
    return ((vol + 100) / 300.0).astype(np.float32)

def norm_mr(vol):
    mn, mx = float(vol.min()), float(vol.max())
    return ((vol - mn) / (mx - mn + 1e-8)).astype(np.float32)


def to_3c(hw):
    return np.repeat(hw[None, ...], 3, axis=0).astype(np.float32)


def nonempty_slices(mask_vol):
    valid = np.any(mask_vol > 0, axis=(0, 1))
    return np.flatnonzero(valid).astype(int)


def crop_bbox_from_ct(ct_vol, valid_slices, crop_thr, crop_margin):
    H, W, _ = ct_vol.shape
    if valid_slices.size == 0:
        return (0, H - 1, 0, W - 1)
    ct_sub = ct_vol[:, :, valid_slices]
    nz = np.argwhere(ct_sub > crop_thr)
    if nz.size == 0:
        return (0, H - 1, 0, W - 1)
    x0, x1 = int(nz[:, 0].min()), int(nz[:, 0].max())
    y0, y1 = int(nz[:, 1].min()), int(nz[:, 1].max())
    x0 = max(0, x0 - crop_margin)
    y0 = max(0, y0 - crop_margin)
    x1 = min(H - 1, x1 + crop_margin)
    y1 = min(W - 1, y1 + crop_margin)
    return (x0, x1, y0, y1)


def apply_crop(x2d, bbox):
    x0, x1, y0, y1 = bbox
    return x2d[x0 : x1 + 1, y0 : y1 + 1]

def random_patch_crop(ct2d, mr2d, mk2d, scale_range=(0.7, 1.0), is_lesion=False):
    H, W, T = ct2d.shape

    scale  = random.uniform(scale_range[0], scale_range[1])
    crop_h = int(H * scale)
    crop_w = int(W * scale)

    # fg/bg 비율 1:1
    use_fg = random.random() < 0.5

    fg_slices = [t for t in range(T) if mk2d[:, :, t].sum() > 0]
    has_fg    = len(fg_slices) > 0

    if use_fg and has_fg:
        # foreground 기준 crop
        t_ref    = random.choice(fg_slices)
        ys, xs   = np.where(mk2d[:, :, t_ref] > 0)
        center_y = int(np.mean(ys))
        center_x = int(np.mean(xs))

        max_offset_y = int(crop_h * 0.3)
        max_offset_x = int(crop_w * 0.3)
        offset_y = random.randint(-max_offset_y, max_offset_y)
        offset_x = random.randint(-max_offset_x, max_offset_x)

        y1 = int(np.clip(center_y - crop_h // 2 + offset_y, 0, H - crop_h))
        x1 = int(np.clip(center_x - crop_w // 2 + offset_x, 0, W - crop_w))
    else:
        # random crop (배경 포함 가능)
        y1 = random.randint(0, max(0, H - crop_h))
        x1 = random.randint(0, max(0, W - crop_w))

    return (
        ct2d[y1:y1+crop_h, x1:x1+crop_w, :],
        mr2d[y1:y1+crop_h, x1:x1+crop_w, :],
        mk2d[y1:y1+crop_h, x1:x1+crop_w, :],
    )

def norm_ct_hecktor(vol: np.ndarray) -> np.ndarray:
    vol = np.clip(vol, -200, 200)
    return ((vol + 200) / 400.0).astype(np.float32)


def norm_pet(vol: np.ndarray) -> np.ndarray:
    vol = np.clip(vol, 0.0, 20.0)
    return (vol / 20.0).astype(np.float32)


def resize3d_cv2(vol, size, num_frames, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    out = np.zeros((size, size, num_frames), dtype=np.float32)
    for t in range(num_frames):
        out[:, :, t] = cv2.resize(
            vol[:, :, t].astype(np.float32),
            (size, size),
            interpolation=interp,
        )
    return out

def apply_augmentation(ct_stack, mr_stack, mask_stack, is_lesion=False):
    p     = 0.7 if is_lesion else 0.6
    flip_lr      = random.random() < p
    flip_ud      = random.random() < p
    rotate        = random.random() < p
    angle         = random.uniform(-10, 10) if is_lesion else random.uniform(-5, 5)
    do_intensity  = is_lesion and (random.random() < 0.7)
    ct_gamma      = random.uniform(0.7, 1.4)
    mr_gamma      = random.uniform(0.7, 1.4)

    for t in range(ct_stack.shape[0]):
        ct_t   = ct_stack[t]    # (3, H, W)
        mr_t   = mr_stack[t]
        mask_t = mask_stack[t]

        if flip_lr:
            ct_t   = ct_t[:, :, ::-1].copy()
            mr_t   = mr_t[:, :, ::-1].copy()
            mask_t = mask_t[:, ::-1].copy()

        if flip_ud:
            ct_t   = ct_t[:, ::-1, :].copy()
            mr_t   = mr_t[:, ::-1, :].copy()
            mask_t = mask_t[::-1, :].copy()

        if rotate:
            ct_rot   = transform.rotate(ct_t[0],   angle, resize=False, preserve_range=True, order=3)
            mr_rot   = transform.rotate(mr_t[0],   angle, resize=False, preserve_range=True, order=3)
            mask_rot = transform.rotate(mask_t,     angle, resize=False, preserve_range=True, order=0)

            ct_t   = np.repeat(ct_rot[None],   3, axis=0).astype(np.float32)
            mr_t   = np.repeat(mr_rot[None],   3, axis=0).astype(np.float32)
            mask_t = mask_rot.astype(np.float32)

        if do_intensity:
            ct_t = np.clip(ct_t ** ct_gamma, 0, 1)
            mr_t = np.clip(mr_t ** mr_gamma, 0, 1)

        ct_stack[t]   = ct_t
        mr_stack[t]   = mr_t
        mask_stack[t] = mask_t

    return ct_stack, mr_stack, mask_stack
