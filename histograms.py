import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------
# 1) Choose a sample + field
# --------------------------
SAMPLE_DIR = Path("batch_50_random/sample_0000")  # <-- change if needed
FIELD_NAME = "P.nii.gz"                            # try: "P.nii.gz" (best), or "N.nii.gz", "S.nii.gz"
SEGM_NAME  = "segm.nii.gz"

# Small constant for log (avoid log(0))
EPS = 1e-8

# --------------------------
# 2) Load volumes
# --------------------------
field = nib.load(SAMPLE_DIR / FIELD_NAME).get_fdata().astype(np.float32)
segm  = nib.load(SAMPLE_DIR / SEGM_NAME).get_fdata().astype(np.float32)

assert field.shape == segm.shape, f"Shape mismatch: field {field.shape} vs segm {segm.shape}"

# --------------------------
# 3) Pick a good slice automatically
#    (slice with largest tumor area)
# --------------------------
tumor_area_per_z = (segm > 0).sum(axis=(0, 1))
z = int(np.argmax(tumor_area_per_z))
slice_raw = field[:, :, z]
slice_mask = segm[:, :, z] > 0

# If segm is empty (shouldn't be), fallback to middle slice
if tumor_area_per_z.max() == 0:
    z = field.shape[2] // 2
    slice_raw = field[:, :, z]
    slice_mask = slice_raw > np.percentile(slice_raw, 95)

# --------------------------
# 4) Compute log-transformed slice
# --------------------------
slice_log = np.log(slice_raw + EPS)

# For display only: robust normalization so it looks good on slides
def robust_norm(img, lo=1, hi=99):
    a, b = np.percentile(img, [lo, hi])
    img = np.clip(img, a, b)
    if b - a < 1e-12:
        return np.zeros_like(img)
    return (img - a) / (b - a)

raw_disp = robust_norm(slice_raw, 1, 99)
log_disp = robust_norm(slice_log, 1, 99)

# --------------------------
# 5) Pick a nice 1D line through tumor center (profile)
# --------------------------
ys, xs = np.where(slice_mask)
if len(xs) > 0:
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
else:
    cx = slice_raw.shape[1] // 2
    cy = slice_raw.shape[0] // 2

profile_raw = slice_raw[cy, :]
profile_log = np.log(profile_raw + EPS)

# Normalize profiles for overlay (so they plot nicely together)
pr = (profile_raw - profile_raw.min()) / (profile_raw.max() - profile_raw.min() + 1e-12)
pl = (profile_log - profile_log.min()) / (profile_log.max() - profile_log.min() + 1e-12)

# --------------------------
# 6) Make a clean slide-ready figure
# --------------------------
fig = plt.figure(figsize=(12, 6), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.25, wspace=0.08)

# Top: intensity profile
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(pr, label="Raw (normalized)")
ax0.plot(pl, label="Log (normalized)")
ax0.axvline(cx, linestyle="--", linewidth=1)
ax0.set_title(f"Why Raw Tumor Values Break Learning (Example: {FIELD_NAME}, slice z={z})")
ax0.set_xlabel("Position along row (voxels)")
ax0.set_ylabel("Normalized intensity")
ax0.legend(frameon=False)
ax0.grid(True, alpha=0.25)

# Bottom-left: raw slice
ax1 = fig.add_subplot(gs[1, 0])
ax1.imshow(raw_disp, cmap="gray")
ax1.scatter([cx], [cy], s=15)
ax1.set_title("Raw values (core dominates)")
ax1.axis("off")

# Bottom-right: log slice
ax2 = fig.add_subplot(gs[1, 1])
ax2.imshow(log_disp, cmap="gray")
ax2.scatter([cx], [cy], s=15)
ax2.set_title("Log-scaled values (boundary visible)")
ax2.axis("off")

# Save for slides
out_path = Path("slide_raw_vs_log.png")
plt.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.show()

print(f"✅ Saved slide-ready image: {out_path.resolve()}")
