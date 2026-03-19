import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# -----------------------------
# 1) Point this at ONE sample folder
# -----------------------------
SAMPLE_DIR = r"C:\School Work\Capstone\DCNN\batch_50_random\sample_0000"

P_PATH    = os.path.join(SAMPLE_DIR, "P.nii.gz")
N_PATH    = os.path.join(SAMPLE_DIR, "N.nii.gz")
SEGM_PATH = os.path.join(SAMPLE_DIR, "segm.nii.gz")

# -----------------------------
# 2) Load volumes
# -----------------------------
P = nib.load(P_PATH).get_fdata().astype(np.float32)
N = nib.load(N_PATH).get_fdata().astype(np.float32)
segm = nib.load(SEGM_PATH).get_fdata()

# Nibabel usually loads as (X, Y, Z). Convert to (Z, Y, X) for easy slicing.
P = np.transpose(P, (2, 1, 0))
N = np.transpose(N, (2, 1, 0))
segm = np.transpose(segm, (2, 1, 0))

# -----------------------------
# 3) Build a continuous tumor field (good for showing log vs raw)
# -----------------------------
tumor = P + N  # continuous density-like field

# Optional: pick the slice with the MOST tumor (better than just middle slice)
tumor_amount_per_slice = tumor.sum(axis=(1, 2))
z = int(np.argmax(tumor_amount_per_slice))

slice_raw = tumor[z]

# Tumor mask from segmentation (non-zero labels are tumor)
mask = (segm[z] > 0).astype(np.uint8)

# -----------------------------
# 4) Log transform
# -----------------------------
eps = 1e-6
slice_log = np.log(slice_raw + eps)

# -----------------------------
# 5) Make the figure "presentation quality"
#    - better contrast for log by clipping percentiles
# -----------------------------
# For raw, keep default scaling (or use percentile clip too if you want)
raw_vmin = 0
raw_vmax = np.percentile(slice_raw, 99.5)

# For log, percentile clip is almost always needed for good visuals
log_vmin = np.percentile(slice_log, 5)
log_vmax = np.percentile(slice_log, 99.5)

plt.figure(figsize=(12, 5))

# --- RAW ---
ax1 = plt.subplot(1, 2, 1)
ax1.set_title("Tumor Field (Raw): P + N")
im1 = ax1.imshow(slice_raw, cmap="hot", vmin=raw_vmin, vmax=raw_vmax)
# overlay segmentation outline
ax1.contour(mask, levels=[0.5], linewidths=1)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.axis("off")

# --- LOG ---
ax2 = plt.subplot(1, 2, 2)
ax2.set_title("Tumor Field (Log-Scaled): log(P + N + ε)")
im2 = ax2.imshow(slice_log, cmap="hot", vmin=log_vmin, vmax=log_vmax)
ax2.contour(mask, levels=[0.5], linewidths=1)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.axis("off")

plt.suptitle("Raw vs Log Transform on Tumor Representation (best-tumor slice)", fontsize=14)
plt.tight_layout()

out_path = os.path.join(SAMPLE_DIR, "tumor_raw_vs_log.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
