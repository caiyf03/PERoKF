import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

SRC_DIR = "/vast/projects/jgu32/lab/cai03/5200_dataset/size512/kf_2d_re1000/fn8/seed0/vorticity"
DST_DIR = "pic"

os.makedirs(DST_DIR, exist_ok=True)

# ---- Natural Sort ----
def natural_key(s):
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', s)]

files = sorted(glob.glob(os.path.join(SRC_DIR, "*.npy")), key=natural_key)

if not files:
    raise FileNotFoundError(f"No .npy files found in: {os.path.abspath(SRC_DIR)}")

saved = 0
for i, fp in enumerate(files):
    #print(f"[Reading] {fp}")

    if i % 16 != 0 and i != 159:
        continue

    data = np.load(fp, mmap_mode="r")
    w = data[0]

    plt.figure(figsize=(6,6))
    plt.imshow(w, cmap="twilight")
    plt.colorbar(label="Vorticity")
    plt.title(os.path.basename(fp))
    plt.axis("off")

    out_path = os.path.join(DST_DIR, os.path.basename(fp).replace(".npy", ".png"))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    saved += 1
    print(f"[Saved] {out_path}\n")

print(f"✅ Done. Saved {saved} images to {DST_DIR}")
