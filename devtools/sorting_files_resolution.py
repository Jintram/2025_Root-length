"""
This python file goes through a directory <MYDIR>, and moves each file
based on its X*Y pixel count (read via bioio) into one of three
subdirectories of <OUTDIR>: "low_res", "medium_res", or "high_res".

    low_res    : total pixels <  2000 * 2000
    medium_res : 2000 * 2000 <= total pixels <= 3000 * 3000
    high_res   : total pixels >  3000 * 3000

The file might be in a subdirectory of <MYDIR>, in which case
the subdirectory structure is copied to the new location.

Install bioio (using conda-forge channel) with 
conda install bioio
conda install bioio-tifffile
"""

# %%

import shutil
from pathlib import Path

from bioio import BioImage
import imageio as io

# Configure source and destination directories, plus the pixel-count thresholds.
MYDIR  = Path("/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/DATA_lowres/")
OUTDIR = Path("/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA_sorted/")
LOW_MAX_PIXELS  = 2000 * 2000   # strictly below -> low_res
HIGH_MIN_PIXELS = 3000 * 3000   # strictly above -> high_res

# %%

# Walk the source tree recursively; rglob yields every file at any depth.
for src in MYDIR.rglob("*"):
    if not src.is_file():
        continue

    # Read image dimensions with bioio; skip files it can't open.
    try:
        dims = BioImage(str(src)).dims
        num_pixels = int(dims.X) * int(dims.Y)
    except Exception as exc:
        # revert to imageio
        try:
            img = io.imread(str(src))
            num_pixels = img.shape[0] * img.shape[1]
        except Exception as exc:
            print(f"Skipping {src}: cannot read image dimensions ({exc})")
            continue

    # Pick destination bucket based on pixel count.
    if num_pixels < LOW_MAX_PIXELS:
        bucket = "low_res"
    elif num_pixels > HIGH_MIN_PIXELS:
        bucket = "high_res"
    else:
        bucket = "medium_res"

    # Group also by original file extension (lowercased, without the dot).
    img_extension = src.suffix.lstrip(".").lower() or "no_ext"

    # Preserve the subdirectory structure by replicating the relative path
    # of the file (relative to MYDIR) under the chosen extension/bucket.
    rel_path = src.relative_to(MYDIR)
    dest = OUTDIR / img_extension / bucket / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Move the file; shutil.move handles cross-filesystem moves transparently.
    print(f"Moving {src} -> {dest}")
    shutil.move(str(src), str(dest))


print("DONE")

# %%
