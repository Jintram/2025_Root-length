"""
This python file goes through a directory <MYDIR>, reads each image
file's X*Y pixel dimensions (via bioio, falling back to imageio), and
collects filename + resolution into a pandas DataFrame. The DataFrame
is then written as an Excel file to <OUTPUTDIR>.

Install bioio (using conda-forge channel) with
conda install bioio
conda install bioio-tifffile
"""

# %%

from pathlib import Path

import pandas as pd
from bioio import BioImage
import imageio as io

# Configure source directory and output location.
MYDIR     = Path("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/originals/")
OUTPUTDIR = Path("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/")

# %%

records = []

# Walk the source tree recursively; rglob yields every file at any depth.
for src in MYDIR.rglob("*"):
    if not src.is_file():
        continue

    # Read image dimensions with bioio; fall back to imageio; else skip.
    try:
        dims = BioImage(str(src)).dims
        x, y = int(dims.X), int(dims.Y)
    except Exception as exc:
        try:
            img = io.imread(str(src))
            y, x = int(img.shape[0]), int(img.shape[1])
        except Exception as exc:
            print(f"Skipping {src}: cannot read image dimensions ({exc})")
            continue

    records.append({
        "filename": src.name,
        "extension": src.suffix.lstrip(".").lower() or "no_ext",
        "x_pixels": x,
        "y_pixels": y
    })

df = pd.DataFrame(records)

out_file = OUTPUTDIR / "image_dimensions.xlsx"
df.to_excel(out_file, index=False)

print(f"DONE - wrote {len(df)} rows to {out_file}")

# %%
