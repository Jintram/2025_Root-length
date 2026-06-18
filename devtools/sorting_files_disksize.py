"""
This python file goes through a directory <MYDIR>, and moves each file
based on its size to either the dir <LOWRES> or the dir <HIGHRES>.

The file might be in a subdirectory of <MYDIR>, in which case
the subdirectory structure is copied to the new location.

"""

import os
import shutil
from pathlib import Path

# Configure source and destination directories, plus the size threshold (bytes).
MYDIR   = Path("/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/")
LOWRES  = Path("/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA_highres/")
HIGHRES = Path("/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA_lowres")
SIZE_THRESHOLD = 2 * 1024 * 1024  # 5 MB; files >= threshold go to HIGHRES

# Walk the source tree recursively; rglob yields every file at any depth.
for src in MYDIR.rglob("*"):
    if not src.is_file():
        continue

    # Pick destination root based on file size.
    dest_root = HIGHRES if src.stat().st_size >= SIZE_THRESHOLD else LOWRES

    # Preserve the subdirectory structure by replicating the relative path
    # of the file (relative to MYDIR) under the chosen destination root.
    rel_path = src.relative_to(MYDIR)
    dest = dest_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Move the file; shutil.move handles cross-filesystem moves transparently.
    print(f"Moving {src} -> {dest}")
    shutil.move(str(src), str(dest))

