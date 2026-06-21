


# %% 

import glob
import imageio as io
import numpy as np

from pathlib import Path

# %%

OUTPUTDIR = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/originals_lowres_tiles/"

filelist = glob.glob("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/lowres-crop__*_img.npy")
filelist = glob.glob("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/*_tile_img.npy")

# loop over filelist, np.load the file, save it to tif in OUTPUTDIR
for file in filelist:
    # file = filelist[0]
    
    img = np.load(file)
    filename_base = Path(file).stem.replace("_img", "")
    outputfile = str(Path(OUTPUTDIR) / (filename_base + ".tif"))
    io.imwrite(outputfile, img)
# %%
