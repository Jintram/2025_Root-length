

# %%

import glob
from pathlib import Path

import numpy as np

# %%

filelist1 = glob.glob("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618_cleaned/originals/*")
filelist2 = glob.glob("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618_cleaned/humanseg/*_seg*")

filelist2_names = [str(Path(X).name) for X in filelist2]

for file1 in filelist1:
    # file1 = filelist1[0]
    filename_base = Path(file1).stem
    
    filename_seg = filename_base + "_seg.npy"
    
    # loose check
    # if np.any([filename_base in X for X in filelist2]):
    #     print(f"for {filename_base}, match found")
    # else:
    #     print(f"for {filename_base}, NO match found")
        
    if np.any([filename_seg == X for X in filelist2_names]):
        print(f"match for {filename_base}")
    else:
        print(f"NO match for {filename_base}")
    
        
    


# %%
