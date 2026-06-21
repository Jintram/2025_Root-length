

# %%

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

import imageio as io



# %%

custom_colors_plantclasses = [
    '#000000', # 0 = bg
    '#90EE90', # 1 = shoot
    '#FFFFFF', # 2 = root 
    '#A52A2A', # 3 = seed
    '#006400' # 4 = leaf
]
# now convert to ListedColormap
cmap_custom_plantclasses = ListedColormap(custom_colors_plantclasses)

seg_mask = np.load("/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/SEG_2026_highresmodel/segfiles/20250611/20250611_OY_02_seg.npz")["img_pred_lbls"]
plt.imshow(seg_mask, cmap =cmap_custom_plantclasses)




# %%

# Checking whether lowres img and seg files are matching

img_npy = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/lowres-crop__20250520_OY04_img.npy", allow_pickle=True)
seg_npy = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/lowres-crop__20250520_OY04_seg.npy", allow_pickle=True)
seg_npy_b2 = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/_BACKUP/humanseg_BACKUP2/lowres-crop__20250520_OY04_seg.npy")
seg_npy_b1 = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/_BACKUP/humanseg_BACKUP/lowres-crop__20250520_OY04_tile_seg.npy")

seg_npy_or = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/20250520_OY04_tile_seg.npy")
img_npy_or = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/20250520_OY04_tile_img.npy")


_ = plt.imshow(img_npy_or)
_ = plt.imshow(seg_npy_or, alpha=(seg_npy_or>0)*1.0)

# %%

another_img = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/20250614_OY_02-2_tile_img.npy")
_ = plt.imshow(another_img)

# %%

img = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/lowres-crop-tile__250506_OY_02_img.npy")
seg = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/lowres-crop-tile__250506_OY_02_seg.npy")

img_ori = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/20250731batch6_OY_02_tile_img.npy")
img_seg = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/20250731batch6_OY_02_tile_seg.npy")

plt.imshow(img_ori)
plt.imshow(img_seg)
