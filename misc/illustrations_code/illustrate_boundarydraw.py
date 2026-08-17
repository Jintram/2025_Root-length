
# %% libs

import numpy as np
import matplotlib.pyplot as plt

# %% cmap

# Define the colors as an array of hex codes; this will correspond to 
# classes that are segmented
custom_colors_plantclasses = [
    '#000000', # class 0, background, black
    '#90EE90', # class 1, shoot, green
    '#FFFFFF', # class 2, root, white
    '#A52A2A', # class 3, seed, brown
    '#006400', # class 4, leaf, dark green
    '#FF0000', # optional bright red color
]

# now convert to ListedColormap 
from matplotlib.colors import ListedColormap
cmap_custom_plantclasses = ListedColormap(custom_colors_plantclasses)

# %% example data

# How the mask was generated
#
# segfile_path = curr_file.fullpath
# segfile_data = np.load(segfile_path, allow_pickle=True)
# img_mask = segfile_data['img_pred_lbls']
# plt.imshow(img_mask[2000:2500, 300:800])
#
# plt.imshow(img_mask[500:1500, 500:1500]); plt.show()
# np.savez("example_mask.npz", mask = img_mask[500:1500, 500:1500])
# np.savez("example_mask2.npz", mask = img_mask[2000:2500, 300:800])

# Load the mask
mask_example = np.load("/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length/misc/illustrations_code/example_mask.npz", allow_pickle=True)["mask"]

# Show
# %matplotlib qt
LOCATION = (201, 565)
plt.imshow(mask_example)
plt.plot(LOCATION[0], LOCATION[1], "xw") 
plt.show()


# %% illustrate function "closest points in bg left/right"

import root_length.functions_pipeline.edit_segfiles as pl_seg

mask_withline = pl_seg.correct_mask_rootshootline(
    mask=mask_example, 
    row=LOCATION[1], 
    col=LOCATION[0])
plt.imshow(mask_withline)
plt.plot(LOCATION[0], LOCATION[1], "xw") 
plt.show()

# plot of how it works
# THIS CODE ONLY WORKS IF correct_mask_rootshootline() WAS MANUALLY EXECUTED
plt.imshow(distances, vmin=0, vmax=35); plt.imshow(region, cmap=cmap_custom_plantclasses, alpha=(region>0)*1.0); 
offset = np.array([row_min, col_min])
plt.plot(LOCATION[1]-offset[0], LOCATION[0]-offset[1], "xk", markersize=30)
plt.plot(left_idx[1], left_idx[0], "xw", markersize=30)
plt.plot(right_idx[1], right_idx[0], "xw", markersize=30)
plt.axvline(LOCATION[0]-offset[1], color="w")

# %% Example where it goes wrong

mask_example2 = np.load("/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length/misc/illustrations_code/example_mask2.npz", allow_pickle=True)["mask"]
plt.imshow(mask_example2); plt.show()
LOCATION2 = (293, 314)


mask_withline = pl_seg.correct_mask_rootshootline(
    mask=mask_example2, 
    row=LOCATION2[1], 
    col=LOCATION2[0])
plt.imshow(mask_withline)
plt.plot(LOCATION2[0], LOCATION2[1], "xw") 
plt.show()

# plot of how it works
# THIS CODE ONLY WORKS IF correct_mask_rootshootline() WAS MANUALLY EXECUTED
plt.imshow(distances, vmin=0, vmax=35); plt.imshow(region, cmap=cmap_custom_plantclasses, alpha=(region>0)*1.0); 
offset = np.array([row_min, col_min])
plt.plot(LOCATION2[1]-offset[0], LOCATION2[0]-offset[1], "xk", markersize=30)
plt.plot(left_idx[1], left_idx[0], "xw", markersize=30)
plt.plot(right_idx[1], right_idx[0], "xw", markersize=30)
plt.axvline(LOCATION2[0]-offset[1], color="w")


# %% illustrate function "draw from closest point"



