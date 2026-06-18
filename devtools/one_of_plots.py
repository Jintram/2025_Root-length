

# %%

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap





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
