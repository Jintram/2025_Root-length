"""
This file uses the `root_length/functions_pipeline/preprocessing_seg.py` 
function to find the cropping for previously made training images.
"""

# %%

import root_length.functions_pipeline.preprocessing_seg as ppseg

import imageio as io

import matplotlib.pyplot as plt

# %%

# let's test with one image
img_highres = io.imread("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/originals/highres__20251104batch24_OY_09.tif")
img_lowres = io.imread("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/lowres_originals_fullsize/20250802batch10_OY_06.jpg")
img = img_lowres
img = img_highres

# import importlib; importlib.reload(ppseg)
img_crop, rect = ppseg.preprocess_getbbox_insideplate2(img, 
                                                       margin_bottom=0.1,
                                                       margin_left=0.05, 
                                                       margin_right=0.05,
                                                       margin_top=0.1)
    # all to 0.05 is a tight fit
    # .1 for extra margin at bottom and top to remove rounded edge


plt.imshow(img_crop)

_ = plt.imshow(img)
_ = plt.axhline(rect[0], color='red')
_ = plt.axhline(rect[1], color='red')
_ = plt.axvline(rect[2], color='red')
_ = plt.axvline(rect[3], color='red')

# %%


