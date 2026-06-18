"""
This file uses the `root_length/functions_pipeline/preprocessing_seg.py` 
function to find the cropping for previously made training images.
"""

# %%

import root_length.functions_pipeline.preprocessing_seg as ppseg

import imageio as io

import matplotlib.pyplot as plt

import glob.glob

from pathlib import Path

import numpy as np

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

# %% Quick check

np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/highres__20251104batch24_OY_09_img_enhanced.npy")
np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/highres__20251104batch24_OY_09_img.npy")
np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/highres__20251104batch24_OY_09_seg.npy")
np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/highres__20251104batch24_OY_09_transform.npy")


# %% now apply it to a list of files

PRIMARY_FILELIST  = glob.glob("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/originals/highres__*")
SECONDARY_DIR     = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/humanseg/"
OUTPUT_DIR = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/crop_highres"

for thefile in PRIMARY_FILELIST:
    # thefile = PRIMARY_FILELIST[0]
    
    # load the image
    img = io.imread(thefile)
    
    # crop the image
    img_crop, rect = ppseg.preprocess_getbbox_insideplate2(img, 
                                                           margin_bottom=0.1,
                                                           margin_left=0.05, 
                                                           margin_right=0.05,
                                                           margin_top=0.1)
    # plt.imshow(img_crop)
    
    # save it to the output dir, with a prefix "crop_"
    output_filepath = str(Path(OUTPUT_DIR) / ("crop_" + Path(thefile).name))
    io.imwrite(output_filepath, img_crop)
    
    # load the secondary hits
    secondary_hits = glob.glob(str(Path(SECONDARY_DIR) / Path(thefile).stem) + "*")
    
    # now crop secondary hits, applying the same rect
    for thefile2 in secondary_hits:
        # thefile2 = secondary_hits[1]
        
        img2 = np.load(thefile2)
        img2_crop = img2[rect[0]:rect[1], rect[2]:rect[3]]
        # plt.imshow(img2_crop)
        
        # save it to the output dir as well, also prefix "crop_"
        output_filepath2 = str(Path(OUTPUT_DIR) / ("crop_" + Path(thefile2).name))
        np.save(output_filepath2, img2_crop)
        
        
        
        
    


# %%

# Quickly check the result

img = io.imread("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/crop_highres/crop_highres__20251108batch25_OY_53.tif")

seg = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/crop_highres/crop_highres__20251108batch25_OY_53_seg.npy")

plt.imshow(img)
plt.imshow(seg, alpha=(seg>0)*1.0)

# seems perfect

# %%


# Quickly test something else

segfile = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/humanseg/250502_OY_09_tile_seg.npy")
segfile2 = np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618_BACKUP/BACKUP/humanseg/lowres-crop__20250520_OY04_tile_seg.npy")

plt.imshow(segfile2)
