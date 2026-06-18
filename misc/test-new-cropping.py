

# %%

import imageio as io

import numpy as np

import matplotlib.pyplot as plt

import scipy.ndimage as ndimage

from skimage.color import rgb2gray
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

img = io.imread("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/originals/highres__20251104batch24_OY_09.tif")



# %%

def autocrop(img):
    
    _ = plt.imshow(img)
    
    # convert to grayscale
    img = img.mean(axis=2)
    
    _ = plt.imshow(img)
    
    # median filter
    img_med = ndimage.median_filter(img, size=20)
    _ = plt.imshow(img_med)
    
    # closed image
    img_closed = ndimage.grey_closing(img_med, size=(20, 20))
    _ = plt.imshow(img_closed)
    
    # opened image
    img_opened = ndimage.grey_opening(img_closed, size=(20, 20))
    _ = plt.imshow(img_opened)
    
    # now take the median of the central 33% square of the opened image
    h, w = img_opened.shape
    median_value_fg = np.median(img_opened[h//3:2*h//3, w//3:2*w//3])
    # and take the median of all edge pixels
    edge_pixels = np.concatenate([img_opened[0, :], img_opened[-1, :], img_opened[:, 0], img_opened[:, -1]])
    median_value_bg = np.median(edge_pixels)
    # go 90% of the way from bg to fg
    threshold_value = median_value_bg + 0.5 * (median_value_fg - median_value_bg)
    # now check
    _ = plt.imshow(img_opened>threshold_value)
    
    # check mean in one direction
    mean_profile_i = np.mean(img_opened, axis=1)
    plt.plot(mean_profile_i)
    plt.ylim(0, np.max(mean_profile_i)*1.1)
    mean_profile_j = np.mean(img_opened, axis=0)
    
    # determine mean boundaries, based halfway point
    threshold_i = (np.max(mean_profile_i)-np.min(mean_profile_i)) / 2 + np.min(mean_profile_i)
    threshold_j = (np.max(mean_profile_j)-np.min(mean_profile_j)) / 2 + np.min(mean_profile_j)
    
    # show those
    plt.plot(mean_profile_i)
    plt.axhline(threshold_i, color='red')
    
    # determine locations of first and last pixels above threshold
    indices_i = np.where(mean_profile_i > threshold_i)[0]
    indices_j = np.where(mean_profile_j > threshold_j)[0]
    if len(indices_i) > 0 and len(indices_j) > 0:
        i_min, i_max = indices_i[0], indices_i[-1]
        j_min, j_max = indices_j[0], indices_j[-1]
    else:
        i_min, i_max, j_min, j_max = None, None, None, None
        
    # now plot those as lines on the image
    plt.imshow(img)
    plt.axhline(i_min, color='red')
    plt.axhline(i_max, color='red')
    plt.axvline(j_min, color='red')
    plt.axvline(j_max, color='red')
        
    
    # now find local contrast (ie local range within 50px proximity)
    img_localmin = ndimage.minimum_filter(img, size=50)
    img_localmax = ndimage.maximum_filter(img, size=50)
    img_delta = img_localmax - img_localmin
    
    _ = plt.imshow(img_delta, cmap="viridis")
    
    
    
# %% Previous function

def bbox_from_mask_light(mask: np.ndarray):
    """
    Get bbox coordinates surrounding all non-zero pixels in a 2D mask.
    """
    # mask = mask_cleaned
    
    # Check if mask is 2D
    if mask.ndim != 2:
        raise ValueError("Input mask must be 2D")
    
    # Project the 2D mask on X or Y (for efficiency)
    rows_any = mask.any(axis=1)
    cols_any = mask.any(axis=0)
        # plt.plot(range(len(rows_any)), rows_any)

    if not rows_any.any():
        return None

    # Recover coordinates of all non-zero pixels in the projections
    r_idx = np.flatnonzero(rows_any) # flat to drop empty dim
    c_idx = np.flatnonzero(cols_any)

    # Recover max and min for both dimensions to get the bbox
    r0, r1 = r_idx[0], r_idx[-1] + 1
    c0, c1 = c_idx[0], c_idx[-1] + 1
    
    return r0, c0, r1, c1

def preprocess_getbbox_insideplate2(img_in_raw, margin_left = 100, margin_right = 100, 
                                         margin_top = 250, margin_bottom = 250,
                                         min_expected_area = 500000):
    # img_in_raw = img_toseg
    # img_in_gray = img  
    # margin_left = 100; margin_right = 100; margin_top = 250; margin_bottom = 250; min_expected_area = 500000
    
    # convert to greyscale
    img_in_gray = rgb2gray(img_in_raw)
    # convert to integer, rescale 0.255
    img_in_gray = (img_in_gray/np.max(img_in_gray) * 255).astype(int)
        # plt.hist(img_in_gray.ravel(), bins=50); plt.show()
        # plt.imshow(img_in_gray); plt.show()
    
    # project the image onto 1d both column-wise and row-wise
    # img_proj_row = img_in_gray.mean(axis=1)
    # img_proj_col = img_in_gray.mean(axis=0)
    # now plot this as line
    # plt.plot(img_proj_row/np.max(img_proj_row)); plt.plot(img_proj_col/np.max(img_proj_col)); plt.show()
    
    # get percentile values
    # p1  = np.percentile(img_in_gray, 1)
    p50 = np.percentile(img_in_gray, 50)
    # p99 = np.percentile(img_in_gray, 99)
    
    # identify the background level using the mode
    # bg_level = np.bincount((img_in_gray.ravel()).astype(int)).argmax()
    threshold = p50
        # plt.imshow(img_in_gray<=threshold)
    
    # Create mask
    # mask_otsu = img_in_gray > threshold_otsu(img_in_gray)
    mask = (img_in_gray > threshold).astype(bool)
        # plt.imshow(mask)
        # %matplotlib qt
    
    # Erosion followed by dilation (opening) to remove small bright spots in the background
    # mask = binary_closing(mask, footprint=disk(5))
    # mask = binary_opening(mask, footprint=disk(5))
        # plt.imshow(mask); plt.show()
    
    # Fill all holes
    # mask_filled = binary_fill_holes(mask)
        # plt.imshow(mask_filled); plt.show()
    
    # remove small parts    
    mask_cleaned = remove_small_objects(mask, min_size=10000)
        # plt.imshow(mask_cleaned); plt.show()
    
    # now get bounding box around the mask
    r0, c0, r1, c1 = bbox_from_mask_light(mask_cleaned)
    # regions = regionprops(label(mask_cleaned))
    # largest_region_idx = np.argmax(np.array([r.area for r in regions]))
    # r1, c1, r2, c2 = regions[largest_region_idx].bbox
        plt.imshow(img)
        plt.axhline(r0, color='red')
        plt.axhline(r1, color='red')
        plt.axvline(c0, color='red')
        plt.axvline(c1, color='red')
    
    # If bbox covering minimum area isn't identified, return full image
    if (r1-r0) * (c1-c0) < min_expected_area:
        print("No large enough region detected, returning full image")
        return img_in_raw.copy(), (0, img_in_raw.shape[0], 0, img_in_raw.shape[1])
    
    # now create the cropped image
    rect = (r0+margin_top, r1-margin_bottom, c0+margin_left, c1-margin_right)
    img_cropped = img_in_raw[rect[0]:rect[1], rect[2]:rect[3]].copy()
    # plt.imshow(img_cropped); plt.show()
    
    return img_cropped, rect