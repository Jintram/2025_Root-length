"""
These function are intended to pre-process images before sending them to 
the segmentation algorithm.

They have been moved from the "cheeky_cells" repo to here.
"""


#%% ################################################################################
# File to pre-process input

import numpy as np

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, disk, binary_opening, binary_closing
from skimage.measure import label, regionprops

from skimage.filters import median

# now do some pre-processing

# %% Functions

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

# %%

def preprocess_getbbox_insideplate(img_in_raw, margin_left = 100, margin_right = 100, 
                                         margin_top = 250, margin_bottom = 250,
                                         min_expected_area = 500000):
    # img_in_raw = img_toseg    
    # margin_left = 100; margin_right = 100; margin_top = 250; margin_bottom = 250; min_expected_area = 500000
    # margin_left = 0.05; margin_right = 0.05; margin_top = 0.05; margin_bottom = 0.05; min_expected_area = 500000
       
    # convert to greyscale
    img_in_gray = rgb2gray(img_in_raw)
        # plt.hist(img_in_gray.ravel()); plt.show()
        # plt.imshow(img_in_gray); plt.show()
    
    # Create mask
    mask_otsu = img_in_gray > threshold_otsu(img_in_gray)
    
    # Fill all holes
    mask_filled = binary_fill_holes(mask_otsu)
        # plt.imshow(mask_filled); plt.show()
    
    # remove small parts    
    mask_cleaned = remove_small_objects(mask_filled, min_size=10000)
        # plt.imshow(mask_cleaned); plt.show()
    
    # now get bounding box around the mask
    r0, c0, r1, c1 = bbox_from_mask_light(mask_cleaned)
    # regions = regionprops(label(mask_cleaned))
    # largest_region_idx = np.argmax(np.array([r.area for r in regions]))
    # r1, c1, r2, c2 = regions[largest_region_idx].bbox
    
    # If bbox covering minimum area isn't identified, return full image
    if (r1-r0) * (c1-c0) < min_expected_area:
        print("No large enough region detected, returning full image")
        return img_in_raw.copy(), (0, img_in_raw.shape[0], 0, img_in_raw.shape[1])
    
    # now create the cropped image
    rect = (r0+margin_top, r1-margin_bottom, c0+margin_left, c1-margin_right)
    img_cropped = img_in_raw[rect[0]:rect[1], rect[2]:rect[3]].copy()
    # plt.imshow(img_cropped); plt.show()
    
    return img_cropped, rect


# %%

def preprocess_getbbox_insideplate2(img_in_raw, margin_left = 100, margin_right = 100, 
                                         margin_top = 250, margin_bottom = 250,
                                         min_expected_area = 500000):
    """
    returns recteangle coordinates based on an original image, 
    that can be used for cropping or masking.
    
    this function uses the 50th percentile for an initial mask, assuming
    the edges of the plate have higher-than-average intensities, 
    to identify a mask outline.
    NOTE, TO DO:
    this is not very elegant, and a bit ad-hoc, can be improved upon.
    
    margins can be given as fraction of image size, or in pixels.
    numbers >0 and <1 will be considered fractions.
    
    rect is built such that an image can be cropped as follows:
    img[rect[0]:rect[1], rect[2]:rect[3]]
    """
    # img_in_raw = img_toseg    
    # margin_left = 100; margin_right = 100; margin_top = 250; margin_bottom = 250; min_expected_area = 500000
    
    # convert relative margins to pixels if they are fraction
    if (margin_left > 0) and (margin_left <1):
        margin_left = int(img_in_raw.shape[1] * margin_left)
    if (margin_right > 0 ) and (margin_right <1):
        margin_right = int(img_in_raw.shape[1] * margin_right)
    if (margin_top > 0) and (margin_top <1):
        margin_top = int(img_in_raw.shape[0] * margin_top)
    if (margin_bottom > 0) and (margin_bottom <1):
        margin_bottom = int(img_in_raw.shape[0] * margin_bottom)
    if (min_expected_area > 0) and (min_expected_area <1):
        min_expected_area = int(img_in_raw.shape[0] * img_in_raw.shape[1] * min_expected_area)
    
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
    
    # If bbox covering minimum area isn't identified, return full image
    if (r1-r0) * (c1-c0) < min_expected_area:
        print("No large enough region detected, returning full image")
        return img_in_raw.copy(), (0, img_in_raw.shape[0], 0, img_in_raw.shape[1])
    
    # now create the cropped image
    rect = (r0+margin_top, r1-margin_bottom, c0+margin_left, c1-margin_right)
    img_cropped = img_in_raw[rect[0]:rect[1], rect[2]:rect[3]].copy()
    # plt.imshow(img_cropped); plt.show()
    
    return img_cropped, rect

# %%

def preprocess_erasebounds(img_in_raw, margin_left = 100, margin_right = 100, 
                                         margin_top = 250, margin_bottom = 250,
                                         min_expected_area = 500000):
    """ 
    Use preprocess_getbbox_insideplate2 erase non-foreground margins.
    
    Use preprocess_getbbox_insideplate2 function to establish the area of interest
    in the sample. Then modify the input image: set all pixels that fall outside 
    that area to the mode of the ROI. Return the modified image.
    
    Note: I wrote this function because there were segmentation issues
    due to rescaling issues in taking full sample vs auto-cropped samples.
    And I wanted to be able to generate predictions that are equal to the
    original size. 
    However, turns out tuning the bg_percentile parameter during rescaling
    already resolved the issue. So this function has become superfluous.
    """
    
    img_cropped, rect = preprocess_getbbox_insideplate2(
        img_in_raw, margin_left = margin_left, margin_right = margin_right, 
        margin_top = margin_top, margin_bottom = margin_bottom,
        min_expected_area = min_expected_area)
    
    the_mask = np.zeros_like(img_in_raw)
    
        # plt.imshow(the_mask*255)
    
    new_img = img_in_raw.copy()
    
    if len(img_in_raw.shape) > 2:
        the_mask[rect[0]:rect[1], rect[2]:rect[3], :] = 1
        mymode = [np.bincount(img_in_raw[:,:,the_chann].ravel()).argmax() for the_chann in range(img_in_raw.shape[2])]
        for ch in range(img_in_raw.shape[2]):
            new_img[the_mask[:,:,ch]==0, ch] = mymode[ch]
    else:
        the_mask[rect[0]:rect[1], rect[2]:rect[3]] = 1
        mymode = np.bincount(img_in_raw[:,:].ravel()).argmax()
        new_img[the_mask[:,:]==0] = mymode
    
    return new_img, rect

# %% 

# # now since we know there's no agar at the edges 
# # and there is agar at the center, we can take those 
# # values as references..    
# img_edgevalues_flat = np.concatenate([
#     img_in_gray[0:10, :].ravel(),
#     img_in_gray[-10:, :].ravel(),
#     img_in_gray[:, 0:10].ravel(),
#     img_in_gray[:, -10:].ravel()])

# (coord1,coord2) = (img_in_gray.shape[0]//3, img_in_gray.shape[0]//3*2)
# (coord3,coord4) = (img_in_gray.shape[1]//3, img_in_gray.shape[1]//3*2)
# img_center_values = img_in_gray[coord1:coord2, coord3:coord4].ravel()

# # for the center, we can take the median
# center_median = np.median(img_center_values)

# # take the 500th lowest value
# edge_lowval = np.sort(img_edgevalues_flat)[np.min([500, len(img_edgevalues_flat)-1])]

    

    


