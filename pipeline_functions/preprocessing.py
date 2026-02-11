
################################################################################
# %%

import numpy as np
from skimage import morphology
import custom_functions.remove_large_objects as cflo
from skimage.measure import label, regionprops

from matplotlib import pyplot as plt
import pipeline_functions.utils as plutils


################################################################################
# %%

def clean_mask(img_mask):
    
    # binary mask
    img_mask_binary = img_mask>0
    # plt.imshow(img_mask_binary)

    # Let's say a typical plant is ±400 pixels high, and 5 pixels wide. That's
    # 2000 px area;
    TYPICAL_PLANT_SIZE = 2000
    MIN_SIZE = TYPICAL_PLANT_SIZE/10
    MAX_SIZE = TYPICAL_PLANT_SIZE*10
    # remove small objects from the mask
    img_mask_cleaned = morphology.remove_small_objects(img_mask_binary, min_size=MIN_SIZE)
    # remove large objects from the mask
    img_mask_cleaned = cflo.remove_large_objects(img_mask_cleaned, max_size=MAX_SIZE)
    # plt.imshow(img_mask_cleaned)

    # now clean the original mask with labels
    img_mask[~img_mask_cleaned] = 0
    # plt.imshow(img_mask, cmap=cmap_plantclasses)
    
    return img_mask

################################################################################
# %% Look for valid bboxes

# Let's look for separate plants in this mask
# I will look for bounding boxes around the mask, and discard boxes that 
# have multiple root areas in it.

def plot_bboxes(img_mask, img_bboxes):
    
    plt.imshow(img_mask, cmap=plutils.cmap_plantclasses)
    
    for bbox in img_bboxes:
        minr, minc, maxr, maxc = bbox.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
    

def find_individual_plants(img_mask, lbl_of_interest=2, min_count=5):    
    """
    Find plant regions in the mask, and nr of root regions in each plant.
    
    Returns a list of images that correspond to separately recognized plants,
    and in addition a corresponding list of individually recognized root
    areas within each plant. (Plants with multiple root areas are likely
    invalid segmentations.)
    Also return regionprops for the simplified mask.
    """
    # img_mask = img_mask_clean.copy(); lbl_of_interest=2; min_count = 5
    
    img_mask_simplified = img_mask>0
    img_mask_simplified_lbl = label(img_mask_simplified)
    img_mask_rprops = regionprops(img_mask_simplified_lbl)
    # plt.imshow(img_mask)
    
    # create bbox, remove other objects outside current object of interest
    # then count objects
    img_mask_bbox_rootcount = np.zeros(len(img_mask_rprops), dtype = int)
    list_img_isolatedplants = np.array([None]*len(img_mask_rprops))
    for CURRENT_LABEL in range(1, len(img_mask_rprops)+1):
        # CURRENT_LABEL = 2
        
        # get current box 
        current_bbox = img_mask_rprops[CURRENT_LABEL-1].bbox
        current_img_bbox = img_mask[current_bbox[0]:current_bbox[2],
                                    current_bbox[1]:current_bbox[3]]
        current_img_lbl_bbox = img_mask_simplified_lbl[
                                    current_bbox[0]:current_bbox[2],
                                    current_bbox[1]:current_bbox[3]]
        
        # now remove the background from this bbox
        current_img_bbox[current_img_lbl_bbox != CURRENT_LABEL] = 0
            # plt.imshow(current_img_lbl_bbox != CURRENT_LABEL)
            # plt.imshow(current_img_bbox)
        
        # now count the separate instances of roots 
        current_img_bbox_rootmask = current_img_bbox == lbl_of_interest
            # plt.imshow(current_img_bbox_rootmask)
        rprops = regionprops(label(current_img_bbox_rootmask))
        # now count the roots in this image, only if size >= min_count
        root_areas = [rprop.area for rprop in rprops if rprop.area >= min_count]
        root_area_count = len(root_areas)

        img_mask_bbox_rootcount[CURRENT_LABEL-1] = root_area_count
        list_img_isolatedplants[CURRENT_LABEL-1] = current_img_bbox
        
    return list_img_isolatedplants, img_mask_bbox_rootcount, img_mask_rprops
    
def plot_separate_plants(list_img_isolatedplants, lbl_count, nr_to_plot = 10):
    """
    Create a figure with multiple panels, each corresponding to an individual
    plant, based on list_img_isolatedplants.
    """
    # nr_to_plot = 10; lbl_count = img_mask_bbox_rootcount
    
    list_img_isolatedplants_toplot = \
        list_img_isolatedplants[:np.min([len(list_img_isolatedplants), nr_to_plot])]
        
    n_plants = len(list_img_isolatedplants_toplot)
    n_rows = n_plants // 10
    n_cols = int(np.ceil(n_plants/n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes_flt = np.atleast_1d(axes).flatten()
    
    for i, img in enumerate(list_img_isolatedplants_toplot):
        if img is not None:
            axes_flt[i].imshow(img, cmap=plutils.cmap_plantclasses)
            axes_flt[i].set_title(f"Count: {lbl_count[i]}")
        else:
            axes_flt[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def plot_mask_and_bboxes(labeled_mask, the_rprops):
    """
    Simply plot a labeled mask in the plant color scheme, 
    and throw some bboxes on top using rprops.
    """
    
    plt.imshow(labeled_mask, cmap=plutils.cmap_plantclasses)
    
    for rprop in the_rprops:
        minr, minc, maxr, maxc = rprop.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
    
    
    
# %%
