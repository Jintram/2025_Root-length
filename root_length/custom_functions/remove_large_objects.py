



"""Provides function to remove large objects."""


import numpy as np

from skimage.measure import label, regionprops


def remove_large_objects(img_mask_binary, max_size):
    """
    Using regionprops to remove large regions from binary mask.
    
    TO DO: would be more efficient to do this other way around (ie remove large
    regions from copied image)
    """
    
    # Label connected regions
    label_img = label(img_mask_binary)

    # Create an empty mask
    filtered_mask = np.zeros_like(img_mask_binary)

    # Filter regions by area
    for region in regionprops(label_img):
        if region.area <= max_size:
            filtered_mask[label_img == region.label] = 1
    
    return filtered_mask.astype(bool)


