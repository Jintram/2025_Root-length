

################################################################################
# %% Libs

import imageio as iio
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# load closing image operation
from skimage.morphology import closing, disk

################################################################################
# %% Function: distance from each region to nearest other region

def region_distances(mask):
    """ 
    Find inter-island distances.
    
    Label connected components ("island"), then for each one find the min distance
    to the nearest pixel belonging to any *other* component.
    """

    # Label
    labels, n_labels = ndimage.label(mask)
    distances = np.array([], dtype=int)
    
    # Loop over each "island"
    for island_idx in range(1, n_labels + 1):

        # create mask for other "islands"
        mask_others = (labels > 0) & (labels != island_idx)
        if not mask_others.any():
            distances[island_idx] = 0
            continue

        # get distances to closest for each pixel in the island
        current_distances = ndimage.distance_transform_edt(~mask_others)
        # now select and save the smallest for island i
        current_distance_min = current_distances[labels == island_idx].min()
        distances = np.append(distances, (current_distance_min.astype(int)))
        
    return distances

################################################################################
# %% Load example image

img_test = iio.imread("../example_files/idealized_root_masks/root_with_hole.tif")
img_test = iio.imread("../example_files/idealized_root_masks/root_with_hole_2.tif")

################################################################################
# %%

# show example image
plt.imshow(img_test)
plt.show()

# determine the distance to any nearest other region
distances = region_distances(img_test > 0)
print(distances)

# now expand using
closing_distance = np.int64(1+np.ceil(distances.max()))
img_test_pad = np.pad(img_test, closing_distance)
plt.imshow(img_test_pad)
img_test_closed = closing(img_test_pad, disk(closing_distance))
plt.imshow(img_test_closed)

plt.imshow(img_test_closed+img_test_pad)

# now skeletonize this
from skimage import morphology
img_test_closed_skel = morphology.skeletonize(img_test_closed)
plt.imshow(img_test_closed_skel+img_test_pad)


# OTHER APPROACH
# perform dilation + erosion separately
from skimage.morphology import dilation, erosion
img_test_dilated = dilation(img_test_pad, disk(closing_distance))
plt.imshow(img_test_dilated)
img_test_dilero = erosion(img_test_dilated, disk(closing_distance))
plt.imshow(img_test_dilero)


# get skeleton of dilated mask
img_test_dilated2 = dilation(img_test_pad, disk(closing_distance/2+1))
plt.imshow(img_test_dilated2)
img_test_dila_skel = morphology.skeletonize(img_test_dilated2)
plt.imshow(img_test_dila_skel+img_test_pad)


# get skeleton of dilation + erosion (L-1) mask
# NOTE: THIS DOESN'T WORK -- because the "bridge" can still contain
# rather small neck parts
img_test_dilaero2 = erosion(img_test_dilated2, disk(closing_distance/2+3))
plt.imshow(img_test_dilaero2)
img_test_dilaero2_skel = morphology.skeletonize(img_test_dilaero2)
plt.imshow(img_test_dilaero2_skel+img_test_pad)



# Note: 
# It will be required to assign regions outside the original mask a root
# or shoot label based on what is closest. 


# %%
