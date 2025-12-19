
#%% 

# 
import sys
sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length')

#%% 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import morphology
from skimage.measure import regionprops

# custom libs
import custom_functions.remove_large_objects as cflo
    # import importlib; importlib.reload(cflo)

# Let's set a custom color scheme
custom_colors_plantclasses = \
    [   # background black
        '#000000', 
        # shoot light green
        '#90EE90',
        # root white
        '#FFFFFF', 
        # seed brown
        '#A52A2A', 
        # leaf darkgreen
        '#006400' 
        ]
# Create a custom cmap

cmap_plantclasses = ListedColormap(custom_colors_plantclasses)

#%% 

# assuming we have some input
img_mask = \
    np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/plots/fullimages_predictions_all/predictedmask_idx001.npy")

# now plot
plt.imshow(img_mask, cmap=cmap_plantclasses)

# binary mask
img_mask_binary = img_mask>0
plt.imshow(img_mask_binary)

# Let's say a typical plant is ±400 pixels high, and 5 pixels wide. That's
# 2000 px area;
TYPICAL_PLANT_SIZE = 2000
MIN_SIZE = TYPICAL_PLANT_SIZE/10
MAX_SIZE = TYPICAL_PLANT_SIZE*10
# remove small objects from the mask
img_mask_cleaned = morphology.remove_small_objects(img_mask_binary, min_size=MIN_SIZE)
# remove large objects from the mask
img_mask_cleaned = cflo.remove_large_objects(img_mask_cleaned, max_size=MAX_SIZE)
plt.imshow(img_mask_cleaned)

# now clean the original mask with labels
img_mask[~img_mask_cleaned] = 0
plt.imshow(img_mask, cmap=cmap_plantclasses)


# %% Now let's look whether I can perform some skeletonization

# First, isolate the root segments
img_mask_roots = img_mask==2
plt.imshow(img_mask_roots)

# again, remove small parts
mask_roots_cleaned = morphology.remove_small_objects(img_mask_roots, min_size=MIN_SIZE)
plt.imshow(mask_roots_cleaned)

# now let's get labeled map and regionprops
label_img_cleanroots = morphology.label(mask_roots_cleaned)
regions_cleanroots = regionprops(label_img_cleanroots)

# now, isolate the first root bbox imgage
first_root = regions_cleanroots[0]
minr, minc, maxr, maxc = first_root.bbox
mask_firstroot = mask_roots_cleaned[minr:maxr, minc:maxc]

plt.imshow(mask_firstroot)




# %% Now comes the challenge

# I want to be able to deal with curvy roots
# I want to ignore side-branches
# Probably I want to fit a spline locally
# But how do I determine what the main direction of the root is?
    # --> perhpas some opening/closing to determine the main path
    # nevertheless, if there are "real" side branches, how would I be able to
    # remove those?
    # 
    # Maybe also create an idealized root mask, with some challenges in it,
    # to see if I can deal with that..

testmask_file_path = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/idealized_root_images/root_mask_1.tif"

# Load it
mask_firstroot = plt.imread(testmask_file_path)

# now skeletonize this
skeleton_firstroot = morphology.skeletonize(mask_firstroot)

# side-to-side comparison of mask_firstroot and skeleton_firstroot
fig, axs = plt.subplots(1, 2)
axs[0].imshow(mask_firstroot)
axs[1].imshow(skeleton_firstroot)




