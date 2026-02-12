
################################################################################
# %% Load libraries

# libraries
import numpy as np
from matplotlib import pyplot as plt

# custom script location
import sys
sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length')
# load custom scripts
import pipeline_functions.utils as plutils
    # import importlib; importlib.reload(plutils)
import pipeline_functions.preprocessing as plprep
    # import importlib; importlib.reload(plprep)

################################################################################
# %% Load data for 1 sample

# Load input
img_mask = \
    np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/plots/fullimages_predictions_all/predictedmask_idx001.npy")

# Show the input
%matplotlib qt
plt.imshow(img_mask, cmap=plutils.cmap_plantclasses)

################################################################################
# %% Preprocess the input

# Clean the mask
img_mask_clean = plprep.clean_mask(img_mask)
plt.imshow(img_mask_clean, cmap=plutils.cmap_plantclasses)

# Find the separate regions of interest corresponding to individual plants
list_img_isolatedplants, img_mask_bbox_rootcount, img_mask_rprops = \
    plprep.find_individual_plants(img_mask_clean)
selection_of_QCpassed_plants = img_mask_bbox_rootcount==1

# Highlight 
selected_bboxes = np.array(img_mask_rprops)[selection_of_QCpassed_plants]
plprep.plot_mask_and_bboxes(img_mask_clean, selected_bboxes)

################################################################################
# %% Now run a plant through the analysis
# afterwards, show each of the longest lines projected back on the plants

img_plant1_test = list_img_isolatedplants[selection_of_QCpassed_plants][0]
img_mask_labeled = img_plant1_test
plt.imshow(img_mask_labeled, cmap=plutils.cmap_plantclasses)



# %%
