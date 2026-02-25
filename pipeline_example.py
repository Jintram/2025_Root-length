
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
import pipeline_functions.determine_length as pllen
    # import importlib; importlib.reload(pllen)

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
# %% Now set up a list of plant samples

# Build RootSample objects for each QC-passed plant
current_sample_all_plants = []
for idx, plant_mask in enumerate(list_img_isolatedplants[selection_of_QCpassed_plants]):
    
    root_mask = plant_mask == 2
    shoot_mask = plant_mask == 1
    original_bbox = \
        np.array(img_mask_rprops)[selection_of_QCpassed_plants][idx].bbox
    
    current_sample_all_plants.append(
        pllen.RootSample(
            root_mask=root_mask,
            shoot_mask=shoot_mask,
            plant_mask=plant_mask,
            bbox=original_bbox
        )
    )
    
# Now process all plants
for i, sample in enumerate(current_sample_all_plants):
    
    print(f"Currently processing plant {i+1} of {len(current_sample_all_plants)}")
    
    # Perform calculations based on this plant mask
    # and store the result in the same "plant container"
    current_sample_all_plants[i] = pllen.run_default_length_pipeline(sample)
    
    if current_sample_all_plants[i] is None:
        print("^ FAILED")


# Now plot the result
pllen.plot_all_plants_projected(
        sample_image = img_mask,
        plant_results = current_sample_all_plants)
    














################################################################################
# %% Now run a plant through the analysis
# afterwards, show each of the longest lines projected back on the plants

img_plant1_test = list_img_isolatedplants[selection_of_QCpassed_plants][0]
img_mask_labeled = img_plant1_test
plt.imshow(img_mask_labeled, cmap=plutils.cmap_plantclasses)



# %%


root_mask  = (img_mask_labeled == 2)
shoot_mask = (img_mask_labeled == 1)
current_rootsample = subfns.RootSample(root_mask  = root_mask,
                                        shoot_mask = shoot_mask,
                                        plant_mask = img_mask_labeled)