
"""
Analyze all plants in a segfile, save output to plots and .tsv files.

To switch between modes of displaying plots:
%matplotlib qt
%matplotlib inline
"""


################################################################################
# %% libraries

import numpy as np
from matplotlib import pyplot as plt

# load custom scripts
import functions_pipeline.utils as plutils
    # import importlib; importlib.reload(plutils)
import functions_pipeline.preprocessing as plprep
    # import importlib; importlib.reload(plprep)
import functions_pipeline.determine_length as pllen
    # import importlib; importlib.reload(pllen)

################################################################################
# %% Load data for 1 sample

def identify_plants(img_mask_clean):
        
    # Find the separate regions of interest corresponding to individual plants
    list_img_indivplants, img_mask_bbox_lblcount, img_mask_rprops = \
        plprep.find_individual_plants(img_mask_clean)
    
    # Now determine the selection of QC passed plants
    sel_plants = \
        (
            # contains precisely one root
            (img_mask_bbox_lblcount[:,1] ==1) & 
            # and at least one other category
            (img_mask_bbox_lblcount[:,[0,2,3]].any(axis=1))
        )
        
    # Highlight
    selected_bboxes = np.array(img_mask_rprops)[sel_plants]
    
    
    return list_img_indivplants, sel_plants, img_mask_rprops

def analyze_plate(segfile_path):
    # segfile_path = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SEGMENTATION/202602/segfiles/20250520/20250520_OY03.npz"

    # Load input
    img_mask = \
        np.load(segfile_path)['img_pred_lbls']

    # Clean the mask
    img_mask_clean = plprep.clean_mask(img_mask)
    # Find the plants
    list_img_indivplants, sel_plants, img_mask_rprops = \
        identify_plants(img_mask_clean)
    # Plot 
    plprep.plot_mask_and_bboxes(img_mask_clean, selected_bboxes) 

    # Identify 

    # plt.imshow(img_mask, cmap=plutils.cmap_plantclasses)

    # Preprocess the input



    # Build RootSample objects for each QC-passed plant
    current_sample_all_plants = []
    for idx, plant_mask in enumerate(list_img_indivplants[sel_plants]):
        
        root_mask = plant_mask == 2
        shoot_mask = plant_mask == 1
        original_bbox = \
            np.array(img_mask_rprops)[sel_plants][idx].bbox
        
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
        

