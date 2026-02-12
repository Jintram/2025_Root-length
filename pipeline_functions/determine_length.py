

# %%
# Main function (that calls subfunctions) to determine the length
# of a root based on an already segmented image of one individual plant.

import pipeline_functions.determine_length_subfunctions as subfns

def get_root_length(img_mask_labeled):
    """ Based on a segmentation mask of one individual plant, get root length.

    Args:
        img_mask_labeled: labeled mask that contains 1 individual plant
    """
    
    # Create a rootSample dataclass object
    root_mask  = (img_mask_labeled == 2)
    shoot_mask = (img_mask_labeled == 1)
    current_rootsample = subfns.RootSample(root_mask  = root_mask,
                                           shoot_mask = shoot_mask,
                                           plant_mask = img_mask_labeled)
        # plt.imshow(current_rootsample.root_mask)
    
    
    
    
    
    
    
    
    
    
    
    


