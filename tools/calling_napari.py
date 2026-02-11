
# %%

import napari
from skimage.io import imread
import numpy as np

# %%

# Simple script to open a file and edit it with napari
def edit_w_napari(my_file):
    
    my_img = imread(my_file)
    
    viewer = napari.Viewer()
    
    # add the image as a labaled mask
    viewer.add_labels(my_img)
    
    napari.run()
    
    return my_img

# %%

testmask_file_path_1 = "/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length/example_files/idealized_root_masks/root_mask_1.tif"
testmask_file_path_2 = "/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length/example_files/idealized_root_masks/root_mask_2.tif"


# %% modify image 1

napari.settings.get_settings().application.ipy_interactive = False
modified_image_1  = edit_w_napari(testmask_file_path_1)

np.save("/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length/example_files/idealized_root_masks/root_mask_1_labeled.npy", 
        modified_image_1)

# %% modify image 2

napari.settings.get_settings().application.ipy_interactive = False
modified_image_2  = edit_w_napari(testmask_file_path_2)

np.save("/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length/example_files/idealized_root_masks/root_mask_2_labeled.npy", 
        modified_image_2)

# %%

napari.settings.get_settings().application.ipy_interactive = True