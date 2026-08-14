# %% ###########################################################################

# Segments arabidopsis root/shoot dataset
# 
# This file uses the code in "cheeky_cells" to segment a test set of plants.

# %% ###########################################################################
# Libraries

import cheeky_cells.orchestrators.orchestrate_phase3_clean as o3
    # import importlib; importlib.reload(o3)
    # import importlib; importlib.reload(crw)

# Dataset-specific imports
import cheeky_cells.prepostprocessing_input.ara_roots.ara_plotting as plt_ara

import cheeky_cells.plotting.plotting as pp
    # import importlib; importlib.reload(pp)

# %% ###########################################################################
# Configuration

# dataset spcecific config
# Per-run segmentation output: convention <data_root>/SEGMENTATIONS_<date_id>/.
SEGMENTATION_DIR = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/SEG_2026_highresmodel-crop_TESTSET/'
CURRENT_MODEL = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618_cleaned/models/modelUNet20260619_2100__trained0d19h46m.pth'
DATA_DIR = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/tif/high_res/20250527/'

    
# Now initialize a configuration
config3_ara_root = o3.Phase3Config(
    segmentation_dir = SEGMENTATION_DIR,
    nr_classes = 5,
    nr_channels_input = 3, # (input is rgb, so 3 channels)
    model_checkpoint_to_load = CURRENT_MODEL,
    bg_percentile = 10,
    data_path_input = DATA_DIR,
    fn_specific_preprocessing = None, 
    fn_plotting = pp.overlayplot,
    cmap_custom = plt_ara.cmap_custom_plantclasses,
    DPI_plots = 1200
)

# Collect all files that are to be segmented, store data in metadata
config3_ara_root = o3.collect_filelist(config3_ara_root)
    # config = config3_ara_root
    
    # hacky option (do not use)
    # file_formats =["_img.npy"]
    
    
# now segment them
o3.segment_all_files(config3_ara_root,
                     # max_files_to_process=200, # for test-run purposes
                     overwrite_files=True
                     )

