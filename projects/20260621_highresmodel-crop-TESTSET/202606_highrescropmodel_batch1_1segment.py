# %% ###########################################################################

# Segments arabidopsis root/shoot dataset


# %% ###########################################################################
# Libraries

import cheeky_cells.orchestrators.orchestrate_phase3_clean as o3
    # import importlib; importlib.reload(o3)
    # import importlib; importlib.reload(crw)

# Dataset-specific imports
# To pre-process a raw image
import root_length.functions_pipeline.preprocessing_seg as pp_ara
import cheeky_cells.prepostprocessing_input.ara_roots.ara_plotting as plt_ara
    # import importlib; importlib.reload(pp_ara)

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
    fn_specific_preprocessing = None, # pp_ara.preprocess_getbbox_insideplate2,
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


# %%
# DEBUGGING

if False:
    
    # Given a filename (e.g. 20250620_OY_06), find index in the df_metadata_input
    np.where(df_metadata_input.loc[:,'filename'].str.contains('20250617_OY_17'))
    np.where(df_metadata_input.loc[:,'filename'].str.contains('20250617_OY_15'))
    


# %% 

# REMOVE THIS CODE

# import os
# for file_idx in range(452, 1000):
#     print(f"Processing file idx {file_idx} ..")
#     filepath_segfile = \
#         os.path.join(config.segmentation_dir, "segfiles/", 
#                         df_metadata_input.loc[file_idx, 'subdir'], 
#                         f'segfile_idx{file_idx:03d}.npz')
#     # remove filepath_segfile if it's there
#     if os.path.exists(filepath_segfile):
#         os.remove(filepath_segfile)
