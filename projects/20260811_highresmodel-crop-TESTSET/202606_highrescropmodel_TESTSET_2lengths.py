
################################################################################
# %% Load libraries

import os

import root_length.functions_files.filelisting as pl_flist
    # import importlib; importlib.reload(pl_flist)
import root_length.functions_pipeline.analyze_plate as pl_analyze
    # import importlib; importlib.reload(pl_analyze)
import root_length.functions_pipeline.edit_segfiles as pl_edit
    # import importlib; importlib.reload(pl_edit)

################################################################################
# %% Gather the file list df.

# dataset spcecific config
DIR_INPUTFILES = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/SEG_2026_highresmodel-crop_TESTSET/segfiles/'
DIR_OUTPUTFILES = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/LEN_2026_highresmodel-crop_TESTSET/'
    # DIR_OUTPUTFILES = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/LEN_2026_highresmodel-crop-test/'
DIR_IMAGEFILES = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/tif/high_res/20250527/'

# DIR_INPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/model_seg/segfiles/'
# DIR_OUTPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/model_seg/segfiles/test/'

# Generate list of files to analyze
df_filelist, metadata_toseg_filepath = \
    pl_flist.gen_metadatafile_segfiles(
        directory_inputfiles=DIR_INPUTFILES,
        directory_outputfiles=DIR_OUTPUTFILES,
    )
    # directory_inputfiles = DIR_INPUTFILES; directory_outputfiles = DIR_OUTPUTFILES


################################################################################
# %% Compute plate-area rect per file and store as `mask_rect` in each segfile.
# The rect can be corrected by hand in the napari editor below; the analysis
# ignores labels outside it when loading, so nothing needs clearing here.

# import importlib; importlib.reload(pl_edit)

pl_edit.compute_and_save_mask_rect_all(
    df_filelist=df_filelist,
    dir_inputfiles=DIR_INPUTFILES,
    dir_imagefiles=DIR_IMAGEFILES,
    only_process_n=None,        # int N for a test run, or None to process all
    clear_outside_mask=False,  # True to destructively zero labels outside the rect
    overwrite=False
)

################################################################################
# %% (Optional) Interactively edit segmentation files with napari
# Uncomment below to review and correct segmentations before analysis.

# DIR_IMAGEFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/Originals/' 

# for debugging:
# import importlib; importlib.reload(pl_edit)

pl_edit.edit_all_segfiles(df_filelist=df_filelist,
                         dir_inputfiles=DIR_INPUTFILES,
                         dir_imagefiles=DIR_IMAGEFILES)

# DEBUGGING REMOVE
# import numpy as np
# mytest = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/202602/SEG/segfiles/20250611/20250611_OY_16_seg.npz')
# mytest['prepr_info']
# mytest.keys()

################################################################################
# %% Run the analysis
# import importlib; importlib.reload(pl_analyze)

from root_length.functions_pipeline.config import ConfigPipeline
    # import importlib; importlib.reload(determine_length)

# Set configuration parameters
config_pipeline = \
    ConfigPipeline(
        # Smooths the root/shoots to avoid spurious branching
        smoothing_diskradius=5,
        dilation_radius_maximum=15,
        dpi_plots=1200
        )

# test run
pl_analyze.analyze_all_plates(df_filelist=df_filelist[:10],
                        output_dir=DIR_OUTPUTFILES, 
                        config_pipeline=config_pipeline)

# full run
pl_analyze.analyze_all_plates(df_filelist=df_filelist,
                        output_dir=DIR_OUTPUTFILES,
                        config_pipeline=config_pipeline)

# Now make one big overview dataframe
pl_analyze.generate_df_all(df_filelist, DIR_OUTPUTFILES)


