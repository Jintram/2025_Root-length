
################################################################################
# %% Load libraries

import os

import root_length.functions_files.filelisting as ffl
    # import importlib; importlib.reload(ffl)
import root_length.functions_pipeline.analyze_plate as plap
    # import importlib; importlib.reload(plap)
import root_length.functions_pipeline.edit_segfiles as pledit
    # import importlib; importlib.reload(pledit)

################################################################################
# %% Gather the file list df.

# dataset spcecific config
DIR_INPUTFILES = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/SEG_2026_highresmodel-crop/segfiles/'
DIR_OUTPUTFILES = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/LEN_2026_highresmodel-crop/'
DIR_IMAGEFILES = '/Users/m.wehrens/Data_notbacked/2025_hypocotyl_images/DATA/'

# DIR_INPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/model_seg/segfiles/'
# DIR_OUTPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/model_seg/segfiles/test/'

import root_length.functions_files.filelisting as gfl
    # import importlib; importlib.reload(gfl)

# Generate list of files to analyze
df_filelist, metadata_toseg_filepath = \
    gfl.gen_metadatafile_segfiles(
        directory_inputfiles=DIR_INPUTFILES,
        directory_outputfiles=DIR_OUTPUTFILES,
    )
    # directory_inputfiles = DIR_INPUTFILES; directory_outputfiles = DIR_OUTPUTFILES


################################################################################
# %% Compute plate-area rect per file and store as `mask_rect` in each segfile.
# Reconstruct the mask later via preprocessing.rect_to_mask(rect, shape).

# TO DO: option to clear areas outside the mask,
# TEST parameter that only touches the first X files

# import importlib; importlib.reload(pledit)

pledit.compute_and_save_mask_rect_all(
    df_filelist=df_filelist,
    dir_inputfiles=DIR_INPUTFILES,
    dir_imagefiles=DIR_IMAGEFILES,
    only_process_n=None,        # int N for a test run, or None to process all
    clear_outside_mask=True,   # True to also zero seg labels outside the rect
    overwrite=False
)

################################################################################
# %% (Optional) Interactively edit segmentation files with napari
# Uncomment below to review and correct segmentations before analysis.

# DIR_IMAGEFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SELECTION_ML/Originals/' 

# for debugging:
# import importlib; importlib.reload(pledit)

pledit.edit_all_segfiles(df_filelist=df_filelist,
                         dir_inputfiles=DIR_INPUTFILES,
                         dir_imagefiles=DIR_IMAGEFILES)

# DEBUGGING REMOVE
# import numpy as np
# mytest = np.load('/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/202602/SEG/segfiles/20250611/20250611_OY_16_seg.npz')
# mytest['prepr_info']
# mytest.keys()

################################################################################
# %% Run the analysis

# test run
plap.analyze_all_plates(df_filelist=df_filelist[:10],
                        output_dir=DIR_OUTPUTFILES)

# full run
plap.analyze_all_plates(df_filelist=df_filelist,
                        output_dir=DIR_OUTPUTFILES)

# Now make one big overview dataframe
plap.generate_df_all(df_filelist, DIR_OUTPUTFILES)





# %% misc code

# find plant with id 250502_OY_09
matching_idx = df_filelist.index[df_filelist["filename"].str.contains("250502_OY_09", na=False)]
print(matching_idx.tolist())



# checking out a sample, and why it doesn't have a rect?
import numpy as np
filepath = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/202602/SEG/segfiles/20250611/20250617_OY_15_seg.npz"
test2=np.load(filepath)