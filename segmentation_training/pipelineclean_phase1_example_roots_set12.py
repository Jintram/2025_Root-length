"""
This file was only used to generate the metadata file for the 
training of combined sets "1" and "2", which were already made 
earlier.
The test sets could be corrected using the code here, however,
if desired.
"""

# %% ###########################################################################
# Libraries

import sys
import os

import cheeky_cells.orchestrators.orchestrate_phase1_clean as o1
    # import importlib; importlib.reload(o1)
import cheeky_cells.annotating_data.dedicated_segmentation as cds
    # import importlib; importlib.reload(cds)

# custom color map
import cheeky_cells.prepostprocessing_input.ara_roots.ara_plotting as arootp

# %% ###########################################################################
# Configuration

# Directory where to find the images that you want to annotate
INPUTDIRECTORY = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/originals/"
# Directory with model-training related data (trained model itself, training data, ..)
TRAINING_DIR = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618/"

# Configuration for this dataset
config1 = o1.Phase1Config(
    inputdirectory = INPUTDIRECTORY,
    training_dir = TRAINING_DIR,
    tile_size = 5000,
    bg_percentile = 20,
    # file_formats = ('.jpg', '.tif'),
        # Note: "file_formats" needs to be a "tuple", either ('entry1',) or 
        # ('entry1', 'entry2') etc. Trailing comma required for length 1.
    segfn = cds.basicplantseg1,
    rescalegreyforseg=True,
    mylabelcolormap=arootp.custom_colors_plantclasses,
    file_infix='', # defaults to _tile for historic reasons, but we don't have tiles now
    # copy_originals=True,  # uncomment to stage originals into TRAINING_DIR/originals/
)

# to plug in model output (manually rename 'segfiles' first)
# COMMENT THIS OUT IF YOU DON'T HAVE ALREADY SEGMENTED DATA
# THE DIRECTORY segfiles_humancorr SHOULD BE MANUALLY CREATED
# SEE: documentation/use-segresults-newtraining.md
# config1.segfolder = os.path.join(config1.training_dir, 'segfiles_humancorr/')

o1.phase1_setup(config1)

# %% ###########################################################################
# Running it

FIX THIS!! SOMEHOW THIS GENERATES NEW FILES, BUT THAT'S NOT WHAT'S INTENDED

SEE DIR
/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618

if False:

    # First, adjust the metadata_imagefiles_autogen.xlsx as required
    # (and rename it to avoid overwriting behavior).
    # Set the path to the metadata file below
    METADATA_FILE = \
        config1.training_dir + "metadata_imagefiles_autogen.xlsx"

    # Update config1 accordingly
    config1.metadatafiles_path = METADATA_FILE
        
    # to use a custom napari edit function (not required, but can be convenient)  
    import root_length.functions_pipeline.edit_segfiles as pl_edit
        # import importlib; importlib.reload(pl_edit)
    config1.my_napari_function = pl_edit.edit_annotation_napari
        
    # now create annotated pictures
    o1.phase1_annotate(config1)
# %%
