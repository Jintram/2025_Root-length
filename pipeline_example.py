
################################################################################
# %% Load libraries

import os

import functions_files.filelisting as ffl
    # import importlib; importlib.reload(ffl)
import functions_pipeline.analyze_plate as plap
    # import importlib; importlib.reload(plap)

################################################################################
# %% Gather the file list df.

# dataset spcecific config
DIR_INPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/202602/SEG/segfiles/'
DIR_OUTPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/202602/LEN/'

import functions_files.filelisting as gfl
    # import importlib; importlib.reload(gfl)

# Generate list of files to analyze
df_filelist, metadata_toseg_filepath = \
    gfl.gen_metadatafile_segfiles(
        directory_inputfiles=DIR_INPUTFILES,
        directory_outputfiles=DIR_OUTPUTFILES,
    )
    # directory_inputfiles = DIR_INPUTFILES; directory_outputfiles = DIR_OUTPUTFILES

# find plant with id 250502_OY_09
matching_idx = df_filelist.index[df_filelist["filename"].str.contains("250502_OY_09", na=False)]
print(matching_idx.tolist())

# Run the analysis
plap.analyze_all_plates(df_filelist=df_filelist, 
                        output_dir=DIR_OUTPUTFILES)

# Now make one big overview dataframe
plap.generate_df_all(df_filelist, DIR_OUTPUTFILES)





# %%
