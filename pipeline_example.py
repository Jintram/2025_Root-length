
################################################################################
# %% Load libraries

import os

################################################################################
# %% Gather the file list df.

# dataset spcecific config
DIR_INPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/202602/SEG/segfiles/'
DIR_OUTPUTFILES = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/202602/LEN/'

import functions_files.filelisting as gfl
    # import importlib; importlib.reload(gfl)

df_filelist, metadata_toseg_filepath = \
    gfl.gen_metadatafile_segfiles(
        directory_inputfiles=DIR_INPUTFILES,
        directory_outputfiles=DIR_OUTPUTFILES,
    )
    # directory_inputfiles = DIR_INPUTFILES; directory_outputfiles = DIR_OUTPUTFILES

FILE_IDX = 1
basedir, subdir, filename = df_filelist.loc[FILE_IDX,['basedir', 'subdir', 'filename']]
segfile_path1 = os.path.join(basedir, subdir, filename)
segfile_path = segfile_path1





