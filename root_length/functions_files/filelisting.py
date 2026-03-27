
################################################################################
# %% 

import glob
import os
import pandas as pd
import re

from dataclasses import dataclass

################################################################################
# %% 

@dataclass
class fileinfo:
        basedir: str
        subdir:  str
        filename: str
        outputdir: str | None = None
        @property
        def fullpath(self) -> str:
            return os.path.join(self.basedir, self.subdir, self.filename)
        @property
        def filebasename(self) -> str:
            return self.filename.replace("_seg.npz","").replace("_seg.npy","")

def gen_metadatafile_segfiles(
        directory_inputfiles,
        directory_outputfiles,
        file_formats=['.npz']):
    """
    Generate dataframe with list of files to analyze.
    
    Given a folder, generate an excel file with column 'filename' with all image files
    from a given directory and its subdirectories.
    """
    
    # make output directory
    os.makedirs(directory_outputfiles, exist_ok=True)
    
    # find all image files
    all_paths = glob.glob(os.path.join(directory_inputfiles, '**', f'*[{"|".join(file_formats)}]'))
    
    # Now get the subdirs and filenames
    all_subdirs = [
        os.path.relpath(os.path.dirname(current_path), directory_inputfiles)
            for current_path in all_paths
        ]
    all_filenames = [
        os.path.basename(current_path) 
            for current_path in all_paths
        ]
                        
    # Now convert this to a table, with one row per file
    nr_of_rows = len(all_filenames)
    df_filelist = pd.DataFrame({
            'basedir': nr_of_rows * [directory_inputfiles],
            'subdir': all_subdirs,
            'filename': all_filenames
            })
        
    # now save the metadata file
    metadata_toseg_filepath = os.path.join(directory_outputfiles, 'metadata_segfiles_toanalyze_autogen.xlsx')
    df_filelist.to_excel(metadata_toseg_filepath, index=False)
    
    return df_filelist, metadata_toseg_filepath

# %%
