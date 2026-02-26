
"""
Analyze all plants in a segfile, save output to plots and .tsv files.

To switch between modes of displaying plots:
%matplotlib qt
%matplotlib inline
"""


################################################################################
# %% libraries

import os
import time 

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# load custom scripts
import functions_pipeline.utils as plutils
    # import importlib; importlib.reload(plutils)
import functions_pipeline.preprocessing as plprep
    # import importlib; importlib.reload(plprep)
import functions_pipeline.determine_length as pllen
    # import importlib; importlib.reload(pllen)
import functions_files.filelisting as ffl
    # import importlib; importlib.reload(ffl)
    

################################################################################
# %% Load data for 1 sample

def identify_plants(img_mask_clean):
        
    # Find the separate regions of interest corresponding to individual plants
    list_img_indivplants, img_mask_bbox_lblcount, img_mask_rprops = \
        plprep.find_individual_plants(img_mask_clean)
    
    # Now determine the selection of QC passed plants
    sel_plants = \
        (
            # contains precisely one root
            (img_mask_bbox_lblcount[:,1] ==1) & 
            # and at least one other category
            (img_mask_bbox_lblcount[:,[0,2,3]].any(axis=1))
        )
        
    # Highlight
    # selected_bboxes = np.array(img_mask_rprops)[sel_plants]
    
    
    return list_img_indivplants, sel_plants, np.array(img_mask_rprops)

# %%

def analyze_plate(curr_file):
    # segfile_path = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/SEGMENTATION/202602/segfiles/20250520/20250520_OY03.npz"

    # Load input
    img_mask = \
        np.load(curr_file.fullpath)['img_pred_lbls']

    # Clean the mask
    img_mask_clean = plprep.clean_mask(img_mask)
    
    # Find the plants
    list_img_indivplants, sel_plants, img_mask_rprops = \
        identify_plants(img_mask_clean)
    
    # Plot overview
    # %matplotlib qt
    # plprep.plot_mask_and_bboxes(labeled_mask=img_mask_clean,the_rprops=img_mask_rprops[sel_plants],curr_file=curr_file)

    # Build RootSample objects for each QC-passed plant
    current_sample_all_plants = []
    for idx, plant_mask in enumerate(list_img_indivplants[sel_plants]):
        
        root_mask = plant_mask == 2
        shoot_mask = plant_mask == 1
        original_bbox = \
            np.array(img_mask_rprops)[sel_plants][idx].bbox
        
        current_sample_all_plants.append(
            pllen.RootSample(
                root_mask=root_mask,
                shoot_mask=shoot_mask,
                plant_mask=plant_mask,
                bbox=original_bbox
            )
        )
        
    # Now process all plants
    for i, sample in enumerate(current_sample_all_plants):
        # i = 14; sample=current_sample_all_plants[i]
        
        print(f"Currently processing plant {i+1} of {len(current_sample_all_plants)}")
        
        # Perform calculations based on this plant mask
        # and store the result in the same "plant container"
        current_sample_all_plants[i] = pllen.run_default_length_pipeline(sample)
        
        if current_sample_all_plants[i] is None:
            print("^ FAILED")

    # preparing output
    output_dir_plot = os.path.join(curr_file.outputdir, 'lenplots/', curr_file.subdir)
    os.makedirs(output_dir_plot, exist_ok=True)
    output_dir_data = os.path.join(curr_file.outputdir, 'data/', curr_file.subdir)
    os.makedirs(output_dir_data, exist_ok=True)
        
    # Now plot the result
    fig, ax = \
        pllen.plot_all_plants_projected(
                sample_image = img_mask,
                plant_results = current_sample_all_plants)
    # and save
    fig.savefig(os.path.join(
        output_dir_plot, curr_file.filebasename + "_all-plants-projected.pdf"))
    plt.close(fig)
    
    # moreover, save the data to a .csv file
    # (cols: plant index and "root length")
    df_out = pd.DataFrame({
        'plant_index': np.arange(len(current_sample_all_plants)),
        'length_pixels': [plant.length_pixels for plant in current_sample_all_plants],
        'length_mm': [plant.length_mm for plant in current_sample_all_plants]
    })    
    output_filename = curr_file.filebasename + "_lengths.tsv"
    df_out.to_csv(
        os.path.join(output_dir_data, output_filename), 
        index=False, sep="\t"
        )
        
# %% runner

def analyze_all_plates(df_filelist, output_dir):

    time_taken = []
    for file_idx in range(len(df_filelist)):
        # file_idx = 37
        
        # Get current time
        start_time = time.time()
        
        # get file info
        basedir, subdir, filename = \
            df_filelist.loc[file_idx,['basedir', 'subdir', 'filename']]
        curr_file = ffl.fileinfo(basedir, subdir, filename, output_dir)
        
        print("========================================================")
        print(f"Processing file {file_idx+1}/{len(df_filelist)}: {curr_file.fullpath}")
        
        # Analyze the plate
        analyze_plate(curr_file)
        
        # record end time
        end_time = time.time(); time_taken.append(end_time - start_time)
        print("Time taken for this file: {:.2f} seconds".format(end_time - start_time),
              "\nAverage time: {:.2f} seconds".format(np.mean(time_taken)))
        
def generate_df_all(df_filelist, datadir):
    """Loop over all files, load dataframes, merge them""""
    
    list_dfs = []
    for file_idx in range(len(df_filelist)):
        # file_idx = 0
        
        # get current file info
        basedir, subdir, filename = \
            df_filelist.loc[file_idx, ['basedir', 'subdir', 'filename']]
        curr_file = ffl.fileinfo(basedir, subdir, filename, datadir)
        # specific info to load
        filename_data = curr_file.filebasename + "_lengths.tsv"
        filepath = os.path.join(
            curr_file.outputdir, 'data/', curr_file.subdir, filename_data)
        
        # load it
        df = pd.read_csv(filepath, sep="\t")
        df.insert(0, "sample_identifier", curr_file.filebasename)
        df.insert(0, "sample_index", file_idx)
        list_dfs.append(df)

    df_all = pd.concat(list_dfs, ignore_index=True)
    
    # save df_all to main outputdir
    df_all.to_excel(os.path.join(datadir, "all_samples_lengths.xlsx"))
    
    return df_all
    
    
    