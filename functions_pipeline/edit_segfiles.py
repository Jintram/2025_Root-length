
"""
Edit segmentation files interactively using napari.

Opens each segmentation file (.npz) in a napari viewer, allowing the user
to correct labeled masks using the default napari label-editing tools.

Keybindings:
    q - quit the loop without saving the current file
    r - call a custom action on the mask at the current mouse position
"""

"""
NOTES (REMOVE THIS)

I want a three step procedure

(1)
clicking the "r" button, a red pixel (label = 5) is placed on the labeled layer,
with displacement equal to the displacement of the original image. 
then, the labeled mask is updated, it finds the closest background pixel to the 
left of this red pixel, and the closest background pixel to the right of this
red pixel; then a line is drawn on the labeled mask between those two background pixels,
also in the color red.

(2)
<to be determined>

(3)
<to be determined>

"""






################################################################################
# %% libraries

import os
import shutil
import glob

import numpy as np
import napari
from magicgui import magicgui

import functions_files.filelisting as ffl
    # import importlib; importlib.reload(ffl)
import custom_functions.custom_mask_action as cfca
    # import importlib; importlib.reload(cfca)
import functions_pipeline.utils as plutils
    # import importlib; importlib.reload(plutils)

import skimage.io as skio

################################################################################
# %% helper: backup

def _backup_if_needed(filepath):
    """
    Create a backup of the file if it hasn't been backed up before.
    
    The backup is saved as <filepath>.bak. If the backup already exists,
    this function does nothing (ie only the very first version is kept).
    """
    
    backup_path = filepath + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"  Backup created: {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")


################################################################################
# %% edit a single segfile

def correct_mask_rootshootline():
    """
    Update mask according to manual correction lines.
    
    Given a plant mask with additional drawn lines, that demarcate the boundary
    between root and shoot, the 
    """

def edit_segfile_single(curr_file, dir_imagefiles=None):
    """
    Open a single segmentation file in napari for interactive editing.
    
    The user can edit the labeled mask using default napari tools.
    Keybindings:
        q - close without saving, signal to quit the loop
        r - call custom_mask_action at the current mouse position
    
    Input parameters:
    - curr_file: a fileinfo object (from functions_files.filelisting) pointing
      to the .npz segmentation file.
    
    Returns:
    - quit_requested: bool, True if the user pressed "q" to quit.
    """
    
    # Load labeled mask from the .npz file
    segfile_path = curr_file.fullpath
    img_mask = np.load(segfile_path)['img_pred_lbls']
    
    # Optionally, load the matching image file based on dir_imagefiles
    img_original = None
    if not dir_imagefiles is None:
        # Generate the image path (assuming naming convention adhered)
        imagefile_path_noext = os.path.join(
            dir_imagefiles, curr_file.subdir, 
            curr_file.filename.replace('_seg.npz', '').replace('_seg.npy', '')
        )
        # Look for file with any extension
        file_hits = glob.glob(imagefile_path_noext+".*")
        # If unique hit found, load
        if len(file_hits) == 1:
            img_original = skio.imread(file_hits[0])
            # Check if crop info is available, if so, load   
            print("Checking for cropping info..")
            if 'prepr_info' in np.load(segfile_path):
                crop_rect = np.load(segfile_path)['prepr_info']
                minr, maxr, minc, maxc = crop_rect
                img_original = img_original[minr:maxr, minc:maxc]
                # plt.imshow(img_original)
                print("Found")
            else:
                print("No cropping info found..")
    
    # State flags (use list to allow mutation inside nested functions)
    quit_requested = [False]
    save_on_close = [True]
    
    # Disable IPython event loop integration so that napari.run() blocks
    # (otherwise it returns immediately in VS Code interactive windows)
    try:
        napari.settings.get_settings().application.ipy_interactive = False
    except Exception as e:
        print(f"  Warning: Could not disable IPython event loop integration: {e}")
    
    # Open napari viewer
    viewer = napari.Viewer(title=f"Editing: {curr_file.filename}")
    # Add original image if available
    img_layer = None
    if not img_original is None:
        img_layer = viewer.add_image(img_original, name='original image')
    # Add labels 
    labels_layer = viewer.add_labels(img_mask, name='segmentation',
                                     colormap=plutils.custom_colors_plantclasses)

    # ----- widget: shift image layer visually ---------------------------------
    @magicgui(
        shift_x={"widget_type": "Slider", "min": -200, "max": 200, "value": 0},
        shift_y={"widget_type": "Slider", "min": -200, "max": 200, "value": 0},
        auto_call=True,
    )
    def shift_widget(shift_x: int = 0, shift_y: int = 0):
        if img_layer is not None:
            img_layer.translate = (shift_y, shift_x)

    viewer.window.add_dock_widget(shift_widget, name="Shift Image")

    # ----- keybinding: q = quit without saving --------------------------------
    @viewer.bind_key('q')
    def _quit_without_saving(viewer):
        """Close viewer without saving, and signal to quit the loop."""
        print("  'q' pressed — closing without saving, quitting loop.")
        quit_requested[0] = True
        save_on_close[0] = False
        viewer.close()
    
    # ----- keybinding: r = custom action --------------------------------------
    @viewer.bind_key('r')
    def _run_custom_action(viewer):
        """Call the custom placeholder function at the mouse position."""
        # Get the current mouse position in data coordinates
        # (napari stores this as the last cursor position on the canvas)
        mouse_pos = viewer.cursor.position
        # Convert to integer row, col
        row = int(round(mouse_pos[-2]))
        col = int(round(mouse_pos[-1]))
        print(f"  'r' pressed at position ({row}, {col})")
        
        # Call the custom action; it may modify the mask
        current_data = labels_layer.data
        modified_data = cfca.custom_mask_action(current_data, (row, col))
        
        # Update the labels layer with the (potentially modified) mask
        labels_layer.data = modified_data
    
    # Run napari (blocks until the viewer is closed)
    napari.run()
    
    # ----- after viewer is closed ---------------------------------------------
    if save_on_close[0]:
        # Create backup before first save
        _backup_if_needed(segfile_path)
        
        # Save the (possibly edited) mask back to the .npz file
        edited_mask = labels_layer.data
        np.savez_compressed(segfile_path, img_pred_lbls=edited_mask)
        print(f"  Saved: {segfile_path}")
    else:
        print(f"  Not saved: {segfile_path}")
    
    return quit_requested[0]


################################################################################
# %% edit all segfiles

def edit_all_segfiles(df_filelist, dir_inputfiles, dir_imagefiles=None):
    """
    Loop over all segmentation files and open each in napari for editing.
    
    This iterates over the file list dataframe (same format as used by 
    analyze_all_plates) and opens each .npz file for interactive editing.
    
    The loop can be terminated early by pressing "q" in the napari viewer.
    
    Input parameters:
    - df_filelist: pd.DataFrame with columns 'basedir', 'subdir', 'filename'.
    - dir_inputfiles: str, the base directory for the segmentation files
      (used as the outputdir placeholder in fileinfo, since we save in-place).
    """
    
    for file_idx in range(len(df_filelist)):
        # file_idx = 200
        
        # Get file info
        basedir, subdir, filename = \
            df_filelist.loc[file_idx, ['basedir', 'subdir', 'filename']]
        curr_file = ffl.fileinfo(basedir, subdir, filename, dir_inputfiles)
        
        print("========================================================")
        print(f"Editing file {file_idx+1}/{len(df_filelist)}: {curr_file.fullpath}")
        
        # Open for editing
        quit_requested = edit_segfile_single(curr_file, dir_imagefiles)
        
        if quit_requested:
            print("Loop terminated by user (pressed 'q').")
            break
    
    print("========================================================")
    print("Done editing segmentation files.")
