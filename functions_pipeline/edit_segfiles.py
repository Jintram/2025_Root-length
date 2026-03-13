
"""
Edit segmentation files interactively using napari.

Opens each segmentation file (.npz) in a napari viewer, allowing the user
to correct labeled masks using the default napari label-editing tools.

Keybindings:
    q - quit the loop without saving the current file
    r - call a custom action on the mask at the current mouse position
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
from skimage.draw import line
from skimage.measure import label as sk_label
from scipy.ndimage import binary_dilation

################################################################################
# %% precomputed distance grid for nearest-background search

_SEARCH_RADIUS = 200
_yy, _xx = np.mgrid[-_SEARCH_RADIUS:_SEARCH_RADIUS + 1,
                    -_SEARCH_RADIUS:_SEARCH_RADIUS + 1]
_DIST_GRID = np.sqrt(_yy**2 + _xx**2).astype(np.float32)

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
# %% correct mask root/shoot line

def correct_mask_rootshootline(mask, row, col):
    """
    Draw a root/shoot boundary line (label=5) on the mask.

    Places a seed pixel at (row, col) with label 5, then finds the nearest
    background pixel (value == 0) to the left and right of the seed using
    a precomputed Euclidean distance grid. A line is drawn between those
    two background pixels using skimage.draw.line.

    Parameters
    ----------
    mask : np.ndarray
        2D labeled mask array (modified in-place).
    row : int
        Row position on the mask.
    col : int
        Column position on the mask.

    Returns
    -------
    mask : np.ndarray
        The modified mask.
    """
    n_rows, n_cols = mask.shape
    R = _SEARCH_RADIUS

    # Bounds check
    if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
        print("  Position out of bounds — skipping.")
        return mask

    # Determine crop region (clamped to image bounds)
    row_min = max(0, row - R)
    row_max = min(n_rows, row + R + 1)
    col_min = max(0, col - R)
    col_max = min(n_cols, col + R + 1)

    # Corresponding slice in the precomputed distance grid
    grid_row_min = R - (row - row_min)
    grid_row_max = R + (row_max - row)
    grid_col_min = R - (col - col_min)
    grid_col_max = R + (col_max - col)

    # Extract local region and matching distance grid
    region = mask[row_min:row_max, col_min:col_max]
    distances = _DIST_GRID[grid_row_min:grid_row_max,
                           grid_col_min:grid_col_max].copy()

    # Mask out foreground pixels (only labels 1, 2, and 5 are foreground)
    foreground = (region == 1) | (region == 2) | (region == 5)
    distances[foreground] = np.inf

    # Local seed position within the cropped region
    local_row = row - row_min
    local_col = col - col_min

    # --- Find nearest background pixel to the LEFT (col < local_col) ---
    left_distances = distances.copy()
    left_distances[:, local_col:] = np.inf   # exclude seed column and right
    if np.any(np.isfinite(left_distances)):
        left_idx = np.unravel_index(np.argmin(left_distances),
                                    left_distances.shape)
        left_point = (left_idx[0] + row_min, left_idx[1] + col_min)
    else:
        print("  No background found to the left — skipping.")
        return mask

    # --- Find nearest background pixel to the RIGHT (col > local_col) ---
    right_distances = distances.copy()
    right_distances[:, :local_col + 1] = np.inf  # exclude seed column and left
    if np.any(np.isfinite(right_distances)):
        right_idx = np.unravel_index(np.argmin(right_distances),
                                     right_distances.shape)
        right_point = (right_idx[0] + row_min, right_idx[1] + col_min)
    else:
        print("  No background found to the right — skipping.")
        return mask

    # Draw line between the two background pixels (label=5, foreground only)
    rr, cc = line(left_point[0], left_point[1],
                  right_point[0], right_point[1])
    fg = np.isin(mask[rr, cc], [1, 2, 5])
    mask[rr[fg], cc[fg]] = 5
    print(f"  Drew line from {left_point} to {right_point}")

    return mask


def correct_mask_throughline(mask, row, col):
    """
    Draw a boundary line (label=5) through the seed point on the mask.

    Finds the nearest background pixel (bg1) to the seed (row, col), then
    walks from the seed in the direction away from bg1 (i.e. bg1 → seed,
    continued) until a second background pixel (bg2) is found. A line is
    drawn from bg1 to bg2.

    If the seed is on background, the call is skipped.

    Parameters
    ----------
    mask : np.ndarray
        2D labeled mask array (modified in-place).
    row : int
        Row position on the mask.
    col : int
        Column position on the mask.

    Returns
    -------
    mask : np.ndarray
        The modified mask.
    """
    n_rows, n_cols = mask.shape
    R = _SEARCH_RADIUS

    # Bounds check
    if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
        print("  Position out of bounds — skipping.")
        return mask

    # Skip if the seed is already on background
    if mask[row, col] == 0:
        print("  Seed is on background — skipping.")
        return mask

    # --- Find bg1: nearest background pixel to the seed ---
    row_min = max(0, row - R)
    row_max = min(n_rows, row + R + 1)
    col_min = max(0, col - R)
    col_max = min(n_cols, col + R + 1)

    grid_row_min = R - (row - row_min)
    grid_row_max = R + (row_max - row)
    grid_col_min = R - (col - col_min)
    grid_col_max = R + (col_max - col)

    region = mask[row_min:row_max, col_min:col_max]
    distances = _DIST_GRID[grid_row_min:grid_row_max,
                           grid_col_min:grid_col_max].copy()
    foreground = (region == 1) | (region == 2) | (region == 5)
    distances[foreground] = np.inf

    if not np.any(np.isfinite(distances)):
        print("  No background found nearby — skipping.")
        return mask

    bg1_local = np.unravel_index(np.argmin(distances), distances.shape)
    bg1 = (bg1_local[0] + row_min, bg1_local[1] + col_min)

    # --- Compute direction from bg1 through seed ---
    dy = row - bg1[0]
    dx = col - bg1[1]
    length = np.sqrt(dy**2 + dx**2)
    if length == 0:
        print("  Seed coincides with bg1 — skipping.")
        return mask
    dy_unit = dy / length
    dx_unit = dx / length

    # --- Walk from seed in direction bg1→seed to find bg2 ---
    max_steps = 2 * R
    bg2 = None
    for step in range(1, max_steps + 1):
        r = int(round(row + dy_unit * step))
        c = int(round(col + dx_unit * step))
        if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
            break
        if mask[r, c] not in (1, 2, 5):
            bg2 = (r, c)
            break

    if bg2 is None:
        print("  No second background pixel found — skipping.")
        return mask

    # Draw line from bg1 to bg2 (label=5, foreground only)
    rr, cc = line(bg1[0], bg1[1], bg2[0], bg2[1])
    fg = np.isin(mask[rr, cc], [1, 2, 5])
    mask[rr[fg], cc[fg]] = 5
    print(f"  Drew through-line from {bg1} to {bg2}")

    return mask


################################################################################
# %% relabel regions based on root/shoot lines

# REMOVE THIS PART
# def _relabel_around_line(mask, line_mask):
#     """
#     Relabel plant regions touching a single red line as shoot (1) or root (2).

#     Identifies connected foreground components (labels 1 and 2) that are
#     adjacent to the given red line. Computes an overall center of mass (CoM)
#     of all touching pixels, and a per-region CoM. Regions whose CoM is above
#     (lower row) the overall CoM are assigned label 1 (shoot); regions below
#     (higher row) are assigned label 2 (root).

#     Parameters
#     ----------
#     mask : np.ndarray
#         2D labeled mask array (modified in-place).
#     line_mask : np.ndarray
#         Boolean mask of one red-line connected component.

#     Returns
#     -------
#     mask : np.ndarray
#         The modified mask.
#     """
#     # Binary foreground: only labels 1 and 2 (NOT 5 / red lines)
#     fg = (mask == 1) | (mask == 2)
#         # plt.imshow(fg)
#     # Label connected components of the foreground
#     cc_labels, n_components = sk_label(fg, return_num=True, connectivity=1)

#     # Dilate the line mask by 1 pixel to find adjacent components
#     line_dilated = binary_dilation(line_mask, iterations=1)

#     # Find which component IDs touch the dilated line
#     touching_ids = set(np.unique(cc_labels[line_dilated])) - {0}

#     if len(touching_ids) == 0:
#         print("    No plant regions touch this line — skipping.")
#         return mask

#     # Build a mask of all touching pixels
#     touching_mask = np.isin(cc_labels, list(touching_ids))

#     # Overall CoM (row coordinate) of all touching pixels
#     touching_rows = np.where(touching_mask)[0]
#     overall_com_row = touching_rows.mean()

#     # Per-region: compute CoM and assign shoot (1) or root (2)
#     for comp_id in touching_ids:
#         comp_mask = (cc_labels == comp_id)
#         comp_rows = np.where(comp_mask)[0]
#         comp_com_row = comp_rows.mean()

#         if comp_com_row < overall_com_row:
#             # Above overall CoM → shoot
#             mask[comp_mask] = 1
#         elif comp_com_row > overall_com_row:
#             # Below overall CoM → root
#             mask[comp_mask] = 2
#         # If equal, leave unchanged

#     # Convert the red line itself to root label (2)
#     mask[line_mask] = 2

#     return mask


def relabel_by_rootshootlines(mask):
    """
    Relabel plant regions based on red boundary lines (label=5).

    Finds plant blobs by treating labels 1, 2, and 5 as foreground. For each
    blob that contains red pixels (label=5):
      1. Compute overall center of mass (row) of the blob.
      2. Remove red pixels to split the blob into sub-regions.
      3. For each sub-region, assign shoot (1) if its CoM is above the overall
         CoM, or root (2) if below.
      4. Convert all red pixels in the blob to root (2).

    Parameters
    ----------
    mask : np.ndarray
        2D labeled mask array (modified in-place).

    Returns
    -------
    mask : np.ndarray
        The modified mask.
    """
    # Step 1: find plant blobs including red lines
    fg_all = (mask == 1) | (mask == 2) | (mask == 5)
    blob_labels, n_blobs = sk_label(fg_all, return_num=True, connectivity=1)

    if n_blobs == 0:
        print("  No plant blobs found — nothing to relabel.")
        return mask

    red_mask = (mask == 5)
    blobs_processed = 0

    for blob_id in range(1, n_blobs + 1):
        blob_mask = (blob_labels == blob_id)

        # Step 2: skip blobs without red pixels
        if not np.any(red_mask & blob_mask):
            continue

        blobs_processed += 1

        # Step 3.1: overall CoM of the entire blob
        blob_rows = np.where(blob_mask)[0]
        overall_com_row = blob_rows.mean()

        # Step 3.2: remove red lines within this blob to split into sub-regions
        fg_no_red = blob_mask & ~red_mask
        sub_labels, n_subs = sk_label(fg_no_red, return_num=True, connectivity=1)

        # Step 3.3: assign shoot/root per sub-region
        for sub_id in range(1, n_subs + 1):
            sub_mask = (sub_labels == sub_id)
            sub_rows = np.where(sub_mask)[0]
            sub_com_row = sub_rows.mean()

            if sub_com_row < overall_com_row:
                # Above overall CoM → shoot
                mask[sub_mask] = 1
            elif sub_com_row > overall_com_row:
                # Below overall CoM → root
                mask[sub_mask] = 2
            # If equal, leave unchanged

        # Step 3.4: convert red pixels to root (2) or shoot (1)
        # Red pixels touching any root pixel → root, otherwise → shoot
        root_dilated = binary_dilation(mask == 2, iterations=1)
        blob_red = blob_mask & red_mask
        mask[blob_red & root_dilated] = 2
        mask[blob_red & ~root_dilated] = 1

    if blobs_processed == 0:
        print("  No red lines found in any blob — nothing to relabel.")
    else:
        print(f"  Processed {blobs_processed} blob(s) with red lines.")
    print("  Relabeling complete.")

    return mask


################################################################################
# %% edit a single segfile

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
    segfile_data = np.load(segfile_path)
    img_mask = segfile_data['img_pred_lbls']
    # Preserve any extra arrays (e.g. prepr_info) for re-saving later
    extra_arrays = {k: segfile_data[k] for k in segfile_data.files
                    if k != 'img_pred_lbls'}
    
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
            if 'prepr_info' in extra_arrays:
                crop_rect = extra_arrays['prepr_info']
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
    
    # ----- keybinding: r = draw root/shoot boundary line -----------------------
    @viewer.bind_key('r')
    def _run_custom_action(viewer):
        """Draw a horizontal red line (label=5) at the root/shoot boundary."""
        # Get the current mouse position in world coordinates
        mouse_pos = viewer.cursor.position
        mouse_row = int(round(mouse_pos[-2]))
        mouse_col = int(round(mouse_pos[-1]))
        print(f"  'r' pressed at world position ({mouse_row}, {mouse_col})")

        # Subtract the image shift to get the correct label-layer position
        row = mouse_row - shift_widget.shift_y.value
        col = mouse_col - shift_widget.shift_x.value
        print(f"  Mapped to label position ({row}, {col})")

        # Call the standalone function
        mask = labels_layer.data
        mask = correct_mask_rootshootline(mask, row, col)

        # Refresh the labels layer
        labels_layer.data = mask
    
    # ----- keybinding: t = draw through-line ----------------------------------
    @viewer.bind_key('t')
    def _run_throughline_action(viewer):
        """Draw a line (label=5) through the seed from bg1 to bg2."""
        # Get the current mouse position in world coordinates
        mouse_pos = viewer.cursor.position
        mouse_row = int(round(mouse_pos[-2]))
        mouse_col = int(round(mouse_pos[-1]))
        print(f"  't' pressed at world position ({mouse_row}, {mouse_col})")

        # Subtract the image shift to get the correct label-layer position
        row = mouse_row - shift_widget.shift_y.value
        col = mouse_col - shift_widget.shift_x.value
        print(f"  Mapped to label position ({row}, {col})")

        # Call the standalone function
        mask = labels_layer.data
        mask = correct_mask_throughline(mask, row, col)

        # Refresh the labels layer
        labels_layer.data = mask
    
    # ----- keybinding: u = relabel by root/shoot lines ------------------------
    @viewer.bind_key('u')
    def _run_relabel_action(viewer):
        """Relabel plant regions as shoot/root based on red lines."""
        print("  'u' pressed — relabeling by root/shoot lines.")
        mask = labels_layer.data
        mask = relabel_by_rootshootlines(mask)
        labels_layer.data = mask
    
    # Run napari (blocks until the viewer is closed)
    napari.run()
    
    # ----- after viewer is closed ---------------------------------------------
    if save_on_close[0]:
        # Create backup before first save
        _backup_if_needed(segfile_path)
        
        # Save the (possibly edited) mask back to the .npz file
        edited_mask = labels_layer.data
        np.savez_compressed(segfile_path, img_pred_lbls=edited_mask,
                            **extra_arrays)
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
