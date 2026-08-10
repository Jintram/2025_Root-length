
"""
Edit segmentation files interactively using napari.

Opens each segmentation file (.npz) in a napari viewer, allowing the user
to correct labeled masks using the default napari label-editing tools.

Keybindings:
    q - quit the loop without saving the current file
    n - go to the next file without saving the current file
    r - call a custom action on the mask at the current mouse position
"""




################################################################################
# %% libraries

import os
import shutil
import glob
from dataclasses import dataclass

import numpy as np
import napari
from magicgui import magicgui
from magicgui.widgets import Container

import root_length.functions_files.filelisting as ffl
    # import importlib; importlib.reload(ffl)
import root_length.custom_functions.custom_mask_action as cfca
    # import importlib; importlib.reload(cfca)
import root_length.functions_pipeline.utils as plutils
    # import importlib; importlib.reload(plutils)
import root_length.functions_pipeline.preprocessing_seg as plprep_seg
    # import importlib; importlib.reload(plprep_seg)
import root_length.functions_pipeline.preprocessing as plprep
    # import importlib; importlib.reload(plprep)

import skimage.io as skio
from skimage.draw import line
from skimage.measure import label as sk_label
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_dilation

################################################################################
# %% precomputed distance grid for nearest-background search

_SEARCH_RADIUS = 200
_yy, _xx = np.mgrid[-_SEARCH_RADIUS:_SEARCH_RADIUS + 1,
                    -_SEARCH_RADIUS:_SEARCH_RADIUS + 1]
_DIST_GRID = np.sqrt(_yy**2 + _xx**2).astype(np.float32)

################################################################################
# %% session state preserved across files

@dataclass
class EditorSessionState:
    """Settings preserved across files during an editing session."""
    # image translation (from the shift widget)
    shift_x: int = 0
    shift_y: int = 0
    # size thresholds (from the remove-small/large widgets)
    remove_small_min_size: int = 100
    remove_large_max_size: int = 10000
    # labels-layer interaction state
    active_label: int = 1
    brush_size: int = 10

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


def _save_segfile(curr_file, mask, extras_update=None):
    """
    Save `mask` into `curr_file`'s .npz, preserving any other arrays on disk.

    Re-reads the file to grab the current set of extra arrays (e.g. prepr_info),
    so any external changes are not clobbered. A one-time backup is made before
    the first overwrite via `_backup_if_needed`.

    `extras_update` is an optional dict of extra arrays to add/overwrite
    (used to write back a `mask_rect` edited in napari).
    """
    segfile_path = curr_file.fullpath
    segfile_data = np.load(segfile_path, allow_pickle=True)
    extra_arrays = {k: segfile_data[k] for k in segfile_data.files
                    if k != 'img_pred_lbls'}
    if extras_update is not None:
        extra_arrays.update(extras_update)
    _backup_if_needed(segfile_path)
    np.savez_compressed(segfile_path, img_pred_lbls=mask, **extra_arrays)
    print(f"  Saved: {segfile_path}")


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
# %% helper: remove small connected regions from labeled mask

def remove_small_foregroundregions(mask, min_size):
    """Remove small connected foreground regions while preserving original labels."""
    cc = sk_label(mask > 0)
    cc_filtered = remove_small_objects(cc, min_size=min_size)
    return np.where(cc_filtered > 0, mask, 0)


def remove_large_foregroundregions(mask, min_size):
    """Remove large connected foreground regions while preserving original labels."""
    cc = sk_label(mask > 0)
    large_only = remove_small_objects(cc, min_size=min_size - 1)
    keep = cc ^ large_only # keep what is not large
    return np.where(keep, mask, 0)


################################################################################
# %% open napari for annotation editing

def edit_annotation_napari(image, segmentation, mylabelcolormap=None,
                           title="Editing segmentation (for roots)",
                           session_state=None, curr_file=None,
                           mask_rect=None):
    '''
    Opens napari with `image` as background and `segmentation` as an editable
    label layer, optionally styled with `mylabelcolormap`.

    `segmentation` is always the complete label image. `mask_rect`, if given
    as (r0, r1, c0, c1), is the plate area: only that part is shown, which is
    what `analyze_plate` will end up measuring. The rest is not thrown away —
    the labels handed back are the complete ones again, with only the shown
    part replaced by the edited version. So the rectangle can be widened later
    and the labels outside it are still there.

    Returns `(seg_data, mask_rect, return_requests)`:
        - seg_data: the complete edited labels, or None if the user chose not
          to save.
        - mask_rect: the plate area as the user left it, or None (both when
          they deleted the rectangle and when nothing is to be saved). Note
          this is the *new* rect; the labels were merged using the old one.
        - return_requests: a dict of control-flow signals for the caller:
            - 'quitloop_flag' (bool): user pressed 'q' to quit the outer loop.
            - 'go_to' (int or None): user asked to jump to this 1-based sample
              number, either by pressing 'j' (without saving) or via the
              "Save + reload plate" button (with saving).
          Callers should treat unknown keys gracefully; new keys may be added.

    If `session_state` (an `EditorSessionState`) is provided, widget and layer
    settings (shift, size thresholds, active label, brush size) are restored
    from it on open and written back to it on close, so they persist across
    files.

    If `curr_file` is provided (a fileinfo-like object with `.fullpath`), a
    "Save now" button is added to the tools panel; clicking it writes the
    current labels back to the .npz via `_save_segfile`, and the rectangle
    becomes editable. If `curr_file` is None there is nowhere to store a rect,
    so no rectangle layer and no save button are shown, and 'w' is a no-op.

    To see the effect of a moved rectangle, use the "Save + reload plate"
    button: it saves and reopens the file, which redoes the filtering with the
    new rect. It is a jump to the current sample, so it needs a caller that
    acts on 'go_to' and a `curr_file.file_idx` (ie `edit_all_segfiles`).

    # Note to self, already used by Napari:
    1, 2, 3, 4, 5, 6, 7, 8, 9, 0, [, ], -, =, a, d, e, f, i, l, m, p, s, v, z
    # Used by this function: q, n, r, t, u, w, j
    '''

    if session_state is None:
        session_state = EditorSessionState()

    # The rect as it is on opening. Kept apart from whatever the user drags it
    # to later (`_rect_from_layer()`), because this is the one that determines
    # which part of the labels was shown, and hence which part may be replaced
    # when saving.
    rect_at_load = plprep.normalize_rect(mask_rect, segmentation.shape)

    # Explain options to user
    print("____\nStarting Napari editor window\n===\nq=quit without saving,"+
          "\nn=next file without saving,"+
          "\nr=draw root/shoot line, \nt=draw through-line, \nu=relabel by lines,"+
          "\nw=save current labels (if curr_file given),"+
          "\nj=jump to sample number (handled by caller; closes without saving)"+
          "\n____")

    # Signals returned to the caller. Mutated by key handlers below.
    return_requests = {'quitloop_flag': False, 'go_to': None}
    save_on_close = [True]
    
    # Disable IPython event loop integration so that napari.run() blocks
    # (otherwise it returns immediately in VS Code interactive windows)
    try:
        napari.settings.get_settings().application.ipy_interactive = False
        print("Disabled ipy_interactive to ensure Napari blocks further execution until user action..")
    except Exception as e:
        pass
    
    # Disable IPython GUI event loop (e.g. "%gui qt" in Spyder) so that
    # napari.run() blocks instead of returning immediately.
    # The previous setting is restored after napari.run() returns.
    _prev_gui = None
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            _prev_gui = ipython.active_eventloop  # e.g. "qt5" or None
            ipython.run_line_magic('gui', '')
        print("Disabled '%gui "+_prev_gui+"' magic to ensure Napari blocks further execution until user action..")
    except Exception:
        pass
    
    # Open napari viewer
    viewer = napari.Viewer(title=title)
    # Add original image if available
    img_layer = None
    if image is not None:
        img_layer = viewer.add_image(image, name='original image')
    # Add labels
    labels_kwargs = dict(name='segmentation')
    if mylabelcolormap is not None:
        labels_kwargs['colormap'] = mylabelcolormap
    # Only the plate area is shown, so that this is what analyze_plate sees too
    labels_layer = viewer.add_labels(
        plprep.apply_mask_rect(segmentation, rect_at_load), **labels_kwargs)

    # ----- overlay: editable mask_rect rectangle ------------------------------
    # Also added when there is no rect yet, so a plate area can be drawn from
    # scratch; but not when there is no file to store one in. Note it shares
    # the coordinate frame of the labels layer, not of the (shift-widget
    # translated) image layer.
    rect_layer = None
    if curr_file is not None:
        if rect_at_load is not None:
            r0, r1, c0, c1 = rect_at_load
            rect_data = [np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])]
        else:
            rect_data = []
        rect_layer = viewer.add_shapes(
            rect_data, shape_type='rectangle',
            edge_color='red', face_color='transparent',
            edge_width=4, name='mask_rect',
        )
        rect_layer.editable = True

    def _rect_from_layer():
        """
        Plate area as the user left it, or None if there is no rectangle.

        Read as a bounding box rather than by corner index, since dragging or
        rotating a napari rectangle does not preserve the order the corners
        were created in. None (a deleted rectangle) means "leave the stored
        one alone" — deleting the shape is more likely a slip than a request
        to drop the plate area.
        """
        if (rect_layer is None) or (len(rect_layer.data) == 0):
            print(f"  Warning: no shapes in the mask_rect layer, not updating mask.")
            return None
        if len(rect_layer.data) > 1:
            print(f"  Warning: {len(rect_layer.data)} shapes in the mask_rect "
                  "layer, using the first one.")
        corners = np.asarray(rect_layer.data[0])
        rows, cols = corners[:, -2], corners[:, -1]
        return plprep.normalize_rect(
            (rows.min(), rows.max(), cols.min(), cols.max()), segmentation.shape)

    def _labels_for_save():
        """The complete labels again, with the edited plate area pasted in."""
        return plprep.paste_inside_rect(labels_layer.data, segmentation,
                                        rect_at_load)

    def _save_now():
        """Write the labels and the plate rect back to the .npz (needs curr_file)."""
        rect = _rect_from_layer()
        _save_segfile(
            curr_file, _labels_for_save(),
            extras_update=(None if rect is None else
                           {'mask_rect': np.asarray(rect, dtype=np.int64)}))

    # ----- widget: shift image layer visually ---------------------------------
    @magicgui(
        shift_x={"widget_type": "Slider", "min": -200, "max": 200, "value": 0},
        shift_y={"widget_type": "Slider", "min": -200, "max": 200, "value": 0},
        auto_call=True,
    )
    def shift_widget(shift_x: int = 0, shift_y: int = 0):
        if img_layer is not None:
            img_layer.translate = (shift_y, shift_x)
    
    # (added to viewer below, using container)
    
    # ----- widget: remove small labeled regions --------------------------------
    @magicgui(
        min_size={"widget_type": "SpinBox", "min": 1, "max": 100000, "value": 100,
                  "label": "Min area (px)"},
        call_button="Remove smaller",
    )
    def remove_small_widget(min_size: int = 100):
        mask = labels_layer.data
        labels_layer.data = remove_small_foregroundregions(mask, min_size=min_size)
        print(f"  Removed regions smaller than {min_size} px.")
    
    # (added to viewer below, using container)
    
    # ----- widget: remove large labeled regions --------------------------------
    @magicgui(
        min_size={"widget_type": "SpinBox", "min": 1, "max": 1000000, "value": 10000,
                  "label": "Max area (px)"},
        call_button="Remove larger",
    )
    def remove_large_widget(min_size: int = 10000):
        mask = labels_layer.data
        labels_layer.data = remove_large_foregroundregions(mask, min_size=min_size)
        print(f"  Removed regions larger than {min_size} px.")
    
    # (added to viewer below, using container)

    # ----- widget: save current labels to disk --------------------------------
    # Saving from within the viewer is only possible when curr_file was given;
    # otherwise the button is omitted and the caller saves what we return.
    tool_widgets = [shift_widget, remove_small_widget, remove_large_widget]
    if curr_file is not None:
        @magicgui(call_button="Save now")
        def save_widget():
            _save_now()
        tool_widgets.append(save_widget)

    # ----- widget: save and reopen the same plate -----------------------------
    # After dragging the rectangle, this is how you see what it did: saving and
    # reopening re-runs the load-time filtering with the new rect. Implemented
    # as a jump to the current sample, so it only works via the file loop that
    # handles 'go_to' (ie edit_all_segfiles, which sets file_idx).
    if (curr_file is not None) and (getattr(curr_file, 'file_idx', None) is not None):
        @magicgui(call_button="Save + reload plate")
        def reload_widget():
            print("  Reload pressed — saving and reopening this plate.")
            return_requests['go_to'] = curr_file.file_idx + 1
            viewer.close()
        tool_widgets.append(reload_widget)

    # ----- dock widgets as a single stacked panel ------------------------------
    tools_container = Container(widgets=tool_widgets)
    viewer.window.add_dock_widget(tools_container, name="Tools")

    # ----- restore persisted session state ------------------------------------
    shift_widget.shift_x.value = session_state.shift_x
    shift_widget.shift_y.value = session_state.shift_y
    if img_layer is not None:
        img_layer.translate = (session_state.shift_y, session_state.shift_x)
    remove_small_widget.min_size.value = session_state.remove_small_min_size
    remove_large_widget.min_size.value = session_state.remove_large_max_size
    labels_layer.selected_label = session_state.active_label
    labels_layer.brush_size = session_state.brush_size
    viewer.layers.selection.active = labels_layer

    # ----- snapshot widget/layer state before Qt tears it down ----------------
    # napari.run() returns *after* the Qt window is destroyed, at which point
    # the underlying QSlider/QSpinBox objects are deleted and reading
    # `magicgui_widget.value` raises "wrapped C/C++ object ... has been deleted".
    # We therefore capture all values into a plain dict inside the Qt window's
    # closeEvent (which fires before destruction) and copy them into
    # session_state after napari.run() returns.
    state_snapshot = {}

    def _snapshot_state():
        try:
            state_snapshot['shift_x'] = int(shift_widget.shift_x.value)
            state_snapshot['shift_y'] = int(shift_widget.shift_y.value)
            state_snapshot['remove_small_min_size'] = int(remove_small_widget.min_size.value)
            state_snapshot['remove_large_max_size'] = int(remove_large_widget.min_size.value)
            state_snapshot['active_label'] = int(labels_layer.selected_label)
            state_snapshot['brush_size'] = int(labels_layer.brush_size)
        except Exception as e:
            print(f"  Warning: could not snapshot session state: {e}")

    qt_window = viewer.window._qt_window
    _original_close_event = qt_window.closeEvent

    def _close_event_with_snapshot(event):
        _snapshot_state()
        _original_close_event(event)

    qt_window.closeEvent = _close_event_with_snapshot

    # ----- keybinding: q = quit without saving --------------------------------
    @viewer.bind_key('q')
    def _quit_without_saving(viewer):
        """Close viewer without saving, and signal to quit the loop."""
        print("  'q' pressed — closing without saving, quitting loop.")
        return_requests['quitloop_flag'] = True
        save_on_close[0] = False
        viewer.close()
    
    # ----- keybinding: n = next file without saving ---------------------------
    @viewer.bind_key('n')
    def _next_without_saving(viewer):
        """Close viewer without saving, continue to the next file."""
        print("  'n' pressed — closing without saving, moving to next file.")
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

    # ----- keybinding: w = save current labels to disk ------------------------
    @viewer.bind_key('w')
    def _run_save_action(viewer):
        """Save the current labels layer to the .npz file (if curr_file given)."""
        if curr_file is None:
            print("  'w' pressed — Save function NOT available now.")
            return
        print("  'w' pressed — saving current labels.")
        _save_now()

    # ----- keybinding: j = jump to a specific sample --------------------------
    @viewer.bind_key('j')
    def _run_jump_action(viewer):
        """Ask for a sample number and signal the caller to jump there."""
        from qtpy.QtWidgets import QInputDialog
        num, ok = QInputDialog.getInt(
            qt_window, "Jump to sample",
            "Sample number (1-based):", 1, 1, 999999, 1)
        if not ok:
            print("  'j' pressed — jump cancelled.")
            return
        print(f"  'j' pressed — requesting jump to sample {num} (without saving).")
        return_requests['go_to'] = num
        save_on_close[0] = False
        viewer.close()

    # Run napari (blocks until the viewer is closed)
    napari.run()

    # ----- persist snapshotted state back to session_state --------------------
    # `state_snapshot` was populated by the closeEvent hook above, while the
    # Qt widgets were still alive.
    if state_snapshot:
        session_state.shift_x = state_snapshot['shift_x']
        session_state.shift_y = state_snapshot['shift_y']
        session_state.remove_small_min_size = state_snapshot['remove_small_min_size']
        session_state.remove_large_max_size = state_snapshot['remove_large_max_size']
        session_state.active_label = state_snapshot['active_label']
        session_state.brush_size = state_snapshot['brush_size']

    # Restore IPython GUI event loop if it was active before
    if _prev_gui is not None:
        try:
            ipython.run_line_magic('gui', _prev_gui)
            print("Re-establishing Spyder %gui setting ("+_prev_gui+")")
        except Exception:
            pass
    
    # ----- after viewer is closed ---------------------------------------------
    # (layers are plain python/numpy objects, so unlike the Qt widgets above
    # they can still be read now that the window is gone)
    if save_on_close[0]:
        return _labels_for_save(), _rect_from_layer(), return_requests
    else:
        return None, None, return_requests


################################################################################
# %% adapter for cheeky_cells

def edit_annotation_napari_cheekycells(image, segmentation, mylabelcolormap=None,
                                       title="Editing segmentation"):
    """
    `edit_annotation_napari` in the shape cheeky_cells expects it.

    cheeky_cells (annotation_aided.annotate_tiles) calls its
    `mynaparifunction` with 3 positional arguments and unpacks 2 return
    values, which is what this repo's editor looked like before it grew the
    plate rectangle. Rather than freeze the editor into that shape, this
    adapter absorbs the difference: assign *this* to `config1.my_napari_function`.

    There is no plate rectangle when annotating training tiles, and
    cheeky_cells saves the returned array itself, so `mask_rect` and
    `curr_file` are left out and the rect is dropped from the return.

    Delete this once cheeky_cells is updated to the current signature.
    (`return_requests` is passed back as-is; cheeky_cells already unwraps
    'quitloop_flag' from a dict in that position.)
    """
    seg_data, _, return_requests = edit_annotation_napari(
        image, segmentation, mylabelcolormap=mylabelcolormap, title=title)

    return seg_data, return_requests


################################################################################
# %% edit a single segfile

def edit_segfile_single(curr_file, dir_imagefiles=None, session_state=None):
    """
    Open a single segmentation file in napari for interactive editing.

    The user can edit the labeled mask using default napari tools.
    See `edit_annotation_napari` for the full keybinding list.

    Only the plate area (`mask_rect`) is shown, the same part `analyze_plate`
    measures; the labels outside it are kept and written back untouched. This
    function does the .npz translation: rects in and out of napari are plain
    (r0, r1, c0, c1) tuples, the int64 array is a storage detail.

    Input parameters:
    - curr_file: a fileinfo object (from root_length.functions_files.filelisting)
      pointing to the .npz segmentation file.
    - session_state: optional EditorSessionState, preserving widget/layer
      settings across calls (e.g. when looping over many files).

    Returns:
    - return_requests: dict forwarded from `edit_annotation_napari`. Currently
      used keys: 'quitloop_flag' (bool), 'go_to' (int or None).
    """
    
    # Load labeled mask from the .npz file
    segfile_path = curr_file.fullpath
    segfile_data = np.load(segfile_path, allow_pickle=True)
    img_mask = segfile_data['img_pred_lbls']
    # Preserve any extra arrays (e.g. prepr_info) for cropping use below
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
            # Load crop_rect, and handle incvonenient formatting
            # (TO DO: at save time, skip writing when is None)
            crop_rect = extra_arrays.get('prepr_info')
            # Handle array(None) or array([]) cases
            if isinstance(crop_rect, np.ndarray) and crop_rect.ndim == 0:
                crop_rect = crop_rect.item()
            # Load info
            if (crop_rect is not None):
                minr, maxr, minc, maxc = crop_rect
                img_original = img_original[minr:maxr, minc:maxc]
                # plt.imshow(img_original)
                print("Found")
            else:
                print("No cropping info found..")
    
    # Open napari for editing. The complete mask goes in; only the plate area
    # is shown (as analyze_plate sees it), and the complete mask comes back.
    seg_data, mask_rect, return_requests = edit_annotation_napari(
        image=img_original,
        segmentation=img_mask,
        mylabelcolormap=plutils.custom_colors_plantclasses,
        title=f"Editing #{curr_file.file_idx+1}: {curr_file.filename} ({curr_file.subdir})",
        session_state=session_state,
        curr_file=curr_file,
        mask_rect=extra_arrays.get('mask_rect'),
    )

    # Save or skip based on napari result. A None mask_rect (rectangle deleted,
    # or never there) leaves whatever is stored on disk alone.
    if seg_data is not None:
        _save_segfile(curr_file, seg_data,
                      extras_update=(None if mask_rect is None else
                                     {'mask_rect': np.asarray(mask_rect, dtype=np.int64)}))
    else:
        print(f"  Not saved: {segfile_path}")
    
    return return_requests


################################################################################
# %% edit all segfiles

def edit_all_segfiles(df_filelist, dir_inputfiles, dir_imagefiles=None,
                      session_state=None):
    """
    Loop over all segmentation files and open each in napari for editing.
    
    This iterates over the file list dataframe (same format as used by 
    analyze_all_plates) and opens each .npz file for interactive editing.
    
    The loop can be terminated early by pressing "q" in the napari viewer.
    
    Input parameters:
    - df_filelist: pd.DataFrame with columns 'basedir', 'subdir', 'filename'.
    - dir_inputfiles: str, the base directory for the segmentation files
      (used as the outputdir placeholder in fileinfo, since we save in-place).
    - session_state: optional EditorSessionState. If None, a fresh one is created
      so that widget/layer settings (shift, size thresholds, active label, brush
      size) are preserved across files within this loop.
    """

    if session_state is None:
        session_state = EditorSessionState()

    n_files = len(df_filelist)
    file_idx = 0
    while file_idx < n_files:
        
        ### Basic stuff -------
        # Get file info
        basedir, subdir, filename = \
            df_filelist.loc[file_idx, ['basedir', 'subdir', 'filename']]
        curr_file = ffl.fileinfo(basedir, subdir, filename, dir_inputfiles,
                                 file_idx = file_idx)

        print("========================================================")
        print(f"Editing file {file_idx+1}/{n_files}: {curr_file.fullpath}")

        ### Call Napari, open for editing -------
        return_requests = edit_segfile_single(curr_file, dir_imagefiles,
                                              session_state=session_state)


        ### Handling final admin -------
        if return_requests['quitloop_flag']:
            print("Loop terminated by user (pressed 'q').")
            break

        if return_requests['go_to'] is not None:
            # User asked to jump; clamp into valid range and convert to 0-based.
            target = max(1, min(n_files, return_requests['go_to']))
            if target != return_requests['go_to']:
                print(f"  Requested sample {return_requests['go_to']} "
                      f"out of range; clamped to {target}.")
            file_idx = target - 1
            return_requests['go_to'] = None
        else:
            file_idx += 1

    print("========================================================")
    print("Done editing segmentation files.")


################################################################################
# %% compute & save plate-area rect into each segfile

def compute_and_save_mask_rect_all(df_filelist, dir_inputfiles, dir_imagefiles,
                                   clear_outside_mask=False,
                                   only_process_n=None,
                                   overwrite=False,
                                   margin_left=0.05, margin_right=0.05,
                                   margin_top=0.1, margin_bottom=0.1,
                                   min_expected_area=0.5):
    """
    Compute a plate-area rect per file and store it as `mask_rect` in the .npz.

    Loops over `df_filelist` (as produced by `gen_metadatafile_segfiles`), loads
    each original image, runs `preprocess_getbbox_insideplate2` to obtain a
    rect = (minr, maxr, minc, maxc), and writes it into the matching seg .npz
    alongside the existing arrays (e.g. `img_pred_lbls`, `prepr_info`).

    Only the 4 numbers are stored, never a raster mask.

    Parameters:
    - clear_outside_mask: if True, also zero out the segmentation labels
      (`img_pred_lbls`) outside the computed rect before saving. Off by
      default, since `analyze_plate` discards labels outside the rect when
      loading anyway (see ConfigPipeline.apply_mask_rect); clearing here is
      destructive and cannot be undone by widening the rect later.
    - only_process_n: int or None. If int, only the first N rows of df_filelist
      are processed (useful for test runs); after that the loop exits. Default
      None processes all files.
    - overwrite: if True, recompute and overwrite an existing `mask_rect`.

    Notes:
    - If the segfile already has a `prepr_info` crop rect, the original image is
      cropped to it first so that `mask_rect` lives in the same coordinate frame
      as the seg arrays.
    - Margins default to 0.05 (5% of image size); see preprocess_getbbox_insideplate2.
    - `.npy` files are skipped (they can't hold extra keys).
    - A one-time backup of each seg file is made via `_backup_if_needed`.
    """

    n_files = len(df_filelist)
    if only_process_n is not None:
        n_files = min(n_files, only_process_n)
    for file_idx in range(n_files):
        basedir, subdir, filename = \
            df_filelist.loc[file_idx, ['basedir', 'subdir', 'filename']]
        curr_file = ffl.fileinfo(basedir, subdir, filename, dir_inputfiles)

        print(f"[{file_idx+1}/{n_files}] {curr_file.fullpath}")

        # .npy files can't store extra keys; skip
        if filename.endswith('.npy'):
            print("  Skipped: .npy files can't store extra keys.")
            continue

        # Load existing arrays from the segfile
        segfile_path = curr_file.fullpath
        segfile_data = np.load(segfile_path, allow_pickle=True)
        existing = {k: segfile_data[k] for k in segfile_data.files}

        if (not overwrite) and ('mask_rect' in existing):
            print("  Skipped: 'mask_rect' already present (use overwrite=True to recompute).")
            continue

        # Locate matching original image (any extension)
        imagefile_path_noext = os.path.join(
            dir_imagefiles, curr_file.subdir, curr_file.filebasename)
        file_hits = glob.glob(imagefile_path_noext + ".*")
        if len(file_hits) != 1:
            print(f"  Skipped: expected 1 image match, found {len(file_hits)} "
                  f"for {imagefile_path_noext}.*")
            continue
        img_original = skio.imread(file_hits[0])

        # If seg arrays were produced on a cropped image (prepr_info present),
        # apply that crop so mask_rect ends up in the seg coordinate frame.
        crop_rect = existing.get('prepr_info')
        if isinstance(crop_rect, np.ndarray) and crop_rect.ndim == 0:
            crop_rect = crop_rect.item()
        if crop_rect is not None:
            minr, maxr, minc, maxc = crop_rect
            img_original = img_original[minr:maxr, minc:maxc]

        # Compute the rect
        _, rect = plprep_seg.preprocess_getbbox_insideplate2(
            img_original,
            margin_left=margin_left, margin_right=margin_right,
            margin_top=margin_top, margin_bottom=margin_bottom,
            min_expected_area=min_expected_area)

        # Optionally clear seg labels outside the rect
        if clear_outside_mask and ('img_pred_lbls' in existing):
            seg = existing['img_pred_lbls']
            existing['img_pred_lbls'] = plprep.apply_mask_rect(
                seg, plprep.normalize_rect(rect, seg.shape))
            print("  Cleared seg labels outside rect.")

        # Save (preserve all existing keys, add/overwrite mask_rect)
        _backup_if_needed(segfile_path)
        existing['mask_rect'] = np.asarray(rect, dtype=np.int64)
        np.savez_compressed(segfile_path, **existing)
        print(f"  Saved mask_rect={tuple(int(v) for v in rect)}")

    print("Done computing mask_rect for all segfiles.")
