"""Subfunctions for estimating root length from a single-plant segmentation mask.

This module intentionally stays simple and explicit so it is easy to read for
Python beginners:
- two lightweight data containers (`TissueSample`, `PlantSample`)
- small, single-purpose functions
- one optional orchestration function (`run_default_length_pipeline`)
"""

################################################################################
# %% Libraries

import warnings

from dataclasses import dataclass
from typing import Callable, Iterable

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt
from skimage import morphology
from skimage.measure import label, regionprops

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import root_length.functions_pipeline.utils as plutils

# Sums the 8 neighbors around one pixel
KERNEL_NEIGHBOR_COUNT = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],
    dtype=int,
)


################################################################################
# %% Description of how to store data for one plant

@dataclass
class TissueSample:
    """Container for one tissue (root or shoot) of a single plant.

    Fields are tissue-agnostic: the same dataclass is instantiated for both
    root and shoot, and step functions operate via plain attribute access.
    The `anchor_mask` field is the *other* tissue's mask, used by
    `find_start_label_close_to_anchor` to bias the longest path's start.
    """

    mask: np.ndarray
    anchor_mask: np.ndarray | None = None

    clean_mask: np.ndarray | None = None
    skeleton: np.ndarray | None = None
    skeleton_nobranchpoints: np.ndarray | None = None
    branchpoint_coords: np.ndarray | None = None
    endpoint_coords: np.ndarray | None = None

    labeled_segments: np.ndarray | None = None
    segment_graph: nx.Graph | None = None

    start_labels: list[int] | None = None
    longest_path: list[int] | None = None
    mask_longest_path: np.ndarray | None = None

    length_pixels: float | None = None
    length_mm: float | None = None
    
@dataclass
class ConfigPipeline:
    smoothing_diskradius: int | None = None
    # derive root and shoot centerlines from one shared skeleton
    shared_skeleton: bool = True
    # largest dilation radius allowed to bridge holes in the root+shoot mask
    dilation_radius_maximum: int = 10

@dataclass
class PlantSample:
    """Container for one whole plant: a root + a shoot + plant-level metadata."""

    root: TissueSample
    shoot: TissueSample
    plant_mask: np.ndarray | None = None
    pixel_size_mm: float | None = None
    # position of the bbox in the original image
    bbox: tuple[int, int, int, int] | None = None

    # the root+shoot mask, if the two are analyzed as one object
    combined: TissueSample | None = None
    # administration of decisions taken:
    dilation_radius_used: int | None = None
    used_fallback: bool | None = None


################################################################################
# %% Basic preprocessing


def ensure_binary_mask(sample: TissueSample) -> TissueSample:
    """Convert `sample.mask` to boolean in place and return the sample."""

    sample.mask = sample.mask.astype(bool)

    return sample

def keep_largest_connected_component(sample: TissueSample) -> TissueSample:
    """
    Keep only the largest connected object in `sample.mask`.

    In principle, an ideal mask only contains one ROI. However, it might occur
    that other parts of the plant are labeled as the same tissue but not
    connected to the main area. We want to focus on the main area, so this
    function analyzes the regions, warns if multiple are present, and retains
    the largest one only. Result is stored in `sample.clean_mask`.
    """

    # Create labeled mask and props for the tissue mask
    labeled = label(sample.mask)
    props = regionprops(labeled)

    # In case there's nothing at all
    if not props:
        sample.clean_mask = np.zeros_like(sample.mask, dtype=bool)
        return sample

    # Warn (but continue) if multiple components were found
    if len(props) > 1:
        warnings.warn(
            f"mask has {len(props)} connected components, keeping largest"
        )

    # obtain the region properties element corresponding to the largest region
    def get_region_area(region):
        return region.area
    largest_region = max(props, key=get_region_area)

    # Now create a new mask, corresponding to the largest region
    sample.clean_mask = (labeled == largest_region.label)
        # plt.imshow(sample.clean_mask)

    return sample

def apply_smoothing(sample, config_pipeline):
    """ Smooths mask to avoid spurious branching. """
    
    footprint = morphology.disk(config_pipeline.smoothing_diskradius)
    sample.clean_mask = morphology.closing(sample.clean_mask,
                                           footprint=footprint)

    return sample

################################################################################
# %% Shared skeleton for root and shoot

# Skeletonizing root and shoot separately distorts both centerlines at the
# border between them, because the medial axis of a truncated shape retracts
# from the cut. The functions below instead treat root+shoot as one object,
# skeletonize that once, and divide the result again afterwards.

def region_distances(mask):
    """
    Find inter-island distances.

    Label connected components ("islands"), then for each one find the min
    distance to the nearest pixel belonging to any *other* component. Returns
    an empty array when there is nothing to bridge (mask is already whole).
    """

    # Label, using the same 8-connectivity as the rest of the pipeline
    labels, n_labels = label(mask, return_num=True)

    # A whole (or empty) mask has no gaps to report
    if n_labels <= 1:
        return np.array([], dtype=float)

    # Loop over each "island"
    distances = []
    for island_idx in range(1, n_labels + 1):

        # create mask for other "islands"
        mask_others = (labels > 0) & (labels != island_idx)

        # get distances to closest for each pixel in the island
        current_distances = distance_transform_edt(~mask_others)
        # now select and save the smallest for island i
        distances.append(current_distances[labels == island_idx].min())

    return np.array(distances)


def dilate_to_connect(mask, config_pipeline):
    """
    Dilate a mask just enough to merge its separate parts into one region.

    Holes in a mask (e.g. a seed interrupting the root) split it in two, which
    would fragment the skeleton. Two regions separated by a gap g merge once
    both are dilated by g/2. The surplus material added by the
    dilation is dealt with by `prune_skeleton_outside_mask` later on.

    The radius is kept as small as possible, because dilation distorts: bends
    tighter than the radius get straightened out, and structures passing within
    twice the radius of each other get fused. `dilation_radius_maximum` limits
    that damage; beyond it the gap is left alone and (None) is returned as
    radius, signalling the caller to fall back to per-tissue processing.
    """

    # Determine gaps between the separate parts, if any
    distances = region_distances(mask)
    if distances.size == 0:
        return mask, 0

    # Half the largest gap suffices, +1 as margin for the discrete disk
    radius = int(np.ceil(distances.max() / 2)) + 1

    # abort if radius too large
    if radius > config_pipeline.dilation_radius_maximum:
        return mask, None

    # perform dilation
    mask_dilated = morphology.binary_dilation(mask, morphology.disk(radius))

    # Give up if this didn't actually result in one single region
    if label(mask_dilated, return_num=True)[1] >1:
        return mask, None

    return mask_dilated, radius


def prune_skeleton_outside_mask(skeleton, mask):
    """
    Strip skeleton branch ends that stick out of `mask`.

    Dilating before skeletonizing also extends the free ends of the object (the
    root tip, the shoot apex) by the dilation radius, which would inflate the
    measured length. Repeatedly removing end points that lie outside the
    original mask undoes that. Bridges over holes survive, since they run
    between two parts of the mask and hence never end in a free end.
    """

    skeleton = skeleton.copy()
    while True:

        # End points have one neighbor at most (isolated pixels have none)
        # count number of connections of each pixel
        neighbor_counts = convolve(skeleton.astype(int), KERNEL_NEIGHBOR_COUNT,
                                   mode="constant", cval=0)
        # remove loose ends (<= neighbor) outside original mask
        pixels_to_remove = skeleton & (neighbor_counts <= 1) & ~mask

        # Done once all remaining end points lie inside the mask
        if not pixels_to_remove.any():
            return skeleton

        skeleton = skeleton & ~pixels_to_remove


def assign_nearest_tissue(root_mask, shoot_mask):
    """
    Label every pixel by the tissue it lies closest to.

    Dilation adds pixels that belong to neither original mask, and
    other tissues (a seed, typically) can sit in between the two. Handing those
    pixels to the nearest tissue divides the image cleanly in two, so the shared
    skeleton can be split up without losing pixels in between.
    """

    # 2 = root, 1 = shoot, following the labels used in the plant masks
    tissue_ids = np.where(root_mask, 2, np.where(shoot_mask, 1, 0))

    # For each pixel, look up the identity of the nearest labeled pixel
    _, (rows_nearest, cols_nearest) = \
        distance_transform_edt(tissue_ids == 0, return_indices=True)

    return tissue_ids[rows_nearest, cols_nearest]


def prepare_shared_skeleton(plant: PlantSample,
                            config_pipeline: ConfigPipeline) -> PlantSample:
    """
    Skeletonize the root+shoot mask as a whole, and split the result per tissue.

    Sets `skeleton` (plus `clean_mask` and `anchor_mask`) on both tissues, which
    makes `run_tissue_pipeline` skip its own skeletonization. If the mask cannot
    be made whole, nothing is set and `used_fallback` is raised instead, so that
    each tissue simply gets processed on its own as before.
    """

    # Root and shoot are analyzed as one object
    mask_combined = plant.root.mask.astype(bool) | plant.shoot.mask.astype(bool)
    plant.combined = TissueSample(mask=mask_combined)

    # Work on a padded copy, as the plant touches the edges of its bounding box
    # and both the closing and the dilation need room around it
    pad = config_pipeline.dilation_radius_maximum
    plant.combined.clean_mask = np.pad(mask_combined, pad)

    # smooth (morphological closing) the mask if desired
    if config_pipeline.smoothing_diskradius is not None:
        plant.combined = apply_smoothing(plant.combined, config_pipeline)

    # Bridge holes that would otherwise fragment the skeleton
    plant.combined.clean_mask, plant.dilation_radius_used = \
        dilate_to_connect(plant.combined.clean_mask, config_pipeline)

    # Bail out if the mask couldn't be made whole
    plant.used_fallback = plant.dilation_radius_used is None
    if plant.used_fallback:
        return plant

    # Skeletonize the whole object, then undo the padding
    skeleton = morphology.skeletonize(plant.combined.clean_mask)
    n_rows, n_cols = mask_combined.shape
    plant.combined.clean_mask = \
        plant.combined.clean_mask[pad:pad+n_rows, pad:pad+n_cols]
    skeleton = skeleton[pad:pad+n_rows, pad:pad+n_cols]

    # Remove the spurs that the dilation grew at the free ends
    plant.combined.skeleton = prune_skeleton_outside_mask(skeleton, mask_combined)
        # plt.imshow(plant.combined.skeleton + plant.combined.mask)

    # Now divide the shared skeleton over the two tissues
    nearest_tissue = assign_nearest_tissue(plant.root.mask, plant.shoot.mask)
    for tissue, tissue_id in ((plant.root, 2), (plant.shoot, 1)):
        tissue.clean_mask = plant.combined.clean_mask & (nearest_tissue == tissue_id)
        tissue.skeleton = plant.combined.skeleton & (nearest_tissue == tissue_id)

    # Anchor each tissue to the other one's skeleton, such that the longest
    # path is made to start at the root/shoot junction
    plant.root.anchor_mask = plant.shoot.skeleton
    plant.shoot.anchor_mask = plant.root.skeleton

    return plant

################################################################################
# %% Branch analysis

def generate_skeleton(sample: TissueSample) -> TissueSample:
    """Skeletonize `sample.clean_mask`."""

    sample.skeleton = morphology.skeletonize(sample.clean_mask)

    return sample


def analyze_skeleton_branchpoints(sample: TissueSample) -> TissueSample:
    """Remove branch-point pixels from `sample.skeleton`, and locate the nodes."""

    # Create an equal-sized array that gives the neighbor count for each pixel
    # in the skeleton.
    neighbor_counts = convolve(
        sample.skeleton.astype(int),
        KERNEL_NEIGHBOR_COUNT,
        mode="constant",
        cval=0,
    )

    # Now only keep parts of the skeleton that have <=2 neighbors
    sample.skeleton_nobranchpoints = \
        sample.skeleton & (neighbor_counts <= 2)
        # plt.imshow(sample.skeleton_nobranchpoints)

    # and collect the x,y locations of both the branch points as
    # well as the end points.
    # Locations of branch points
    sample.branchpoint_coords = np.column_stack(
        np.where(sample.skeleton & (neighbor_counts > 2))
    )
    # Locations of end points
    sample.endpoint_coords = np.column_stack(
        np.where(sample.skeleton & (neighbor_counts == 1))
    )

    return sample


def label_skeleton_segments(sample: TissueSample) -> TissueSample:
    """Label line segments in skeleton and assign separate labels to nodes."""

    # Now get the labeled skeleton
    labeled_segments = morphology.label(sample.skeleton_nobranchpoints)
    max_label = int(labeled_segments.max())
        # plt.imshow(labeled_segments)

    # Collect a list of pixel locations that require to be assigned a new label
    pixel_coords_list = []
    if sample.branchpoint_coords is not None and sample.branchpoint_coords.size > 0:
        pixel_coords_list.append(sample.branchpoint_coords)
    if sample.endpoint_coords is not None and sample.endpoint_coords.size > 0:
        pixel_coords_list.append(sample.endpoint_coords)

    # Now loop over those pixels (if available)
    if pixel_coords_list:
        pixel_coords = np.vstack(pixel_coords_list)
        for idx, coord in enumerate(pixel_coords):
            labeled_segments[coord[0], coord[1]] = idx + max_label + 1

    sample.labeled_segments = labeled_segments

    # plt.imshow(sample.labeled_segments)

    return sample


################################################################################
# %% Graph construction and path finding

# Kernel with distances for 8-connectivity (diagonal neighbors have distance sqrt(2))    
# The distances are divided by 2, because otherwise lengths are counted twice.
DISTANCE_KERNEL = np.array([[np.sqrt(2), 1, np.sqrt(2)],
                            [1, 0, 1],
                            [np.sqrt(2), 1, np.sqrt(2)]]) / 2

    
def get_length_segment(the_mask, distance_kernel = DISTANCE_KERNEL):
    """
    Calculate the length of a line drawn in a matrix.
    
    Assuming that the mask provides pixels which all have >0 neighbors <3, 
    and form one continuous structure, calculate the length of the line
    defined by the structure in the mask.
    
    By construction, the other functions should have created a line
    conforming to the constraints above (pixels with >2 neighbors
    are isolated and processed separately).
    
    For each pixel, distance to its one or two neighboring pixels is 
    determined using convolution with a distance kernel.
    
    test_mask = np.array([[ 0, 0, 1, 0, 0],
                          [ 0, 0, 1, 0, 0],
                          [ 0, 1, 0, 0, 1],
                          [ 0, 0, 1, 1, 0]])
    # expected length (midpoints)
    # 1 + np.sqrt(2) + np.sqrt(2) + 1 + np.sqrt(2) = 6.242640687119285
    """
    # the_mask = test_mask
    # distance_kernel = DISTANCE_KERNEL
    
    # If there's only one pixel, the length is set to 0.5
    # This is an approximation, to also assign branch points a length.
    if np.sum(the_mask) == 1:
        return 0.5
    
    # first test wether all pixels are indeed only connected to <3 neighbors
    neighbor_counts = convolve(the_mask.astype(int),
                               KERNEL_NEIGHBOR_COUNT, 
                               mode='constant', cval=0)
    if (np.any(neighbor_counts[the_mask>0] > 2) or
        np.any(neighbor_counts[the_mask>0] < 1)):
        warnings.warn("Mask is not a valid line (isolated pixels or connected to >2 neighbors)")
        return np.nan

    # now for each pixel, get the total distance to all neighbors
    neighbor_distances = convolve(the_mask.astype(float),
                                  distance_kernel,
                                  mode='constant',
                                  cval=0)
    neighbor_distances[the_mask==0] = 0

    # now sum up the distances for all pixels in the mask
    line_length = np.sum(neighbor_distances[the_mask>0])
    
    return line_length


def plot_graph_nodesize(G, size_metric="length"):
    fig, ax = plt.subplots()
    node_sizes = np.array([G.nodes[n][size_metric] for n in G.nodes])
    nx.draw(G, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=node_sizes*10)
    ax.set_title("Connectivity Graph")
    
    # plt.show()
    return fig, ax


def build_segment_graph(sample: TissueSample) -> TissueSample:
    """
    Create a graph where each segment label is one node.

    Note that there is a small imprecision here, as branch point pixels
    have length = 0.5. (This could lead to the longest path that will be
    identified later not actually being the longest, in extreme edge cases.
    The length calculated later will include these pixels, as then the
    length is calculated again.)
    """

    # Obtain unique segment labels
    unique_labels = np.unique(sample.labeled_segments)
    unique_labels = unique_labels[unique_labels != 0]

    # Initialize a graph using unique labels
    # (Connections are added below)
    graph = nx.Graph()
    graph.add_nodes_from(unique_labels)

    # Dilation element to check all direct neighbors
    structure_all8neihbors = morphology.footprint_rectangle((3, 3))

    # Loop over each segment
    for label_id in unique_labels:
        # Get segment-specific mask
        current_mask = sample.labeled_segments == label_id
        # Dilate it
        dilated_mask = morphology.binary_dilation(current_mask, structure_all8neihbors)

        # Now from the dilated mask collect labels in the original mask,
        # thus collecting neighboring segment lables
        neighboring_pixels = sample.labeled_segments[dilated_mask]
        neighbor_labels = np.unique(neighboring_pixels)
        # exclude self and zero
        neighbor_labels = neighbor_labels[
            (neighbor_labels != 0) & (neighbor_labels != label_id)
        ]

        # now add the neighbors to the graph
        for neighbor in neighbor_labels:
            graph.add_edge(int(label_id), int(neighbor))

        # also keep track of original segment area
        graph.nodes[int(label_id)]["area"]   = int(np.sum(current_mask))
        graph.nodes[int(label_id)]["length"] = get_length_segment(current_mask)
            # plt.imshow(current_mask)

    sample.segment_graph = graph

    # plot_graph_nodesize(graph)

    return sample


def find_start_labels_close_to_anchor(sample: TissueSample) -> TissueSample:
    """
    Per connected part of the skeleton, pick the segment nearest `anchor_mask`.

    The anchor is typically the *other* tissue (root anchors to shoot, shoot
    anchors to root), so these labels sit at the root/shoot junction, which is
    where the longest path should start.

    One label is collected per connected part, rather than a single overall
    closest one, because splitting a shared skeleton can leave a tissue in
    several disconnected parts. Small stray parts can then lie closer to the
    anchor than the real junction does, and picking only the overall closest
    label would trap the path search inside such a stray.
    `get_long_path_in_graph_nodearea` simply takes whichever start yields the
    longest path. If `anchor_mask` is None, the step is skipped and that
    function falls back to using any node.
    """

    if sample.anchor_mask is None:
        return sample

    # now get distance map to the anchor tissue
    distance_map = distance_transform_edt(~sample.anchor_mask.astype(bool))

    # for each segment, determine how far it is from the anchor tissue
    labels_all = list(sample.segment_graph.nodes)
    distance_per_label = dict(zip(labels_all, ndimage.minimum(
        distance_map, sample.labeled_segments, index=labels_all)))

    # and per connected part, keep the segment that's closest
    sample.start_labels = \
        [min(part, key=lambda lbl: distance_per_label[lbl])
         for part in nx.connected_components(sample.segment_graph)]

    return sample


def helper_print_graph_node_lengths(graph):
    """Debug fn, print lengths of each of the nodes"""
    for n in graph.nodes:
        print(f"Node {n}: length {graph.nodes[n].get('length', 'N/A')}")

def get_long_path_in_graph_nodearea(sample: TissueSample) -> TissueSample:
    """Find a long path by maximizing sum of node areas along shortest paths."""

    graph = sample.segment_graph

    # If empty graph, simply return
    if graph.number_of_nodes() == 0:
        sample.longest_path = []
        sample.length_pixels = 0.0
        return sample

    # The start labels contain the starting points closest to the anchor
    # tissue (this is required, because there might be a longest path not
    # touching the anchor), so we want to select those as starting nodes.
    source_nodes = sample.start_labels if sample.start_labels else list(graph.nodes)

    # Initialize
    longest_path = []
    max_length = 0.0

    # check all pairs of nodes, and identify the longest shortest path between
    # the starting node and any other node
    # (splitting a shared skeleton can leave a tissue in disconnected pieces,
    # so only consider the nodes that are reachable from the source)
    for source in source_nodes:
        for target in nx.node_connected_component(graph, source):
            if source != target:
                path = nx.shortest_path(graph, source=source, target=target, weight='length')
                # Calculate path length
                path_length = sum(graph.nodes[n].get('length', 1) for n in path)
                if path_length > max_length:
                    max_length = path_length
                    longest_path = path
                # print(f"For node {source}-->{target}, length was {path_length:.2f} pixels")
    #print(f"Longest path length: {max_length}")
    #print(f"Longest path end nodes: {[longest_path[0], longest_path[-1]]}")

    # Now store the longest path
    sample.longest_path = longest_path
    sample.length_pixels = max_length
    return sample

def build_longest_path_mask(sample: TissueSample) -> TissueSample:
    """Create a binary mask of the longest path."""

    # Create a new mask based on the labeled mask, which only retains pixels
    # that are the longest path.
    sample.mask_longest_path = \
        np.isin(sample.labeled_segments, sample.longest_path)
    # plt.imshow(sample.mask_longest_path)

    return sample

def get_length_longestpath(sample: TissueSample) -> TissueSample:
    """Calculate the length of the longest path using the mask of that path."""

    # Calculate length of longest path using the mask of that path
    sample.length_pixels = get_length_segment(sample.mask_longest_path)

    return sample

################################################################################
# %% orchestrator

def return_bbox_foreground(mask):
    """ Return recteangle coordinates surrounding >0 pixels in mask."""
    
    foreground_coords = np.argwhere(mask > 0)

    min_row = foreground_coords[:, 0].min()
    min_col = foreground_coords[:, 1].min()
    max_row = foreground_coords[:, 0].max() + 1  # +1 because slicing is exclusive at the end
    max_col = foreground_coords[:, 1].max() + 1

    return (min_row, min_col, max_row, max_col)
    
def plot_original_and_length(sample: TissueSample, plant_mask: np.ndarray | None = None):
    """Plot the plant mask, with the tissue skeleton and longest path on top."""

    fig, axs = plt.subplots(1, 2)

    # Show original plant mask (or fall back to the tissue mask)
    background = plant_mask if plant_mask is not None else sample.mask
    axs[0].imshow(background, cmap=plutils.cmap_plantclasses)

    # Overlay the skeleton, colored in blue
    axs[0].imshow(sample.skeleton, cmap=ListedColormap(['none', 'blue']),
            alpha=(sample.skeleton>0)*1.0)
        # plt.imshow(sample.mask); plt.imshow(sample.skeleton, cmap=ListedColormap(['none', '#cccccc']))
    # Overlay the longest path, colored in red
    axs[0].imshow(sample.mask_longest_path, cmap=ListedColormap(['none', 'red']),
              alpha=(sample.mask_longest_path>0)*1.0)

    # Now same but zoomed to the tissue bbox
    r0, c0, r1, c1 = return_bbox_foreground(sample.mask)
    axs[1].imshow(sample.mask[r0:r1, c0:c1], cmap=ListedColormap(['black', plutils.custom_colors_plantclasses[2]]))
    axs[1].imshow(sample.skeleton[r0:r1, c0:c1], cmap=ListedColormap(['none', 'blue']),
                alpha=(sample.skeleton[r0:r1, c0:c1] > 0) * 1.0)
    axs[1].imshow(sample.mask_longest_path[r0:r1, c0:c1], cmap=ListedColormap(["none", "red"]),
                alpha=(sample.mask_longest_path[r0:r1, c0:c1] > 0) * 1.0)

    # Cosmetics
    axs[0].axis("off")
    axs[1].axis("off")
    fig.suptitle(f"Estimated length: {sample.length_pixels:.2f} px")

    return fig, axs

def plot_distance_graph(sample: TissueSample):
    """ Show the distance graph alongside the labeled segments."""

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs  # keep compatibility with existing `return fig, ax`

    # Left panel: labeled segments, zoomed to tissue region only
    labeled = sample.labeled_segments
    if sample.mask is not None and np.any(sample.mask > 0):
        r0, c0, r1, c1 = return_bbox_foreground(sample.mask)
        labeled_view = labeled[r0:r1, c0:c1]
    else:
        r0, c0 = 0, 0
        labeled_view = labeled

    im = axs[0].imshow(labeled_view, cmap="nipy_spectral")
    axs[0].set_title("Labeled segments (tissue ROI)")
    axs[0].axis("off")

    rows, cols = np.where(labeled_view > 0)
    for r, c in zip(rows, cols):
        axs[0].text(
            c,
            r,
            str(int(labeled_view[r, c])),
            color="white",
            ha="center",
            va="center",
            fontsize=6,
        )

    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    # Right panel: connectivity graph
    node_sizes = np.array(
        [
            sample.segment_graph.nodes[n].get("length", 1)
            for n in sample.segment_graph.nodes
        ],
    )
    nx.draw(
        sample.segment_graph,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=node_sizes * 10,
        ax=axs[1],
    )
    axs[1].set_title("Connectivity Graph")
    axs[1].axis("off")

    # suptitle with labels of longest path
    longest_path_labels = sample.longest_path if sample.longest_path else []
    axs[1].set_title(f"Connectivity Graph\nLongest path labels: {longest_path_labels}")

    return fig, ax

# %%

def plot_all_plants_projected(
        sample_image: np.ndarray,
        plant_results,
        figsize: tuple[int, int] = (12, 12)):
    """
    Make overview plot.
    
    Run determine_length() for each individual plant image and project all
    traced centerlines + lengths back onto the original sample image.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(sample_image, cmap=plutils.cmap_plantclasses)
    #ax.set_title("All plants projected on sample image", fontsize=13, pad=10)
    #ax.axis("off")

    ax.autoscale(False)

    colors = plt.cm.tab20(np.linspace(0, 1, len(plant_results)))

    for idx, (plant, color) in enumerate(zip(plant_results, colors), start=1):

        # idx = 0; plant = plant_results[idx]; color = colors[idx]

        # show the bbox
        minr, minc, maxr, maxc = plant.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

        # Project skeleton pixels back to full-image coordinates
        # (showing the shared root+shoot skeleton where it is available)
        skeleton_to_plot = plant.root.skeleton
        if plant.combined is not None and plant.combined.skeleton is not None:
            skeleton_to_plot = plant.combined.skeleton
        if skeleton_to_plot is not None and np.any(skeleton_to_plot):
            ax.imshow(
            skeleton_to_plot,
            cmap=ListedColormap(["none", "gray"]),
            alpha=(skeleton_to_plot > 0) * 1.0,
            interpolation="none",
            extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Project root longest-path pixels back to full-image coordinates
        if plant.root.mask_longest_path is not None and np.any(plant.root.mask_longest_path):
            ax.imshow(
                plant.root.mask_longest_path,
                cmap=ListedColormap(["none", "red"]),
                alpha=(plant.root.mask_longest_path > 0) * 1.0,
                interpolation="none",
                extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Project shoot longest-path pixels back to full-image coordinates
        if plant.shoot.mask_longest_path is not None and np.any(plant.shoot.mask_longest_path):
            ax.imshow(
                plant.shoot.mask_longest_path,
                cmap=ListedColormap(["none", "orange"]),
                alpha=(plant.shoot.mask_longest_path > 0) * 1.0,
                interpolation="none",
                extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Optional length label near each bbox (root + shoot)
        if (plant.root.length_pixels is not None
                or plant.shoot.length_pixels is not None):
            root_txt = (f"{plant.root.length_pixels:.1f}"
                        if plant.root.length_pixels is not None else "n/a")
            shoot_txt = (f"{plant.shoot.length_pixels:.1f}"
                         if plant.shoot.length_pixels is not None else "n/a")
            ax.text(
            minc,
            minr - 3,
            f"({idx}) root {root_txt}px\nshoot {shoot_txt}px",
            color=color,
            fontsize=5,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
    
    # reset zoom to full image
    ax.set_xlim(0, sample_image.shape[1])
        
    plt.tight_layout()
    return fig, ax

# %% runner

def run_tissue_pipeline(sample: TissueSample,
                        config_pipeline: ConfigPipeline) -> TissueSample:
    """
    Run the full default sequence of processing steps for one tissue.

    When a shared root+shoot skeleton was prepared already (see
    `prepare_shared_skeleton`), the mask cleaning and skeletonization steps are
    skipped, and this tissue's share of that skeleton is analyzed instead.
    """

    if sample.skeleton is None:

        # Make binary and select largest ROI to analyze
        sample = ensure_binary_mask(sample)
        sample = keep_largest_connected_component(sample)

        # smooth (morphological closing) the mask if desired
        if config_pipeline.smoothing_diskradius is not None:
            sample = apply_smoothing(sample, config_pipeline)

        sample = generate_skeleton(sample)

    # Generate a labeled skeleton to analyze
    sample = analyze_skeleton_branchpoints(sample)
    sample = label_skeleton_segments(sample)

    # Build a graph, and find the longest path
    sample = build_segment_graph(sample)
        # plot_distance_graph(sample)
    sample = find_start_labels_close_to_anchor(sample)
    sample = get_long_path_in_graph_nodearea(sample)
    sample = build_longest_path_mask(sample)
        # plot_original_and_length(sample, plant_mask)
    sample = get_length_longestpath(sample)

    return sample


def run_default_length_pipeline(plant: PlantSample,
                                config_pipeline: ConfigPipeline) -> PlantSample:
    """Run the per-tissue pipeline for both root and shoot of one plant."""

    # Derive both centerlines from one shared skeleton, if desired
    # (falls through to independent processing if that doesn't work out)
    if config_pipeline.shared_skeleton:
        plant = prepare_shared_skeleton(plant, config_pipeline)

    # Process each tissue independently
    plant.root = run_tissue_pipeline(plant.root, config_pipeline)
    plant.shoot = run_tissue_pipeline(plant.shoot, config_pipeline)

    # Add distance in mm (if pixel size is known)
    if plant.pixel_size_mm is not None:
        plant.root.length_mm = plant.root.length_pixels * plant.pixel_size_mm
        plant.shoot.length_mm = plant.shoot.length_pixels * plant.pixel_size_mm
    else:
        plant.root.length_mm = None
        plant.shoot.length_mm = None

    return plant
