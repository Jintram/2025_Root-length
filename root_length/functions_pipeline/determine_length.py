"""Subfunctions for estimating root length from a single-plant segmentation mask.

This module intentionally stays simple and explicit so it is easy to read for
Python beginners:
- one lightweight data container (`RootSample`)
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
class RootSample:
    """Container for one plant sample (root + shoot) during processing.

    Per-tissue intermediates exist for both "root" and "shoot"; the processing
    functions select which one to use via a `tissue: str` parameter and
    read/write the matching `{tissue}_<field>` attribute via getattr/setattr.
    """

    root_mask: np.ndarray
    shoot_mask: np.ndarray
    plant_mask: np.ndarray | None = None
    pixel_size_mm: float | None = None

    clean_root_mask: np.ndarray | None = None
    clean_shoot_mask: np.ndarray | None = None

    root_skeleton: np.ndarray | None = None
    shoot_skeleton: np.ndarray | None = None
    root_skeleton_nobranchpoints: np.ndarray | None = None
    shoot_skeleton_nobranchpoints: np.ndarray | None = None

    root_branchpoint_coords: np.ndarray | None = None
    shoot_branchpoint_coords: np.ndarray | None = None
    root_endpoint_coords: np.ndarray | None = None
    shoot_endpoint_coords: np.ndarray | None = None

    root_labeled_segments: np.ndarray | None = None
    shoot_labeled_segments: np.ndarray | None = None
    root_segment_graph: nx.Graph | None = None
    shoot_segment_graph: nx.Graph | None = None

    root_start_label: int | None = None
    shoot_start_label: int | None = None
    root_longest_path: list[int] | None = None
    shoot_longest_path: list[int] | None = None

    root_length_pixels: float | None = None
    shoot_length_pixels: float | None = None
    root_length_mm: float | None = None
    shoot_length_mm: float | None = None

    # position of the bbox in the original image
    bbox: tuple[int, int, int, int] | None = None

    mask_longest_path_root: np.ndarray | None = None
    mask_longest_path_shoot: np.ndarray | None = None


################################################################################
# %% Basic preprocessing


def ensure_binary_mask(sample: RootSample, tissue: str) -> RootSample:
    """Convert the `{tissue}_mask` to boolean in place and return the sample."""

    mask_attr = f"{tissue}_mask"
    setattr(sample, mask_attr, getattr(sample, mask_attr).astype(bool))

    return sample

def keep_largest_connected_component(sample: RootSample, tissue: str) -> RootSample:
    """
    Keep only the largest connected object in the `{tissue}_mask`.

    In principle, an ideal mask only contains one ROI per tissue. However, it
    might occur that other parts of the plant are labeled as the same tissue
    but not connected to the main area. We want to focus on the main area,
    so this function analyzes the regions, warns if multiple are present,
    and retains the largest one only. Result is stored in `clean_{tissue}_mask`.
    """

    mask = getattr(sample, f"{tissue}_mask")
    clean_attr = f"clean_{tissue}_mask"

    # Create labeled mask and props for the tissue mask
    labeled = label(mask)
    props = regionprops(labeled)

    # In case there's nothing at all
    if not props:
        setattr(sample, clean_attr, np.zeros_like(mask, dtype=bool))
        return sample

    # Warn (but continue) if multiple components were found
    if len(props) > 1:
        warnings.warn(
            f"{tissue} mask has {len(props)} connected components, keeping largest"
        )

    # obtain the region properties element corresponding to the largest region
    def get_region_area(region):
        return region.area
    largest_region = max(props, key=get_region_area)

    # Now create a new mask, corresponding to the largest region
    setattr(sample, clean_attr, (labeled == largest_region.label))
        # plt.imshow(getattr(sample, clean_attr))

    return sample

################################################################################
# %% Branch analysis

def generate_skeleton_no_branchpoints(sample: RootSample, tissue: str) -> RootSample:
    """Skeletonize `clean_{tissue}_mask` and remove branch-point pixels."""

    clean_mask = getattr(sample, f"clean_{tissue}_mask")

    # Obtain the skeleton
    skeleton = morphology.skeletonize(clean_mask)
    setattr(sample, f"{tissue}_skeleton", skeleton)

    # Create an equal-sized array that gives the neighbor count for each pixel
    # in the skeleton.
    neighbor_counts = convolve(
        skeleton.astype(int),
        KERNEL_NEIGHBOR_COUNT,
        mode="constant",
        cval=0,
    )

    # Now only keep parts of the skeleton that have <=2 neighbors
    setattr(sample, f"{tissue}_skeleton_nobranchpoints",
            skeleton & (neighbor_counts <= 2))
        # plt.imshow(getattr(sample, f"{tissue}_skeleton_nobranchpoints"))

    # and collect the x,y locations of both the branch points as
    # well as the end points.
    # Locations of branch points
    setattr(sample, f"{tissue}_branchpoint_coords",
            np.column_stack(np.where(skeleton & (neighbor_counts > 2))))
    # Locations of end points
    setattr(sample, f"{tissue}_endpoint_coords",
            np.column_stack(np.where(skeleton & (neighbor_counts == 1))))

    return sample


def label_skeleton_segments(sample: RootSample, tissue: str) -> RootSample:
    """Label line segments in skeleton and assign separate labels to nodes."""

    skeleton_nobp = getattr(sample, f"{tissue}_skeleton_nobranchpoints")
    branchpoint_coords = getattr(sample, f"{tissue}_branchpoint_coords")
    endpoint_coords = getattr(sample, f"{tissue}_endpoint_coords")

    # Now get the labeled skeleton
    labeled_segments = morphology.label(skeleton_nobp)
    max_label = int(labeled_segments.max())
        # plt.imshow(labeled_segments)

    # Collect a list of pixel locations that require to be assigned a new label
    pixel_coords_list = []
    if branchpoint_coords is not None and branchpoint_coords.size > 0:
        pixel_coords_list.append(branchpoint_coords)
    if endpoint_coords is not None and endpoint_coords.size > 0:
        pixel_coords_list.append(endpoint_coords)

    # Now loop over those pixels (if available)
    if pixel_coords_list:
        pixel_coords = np.vstack(pixel_coords_list)
        for idx, coord in enumerate(pixel_coords):
            labeled_segments[coord[0], coord[1]] = idx + max_label + 1

    setattr(sample, f"{tissue}_labeled_segments", labeled_segments)

    # plt.imshow(getattr(sample, f"{tissue}_labeled_segments"))

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


def build_segment_graph(sample: RootSample, tissue: str) -> RootSample:
    """
    Create a graph where each segment label is one node.

    Note that there is a small imprecision here, as branch point pixels
    have length = 0.5. (This could lead to the longest path that will be
    identified later not actually being the longest, in extreme edge cases.
    The length calculated later will include these pixels, as then the
    length is calculated again.)
    """

    labeled_segments = getattr(sample, f"{tissue}_labeled_segments")

    # Obtain unique segment labels
    unique_labels = np.unique(labeled_segments)
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
        current_mask = labeled_segments == label_id
        # Dilate it
        dilated_mask = morphology.binary_dilation(current_mask, structure_all8neihbors)

        # Now from the dilated mask collect labels in the original mask,
        # thus collecting neighboring segment lables
        neighboring_pixels = labeled_segments[dilated_mask]
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

    setattr(sample, f"{tissue}_segment_graph", graph)

    # plot_graph_nodesize(graph)

    return sample


def find_start_label_close_to_other(sample: RootSample, tissue: str) -> RootSample:
    """
    Pick the segment label nearest to the *other* tissue's mask.

    For `tissue="root"`, the anchor is `shoot_mask` (root starts near shoot).
    For `tissue="shoot"`, the anchor is `root_mask` (shoot starts near root).
    """

    other_tissue = "shoot" if tissue == "root" else "root"
    anchor_mask = getattr(sample, f"{other_tissue}_mask")
    labeled_segments = getattr(sample, f"{tissue}_labeled_segments")

    # now get distance map to the anchor tissue
    distance_map = distance_transform_edt(~anchor_mask.astype(bool))
    # disregard background pixels (set to inf distance)
    distance_map[labeled_segments == 0] = np.inf

    # and find the pixel that is closest to the anchor tissue
    closest_pixel = np.unravel_index(np.argmin(distance_map), distance_map.shape)
    # and its corresponding label
    setattr(sample, f"{tissue}_start_label", int(labeled_segments[closest_pixel]))

    return sample


def helper_print_graph_node_lengths(graph):
    """Debug fn, print lengths of each of the nodes"""
    for n in graph.nodes:
        print(f"Node {n}: length {graph.nodes[n].get('length', 'N/A')}")

def get_long_path_in_graph_nodearea(sample: RootSample, tissue: str) -> RootSample:
    """Find a long path by maximizing sum of node areas along shortest paths."""

    graph = getattr(sample, f"{tissue}_segment_graph")
    start_label = getattr(sample, f"{tissue}_start_label")

    # If empty graph, simply return
    if graph.number_of_nodes() == 0:
        setattr(sample, f"{tissue}_longest_path", [])
        setattr(sample, f"{tissue}_length_pixels", 0.0)
        return sample

    # The start_label should contain the starting point closest to the anchor
    # tissue (this is required, because there might be a longest path not
    # touching the anchor), so we want to select that as starting node.
    source_nodes = [start_label] if start_label in graph else list(graph.nodes)

    # Initialize
    longest_path = []
    max_length = 0.0

    # check all pairs of nodes, and identify the longest shortest path between
    # the starting node and any other node
    # check all pairs of nodes
    for source in source_nodes:
        for target in graph.nodes:
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
    setattr(sample, f"{tissue}_longest_path", longest_path)
    setattr(sample, f"{tissue}_length_pixels", max_length)
    return sample

def build_longest_path_mask(sample: RootSample, tissue: str) -> RootSample:
    """Create a binary mask of the longest path for the given tissue."""

    labeled_segments = getattr(sample, f"{tissue}_labeled_segments")
    longest_path = getattr(sample, f"{tissue}_longest_path")

    # Create a new mask based on the labeled mask, which only retains pixels
    # that are the longest path.
    setattr(sample, f"mask_longest_path_{tissue}",
            np.isin(labeled_segments, longest_path))
    # plt.imshow(getattr(sample, f"mask_longest_path_{tissue}"))

    return sample

def get_length_longestpath(sample: RootSample, tissue: str) -> RootSample:
    """Calculate the length of the longest path using the mask of that path."""

    # Calculate length of longest path using the mask of that path
    mask_longest_path = getattr(sample, f"mask_longest_path_{tissue}")
    setattr(sample, f"{tissue}_length_pixels", get_length_segment(mask_longest_path))

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
    
def plot_original_and_length(sample):
    """Plot the original plant mask, and the longest root branch on top."""
    
    fig, axs = plt.subplots(1, 2)

    # Show original/root mask
    axs[0].imshow(sample.plant_mask, cmap=plutils.cmap_plantclasses)

    # Overlay the skeleton, colored in dark grey
    axs[0].imshow(sample.root_skeleton, cmap=ListedColormap(['none', 'blue']),
            alpha=(sample.root_skeleton>0)*1.0)
        # plt.imshow(sample.root_mask); plt.imshow(sample.root_skeleton, cmap=ListedColormap(['none', '#cccccc']))
    # Overlay the longest path, colored in red
    axs[0].imshow(sample.mask_longest_path_root, cmap=ListedColormap(['none', 'red']),
              alpha=(sample.mask_longest_path_root>0)*1.0)    
    
    # Now same but for the root
    r0, c0, r1, c1 = return_bbox_foreground(sample.root_mask)
    axs[1].imshow(sample.root_mask[r0:r1, c0:c1], cmap=ListedColormap(['black', plutils.custom_colors_plantclasses[2]]))
    axs[1].imshow(sample.root_skeleton[r0:r1, c0:c1], cmap=ListedColormap(['none', 'blue']),
                alpha=(sample.root_skeleton[r0:r1, c0:c1] > 0) * 1.0)
    axs[1].imshow(sample.mask_longest_path_root[r0:r1, c0:c1], cmap=ListedColormap(["none", "red"]),
                alpha=(sample.mask_longest_path_root[r0:r1, c0:c1] > 0) * 1.0)
    
    # Cosmetics
    axs[0].axis("off")
    axs[1].axis("off")
    fig.suptitle(f"Estimated root length: {sample.root_length_pixels:.2f} px")

    return fig, axs

def plot_distance_graph(sample):
    """ Show the distance graph alongside the labeled segments (root only)."""
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs  # keep compatibility with existing `return fig, ax`

    # Left panel: labeled segments, zoomed to root region only
    labeled = sample.root_labeled_segments
    if sample.root_mask is not None and np.any(sample.root_mask > 0):
        r0, c0, r1, c1 = return_bbox_foreground(sample.root_mask)
        labeled_view = labeled[r0:r1, c0:c1]
    else:
        r0, c0 = 0, 0
        labeled_view = labeled

    im = axs[0].imshow(labeled_view, cmap="nipy_spectral")
    axs[0].set_title("Labeled segments (root ROI)")
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
            sample.root_segment_graph.nodes[n].get("length", 1)
            for n in sample.root_segment_graph.nodes
        ],
    )
    nx.draw(
        sample.root_segment_graph,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=node_sizes * 10,
        ax=axs[1],
    )
    axs[1].set_title("Connectivity Graph")
    axs[1].axis("off")
    
    # suptitle with labels of longest path
    longest_path_labels = sample.root_longest_path if sample.root_longest_path else []
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

    for idx, (result, color) in enumerate(zip(plant_results, colors), start=1):
        
        # idx = 0; result = plant_results[idx]; color = colors[idx]
        
        # show the bbox
        minr, minc, maxr, maxc = result.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

        # Project root skeleton pixels back to full-image coordinates
        if result.root_skeleton is not None and np.any(result.root_skeleton):
            ax.imshow(
            result.root_skeleton,
            cmap=ListedColormap(["none", "gray"]),
            alpha=(result.root_skeleton > 0) * 1.0,
            interpolation="none",
            extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Project root longest-path pixels back to full-image coordinates
        if result.mask_longest_path_root is not None and np.any(result.mask_longest_path_root):
            ax.imshow(
                result.mask_longest_path_root,
                cmap=ListedColormap(["none", "red"]),
                alpha=(result.mask_longest_path_root > 0) * 1.0,
                interpolation="none",
                extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Project shoot longest-path pixels back to full-image coordinates
        if result.mask_longest_path_shoot is not None and np.any(result.mask_longest_path_shoot):
            ax.imshow(
                result.mask_longest_path_shoot,
                cmap=ListedColormap(["none", "orange"]),
                alpha=(result.mask_longest_path_shoot > 0) * 1.0,
                interpolation="none",
                extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Optional length label near each bbox (root + shoot)
        if (result.root_length_pixels is not None
                or result.shoot_length_pixels is not None):
            root_txt = (f"{result.root_length_pixels:.1f}"
                        if result.root_length_pixels is not None else "n/a")
            shoot_txt = (f"{result.shoot_length_pixels:.1f}"
                         if result.shoot_length_pixels is not None else "n/a")
            ax.text(
            minc,
            minr - 3,
            f"({idx}) root {root_txt}px / shoot {shoot_txt}px",
            color=color,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
            )
    
    # reset zoom to full image
    ax.set_xlim(0, sample_image.shape[1])
        
    plt.tight_layout()
    return fig, ax

# %% runner

def run_default_length_pipeline(sample: RootSample) -> RootSample:
    """Run the full default sequence of processing steps for root AND shoot."""

    # Run the same pipeline for both tissues
    for tissue in ("root", "shoot"):

        # Make binary and select largest ROI to analyze
        sample = ensure_binary_mask(sample, tissue)
        sample = keep_largest_connected_component(sample, tissue)

        # Generate a labeled skeleton to analyze
        sample = generate_skeleton_no_branchpoints(sample, tissue)
        sample = label_skeleton_segments(sample, tissue)

        # Build a graph, and find the longest path
        sample = build_segment_graph(sample, tissue)
            # plot_distance_graph(sample)
        sample = find_start_label_close_to_other(sample, tissue)
        sample = get_long_path_in_graph_nodearea(sample, tissue)
        sample = build_longest_path_mask(sample, tissue)
            # plot_original_and_length(sample)
        sample = get_length_longestpath(sample, tissue)

        # Add distance in mm (if possible)
        if sample.pixel_size_mm is not None:
            setattr(sample, f"{tissue}_length_mm",
                    getattr(sample, f"{tissue}_length_pixels") * sample.pixel_size_mm)
        else:
            setattr(sample, f"{tissue}_length_mm", None)

    return sample
