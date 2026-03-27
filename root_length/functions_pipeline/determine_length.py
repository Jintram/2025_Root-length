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

import functions_pipeline.utils as plutils

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
    """Container for one plant/root sample during processing."""

    root_mask: np.ndarray
    shoot_mask: np.ndarray
    plant_mask: np.ndarray | None = None
    pixel_size_mm: float | None = None

    clean_root_mask: np.ndarray | None = None
    root_skeleton: np.ndarray | None = None
    root_skeleton_nobranchpoints: np.ndarray | None = None
    branchpoint_coords: np.ndarray | None = None
    endpoint_coords: np.ndarray | None = None

    labeled_segments: np.ndarray | None = None
    segment_graph: nx.Graph | None = None

    start_label: int | None = None
    longest_path: list[int] | None = None
    length_pixels: float | None = None
    length_mm: float | None = None
    
    # position of the bbox in the original image
    bbox: tuple[int, int, int, int] | None = None
    
    mask_longest_root_path: np.ndarray | None = None


################################################################################
# %% Basic preprocessing


def ensure_binary_root_mask(sample: RootSample) -> RootSample:
    """ Simply convert root_mask to boolean and return. """
    
    # Convert to boolean
    sample.root_mask = sample.root_mask.astype(bool)
    
    return sample

def keep_largest_connected_root_component(sample: RootSample) -> RootSample:
    """
    Keep only the largest root object in the root mask.
    
    In principle, an ideal mask only contains one root ROI. However, it might
    occur that there are other parts of the plants labeled as root, that are
    not connected to the main root area. 
    In this case, we want to focus on the main root area. Therefor, this
    function analyzes the root areas, and retains the largest region only.
    
    This is then stored in sample.clean_root_mask.
    """
    
    # Create labeled mask and props for the *root* mask
    labeled = label(sample.root_mask)
    props = regionprops(labeled)
    
    # In case there's no root at all
    if not props:
        sample.clean_root_mask = np.zeros_like(sample.root_mask, dtype=bool)
        return sample

    # obtain the region properties element corresponding to the largest region
    def get_region_area(region):
        return region.area
    largest_region = max(props, key=get_region_area)
    
    # Now create a new mask, corresponding to the largest region
    sample.clean_root_mask = (labeled == largest_region.label)    
        # plt.imshow(sample.clean_root_mask)
    
    return sample

################################################################################
# %% Branch analysis

def generate_root_skeleton_no_branchpoints(sample: RootSample) -> RootSample:
    """Skeletonize root mask and remove branch-point pixels from that skeleton."""

    # Obtain the skeleton
    sample.root_skeleton = morphology.skeletonize(sample.clean_root_mask)

    # Create an equal-sized array that gives the neighbor count for each pixel 
    # in root_skeleton.
    neighbor_counts = convolve(
        sample.root_skeleton.astype(int),
        KERNEL_NEIGHBOR_COUNT,
        mode="constant",
        cval=0,
    )

    # Now only keep parts of the skeleton that have 2 neighbors
    sample.root_skeleton_nobranchpoints = \
        sample.root_skeleton & (neighbor_counts <= 2)
        # plt.imshow(sample.root_skeleton_nobranchpoints)
        
    # and collect the x,y locations of both the branch points as 
    # well as the end points.
    # Locations of branch points
    sample.branchpoint_coords = np.column_stack(
        np.where(sample.root_skeleton & (neighbor_counts > 2))
    )
    # Locations of end points
    sample.endpoint_coords = np.column_stack(
        np.where(sample.root_skeleton & (neighbor_counts == 1))
    )
    
    return sample


def label_skeleton_segments(sample: RootSample) -> RootSample:
    """Label line segments in skeleton and assign separate labels to nodes."""

    # Now get the labeled skeleton
    labeled_segments = morphology.label(sample.root_skeleton_nobranchpoints)
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


def build_segment_graph(sample: RootSample) -> RootSample:
    """
    Create a graph where each segment label is one node.
    
    Note that there is a small imprecision here, as branch point pixels
    have length = 0.5. (This could lead to the longest path that will be 
    identified later not actually being the longest, in extreme edge cases.
    The root length calculated later will include these pixels, as then the
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


def find_start_label_close_to_shoot(sample: RootSample) -> RootSample:
    """Pick segment label nearest to `shoot_mask` (if provided)."""

    # now get distance map to shoot
    distance_map = distance_transform_edt(~sample.shoot_mask.astype(bool))
    # disregard background pixels (set to inf distance)
    distance_map[sample.labeled_segments == 0] = np.inf

    # and find the root pixel that is closest to shoot
    closest_pixel = np.unravel_index(np.argmin(distance_map), distance_map.shape)
    # and its corresponding label
    sample.start_label = int(sample.labeled_segments[closest_pixel])
    
    return sample


def helper_print_graph_node_lengths(graph):
    """Debug fn, print lengths of each of the nodes"""
    for n in graph.nodes:
        print(f"Node {n}: length {graph.nodes[n].get('length', 'N/A')}")

def get_long_path_in_graph_nodearea(sample: RootSample) -> RootSample:
    """Find a long path by maximizing sum of node areas along shortest paths."""

    graph = sample.segment_graph
    
    # If empty graph, simply return
    if graph.number_of_nodes() == 0:
        sample.longest_path = []
        sample.length_pixels = 0.0
        return sample

    # The start_label should contain the starting point closest to the shoot
    # (this is required, because there might be a longest path not touching 
    # the shoot), so we want to select that as starting node.
    source_nodes = [sample.start_label] if sample.start_label in graph else list(graph.nodes)

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
    sample.longest_path = longest_path
    sample.length_pixels = max_length
    return sample

def build_longest_path_mask(sample: RootSample) -> RootSample:
    """Create a binary mask of the longest root path."""

    # Create a new mask based on the labeled root mask, which only retains pixels
    # that are the longest path.
    sample.mask_longest_root_path = \
        np.isin(sample.labeled_segments, sample.longest_path)
    # plt.imshow(sample.mask_longest_root_path)

    return sample

def get_length_longestpath(sample: RootSample) -> RootSample:
    """Calculate the length of the longest path using the mask of that path."""

    # Calculate length of longest path using the mask of that path
    sample.length_pixels = get_length_segment(sample.mask_longest_root_path)
    
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
    """Plot the original plant mask, and the longest branch on top"""
    
    fig, axs = plt.subplots(1, 2)

    # Show original/root mask
    axs[0].imshow(sample.plant_mask, cmap=plutils.cmap_plantclasses)

    # Overlay the skeleton, colored in dark grey
    axs[0].imshow(sample.root_skeleton, cmap=ListedColormap(['none', 'blue']),
            alpha=(sample.root_skeleton>0)*1.0)
        # plt.imshow(sample.root_mask); plt.imshow(sample.root_skeleton, cmap=ListedColormap(['none', '#cccccc']))
    # Overlay the longest path, colored in red
    axs[0].imshow(sample.mask_longest_root_path, cmap=ListedColormap(['none', 'red']),
              alpha=(sample.mask_longest_root_path>0)*1.0)    
    
    # Now same but for the root
    r0, c0, r1, c1 = return_bbox_foreground(sample.root_mask)
    axs[1].imshow(sample.root_mask[r0:r1, c0:c1], cmap=ListedColormap(['black', plutils.custom_colors_plantclasses[2]]))
    axs[1].imshow(sample.root_skeleton[r0:r1, c0:c1], cmap=ListedColormap(['none', 'blue']),
                alpha=(sample.root_skeleton[r0:r1, c0:c1] > 0) * 1.0)
    axs[1].imshow(sample.mask_longest_root_path[r0:r1, c0:c1], cmap=ListedColormap(["none", "red"]),
                alpha=(sample.mask_longest_root_path[r0:r1, c0:c1] > 0) * 1.0)
    
    # Cosmetics
    axs[0].axis("off")
    axs[1].axis("off")
    fig.suptitle(f"Estimated root length: {sample.length_pixels:.2f} px")

    return fig, axs

def plot_distance_graph(sample):
    """ Show the distance graph alongside the labeled segments"""
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs  # keep compatibility with existing `return fig, ax`

    # Left panel: labeled segments, zoomed to root region only
    labeled = sample.labeled_segments
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

    for idx, (result, color) in enumerate(zip(plant_results, colors), start=1):
        
        # idx = 0; result = plant_results[idx]; color = colors[idx]
        
        # show the bbox
        minr, minc, maxr, maxc = result.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

        # Project skeleton pixels back to full-image coordinates
        if result.root_skeleton is not None and np.any(result.root_skeleton):
            ax.imshow(
            result.root_skeleton,
            cmap=ListedColormap(["none", "gray"]),
            alpha=(result.root_skeleton > 0) * 1.0,
            interpolation="none",
            extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Project longest-path pixels back to full-image coordinates
        if result.mask_longest_root_path is not None and np.any(result.mask_longest_root_path):
            ax.imshow(
                result.mask_longest_root_path,
                cmap=ListedColormap(["none", "red"]),
                alpha=(result.mask_longest_root_path > 0) * 1.0,
                interpolation="none",
                extent=(minc, maxc, maxr, minr),  # project ROI back to full image
            )

        # Optional length label near each bbox
        if result.length_pixels is not None:
            ax.text(
            minc,
            minr - 3,
            f"({idx}) {result.length_pixels:.1f}px",
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
    """Run the full default sequence of novice-friendly processing steps."""
    
    # Make binary and select largest root ROI to analyze
    sample = ensure_binary_root_mask(sample)
    sample = keep_largest_connected_root_component(sample)

    # Generate a labeled skeleton to analyze
    sample = generate_root_skeleton_no_branchpoints(sample)
    sample = label_skeleton_segments(sample)

    # Build a graph, and find the longest path
    sample = build_segment_graph(sample)
        # plot_distance_graph(sample)
    sample = find_start_label_close_to_shoot(sample)
    sample = get_long_path_in_graph_nodearea(sample)
    sample = build_longest_path_mask(sample)
        # plot_original_and_length(sample)
    sample = get_length_longestpath(sample)
    
    # Add distance in mm (if possible)
    if sample.pixel_size_mm is not None:
        sample.length_mm = sample.length_pixels * sample.pixel_size_mm
    else:
        sample.length_mm = None

    return sample
