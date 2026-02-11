

# Functions that allow length calculation for an individual plant
# This assumes you provide an image that contains only one 
# individual plant segmentation.

################################################################################
#%% libs 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import morphology
from skimage.measure import regionprops
import networkx as nx
# from skan import Skeleton, summarize
from scipy.ndimage import convolve, distance_transform_edt


# custom libs
import custom_functions.remove_large_objects as cflo
    # import importlib; importlib.reload(cflo)


    
################################################################################
#%% functions to determine length

# remember to only take largest area into account
 
    

# now skeletonize this
skeleton_firstroot = morphology.skeletonize(mask_firstroot)

# side-to-side comparison of mask_firstroot and skeleton_firstroot
# %matplotlib inline
fig, axs = plt.subplots(1, 2)
axs[0].imshow(mask_firstroot)
axs[1].imshow(skeleton_firstroot)
# plt.show()

# save the skeleton to a tiff image
# tiff.imwrite(dir_output_exploratory + "skeleton_firstroot.tif",
#            skeleton_firstroot.astype(np.uint8)*255)



# Let's experiment with simple branching structure

# First let's remove all pixels that touch more than 2 other pixels
# Define a kernel that sums the 8 neighbors
KERNEL_NEIGHBOR_COUNT = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]])

# Calculate the number of neighbors for each pixel
# We convert the boolean skeleton to integer (0/1) for the convolution
neighbor_counts = convolve(skeleton_firstroot.astype(int), KERNEL_NEIGHBOR_COUNT, mode='constant', cval=0)

# Get the coordinates of pixels with >2 neighbor count
branchpoint_coords = np.column_stack(np.where((skeleton_firstroot) & (neighbor_counts > 2)))

# Now get the coordinates of end-points (1 neighbor)
endpoint_coords = np.column_stack(np.where((skeleton_firstroot) & (neighbor_counts == 1)))

# Create a new mask where we keep skeleton pixels that have 2 or fewer neighbors
# This effectively removes the branch points (junctions)
skeleton_no_branchpoints = skeleton_firstroot & (neighbor_counts <= 2)

# Plot the result
plt.figure()
plt.imshow(skeleton_no_branchpoints)
plt.title("Skeleton with branch points removed")
# plot nodes on top, use large open spheres of multipel colors
plt.plot(branchpoint_coords[:, 1], branchpoint_coords[:, 0], 'ro', markersize=8, markerfacecolor='none')
plt.plot(endpoint_coords[:, 1], endpoint_coords[:, 0], 'wo', markersize=8, markerfacecolor='none')
# and save
plt.savefig(dir_output_exploratory + "skeleton_no_branchpoints.png", dpi=300)

%matplotlib qt
plt.show()

################################################################################

# NOTE ADDED LATER: MAYBE TRY THIS FIRST WITHOUT SEPARATELY LABELING THE 
# ENDPOINT COORDINATES? NOT SURE WHAT I NEED THOSE FOR.. 
# now convert the mask with no branchpoints to labeled mask
labeled_skeleton_no_branchpoints = morphology.label(skeleton_no_branchpoints)
max_label = labeled_skeleton_no_branchpoints.max()
# now loop over the branchpoint-coordinates (both end and branchpoint nodes)
# and labels, starting at max_label+1, to each pixel
for idx, coord in enumerate(np.vstack((branchpoint_coords, endpoint_coords))):
    labeled_skeleton_no_branchpoints[coord[0], coord[1]] = idx + max_label + 1

################################################################################

# for plotting purposes
# Get coordinates of all non-zero pixels
rows, cols = np.nonzero(labeled_skeleton_no_branchpoints)
# Get the labels at these coordinates
labels = labeled_skeleton_no_branchpoints[rows, cols]

# now plot the new mask
myratio=labeled_skeleton_no_branchpoints.shape[0]/labeled_skeleton_no_branchpoints.shape[1]
plt.figure(figsize=(3*cm_to_inch, 3*myratio*cm_to_inch))
plt.imshow(labeled_skeleton_no_branchpoints, cmap=cmap_random_rainbow)
# remove axes
plt.axis("off")
# save the figure 
plt.tight_layout()
plt.savefig(dir_output_exploratory + "labeled_skeleton_no_branchpoints.png", dpi=300)

%matplotlib qt
plt.show()

plt.figure(figsize=(0.08*labeled_skeleton_no_branchpoints.shape[0]*cm_to_inch,
                    0.08*labeled_skeleton_no_branchpoints.shape[1]*myratio*cm_to_inch))
# plt.figure(figsize=(12*cm_to_inch, 12*myratio*cm_to_inch))
plt.imshow(labeled_skeleton_no_branchpoints, cmap=cmap_random_rainbow)
for lbl, coord in zip(labels, zip(rows, cols)):
    plt.text(coord[1], coord[0], str(lbl), color='black', fontsize=0.5, ha='center', va='center')
# and save
plt.savefig(dir_output_exploratory + "labeled_skeleton_no_branchpoints_with_labels.pdf", dpi=1200)

%matplotlib qt
plt.show()

################################################################################


# %% 
# this has worked neatly, now, loop over each of the labels, and gather any 
# label that is in the direct neighborhood of any of these pixels

# Initialize networkx graph
G = nx.Graph()

# Get all unique labels excluding background (0)
unique_labels = np.unique(labeled_skeleton_no_branchpoints)
unique_labels = unique_labels[unique_labels != 0]

# Add nodes to the graph
G.add_nodes_from(unique_labels)

# Define a 3x3 structure for 8-connectivity
struct = morphology.square(3)

print("Calculating adjacency and building graph...")
for label_id in unique_labels:
    # Create a binary mask for the current label
    current_mask = labeled_skeleton_no_branchpoints == label_id
    
    # Dilate the mask to find neighbors
    # This expands the current label's area by 1 pixel in all directions
    dilated_mask = morphology.binary_dilation(current_mask, struct)
    
    # Extract the labels from the original image that fall under the dilated mask
    neighboring_pixels = labeled_skeleton_no_branchpoints[dilated_mask]
    
    # Find unique labels in the neighborhood
    neighbor_labels = np.unique(neighboring_pixels)
    
    # Filter out background (0) and the label itself
    neighbor_labels = neighbor_labels[(neighbor_labels != 0) & (neighbor_labels != label_id)]
    
    # Add edges to the graph directly
    for neighbor in neighbor_labels:
        G.add_edge(label_id, neighbor)
    # Now also add the area of the segment as node attribute
    segment_area = np.sum(current_mask)
    G.nodes[label_id]['area'] = segment_area

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

################################################################################

# Visualize the graph abstractly
def visualize_graph_nodesize(G):
    fig, ax = plt.subplots()
    node_sizes = np.array([G.nodes[n]['area'] for n in G.nodes])
    nx.draw(G, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=node_sizes*10)
    ax.set_title("Connectivity Graph")
    
    # plt.show()
    return fig, ax

fig, ax = visualize_graph_nodesize(G)
fig.savefig(dir_output_exploratory + "connectivity_graph_nodesize.png", dpi=300)

################################################################################

# Now simplify the graph
# NOTE: I'm reviewing the code, and should check why this is necessary. 
# Is this just to remove the nodes that are connected to end points? Why
# are end points necessary? Perhaps I should work with the mask that 
# doesn't have them?
def simplify_graph_by_contracting_degree2_nodes(G):
        
    G_simple = G.copy()
    nodes_to_remove = [n for n in G_simple.nodes if G_simple.degree(n) == 2]
    
    # first, set all edges to length 1
    for u, v in G_simple.edges():
        G_simple[u][v]['length'] = 1
    
    for n in nodes_to_remove:
        # get neighbors
        neighbors = list(G_simple.neighbors(n))
        # tidy (remove this node replace by 1 edge) if it has only 2 neighbors
        if len(neighbors) == 2:
            u, v = neighbors
            
            # Now set weight, that will equal the area of the node that was removed
            length = G_simple.nodes[n].get('area', 1)
            
            # weight = 0
            # if G_simple.has_edge(u, n) and G_simple.has_edge(n, v):
            #     weight = G_simple[u][n].get('length', 1) + G_simple[n][v].get('length', 1)
            
            # add the new edge replacing the node
            G_simple.add_edge(u, v, length=length)
            
            # remove the node
            G_simple.remove_node(n)
    
    # now make a list of the area of all nodes
    all_areas = np.array(
        [G_simple.nodes[n]['area'] for n in G_simple.nodes])
    
    # check if they are all 1 
    if np.all(all_areas == 1):
        print("Check passed, all node areas are 1 after simplification.")
    else:
        print("Check failed, not all node areas are 1 after simplification.")
    
    return G_simple

def visualize_graph_edgelength(G):
    
    plt.figure()
    
    node_sizes = np.array([G.nodes[n]['area'] for n in G.nodes])
    
    # Get edge weights
    edge_lengths     = [G[u][v].get('length', 1) for u, v in G.edges()]
    edge_lengths_inv = 1/np.array(edge_lengths)
    
    # now add inverse weights
    for (u, v), inv_l in zip(G.edges(), edge_lengths_inv):
        G[u][v]['inv_l'] = inv_l
        
    # Use edge weights as edge lengths in spring_layout
    pos = nx.spring_layout(G, weight="inv_l")
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', node_size=node_sizes*10)
    
    plt.title("Connectivity Graph (edge length ~ weight)")
    plt.show()

################################################################################

# get the simplified graph
G_simplified = simplify_graph_by_contracting_degree2_nodes(G)
# Now visualize it again
visualize_graph_edgelength(G_simplified)

all_areas = [G_simplified.nodes[n]['area'] for n in G_simplified.nodes]

# Now find the longest possible path with G_simplified
def get_long_path_in_graph_edgelength(G):
    longest_path = []
    max_length = 0
    # check all pairs of nodes
    for source in G.nodes:
        for target in G.nodes:
            if source != target:
                path = nx.shortest_path(G, source=source, target=target, weight='length')
                # Calculate path length
                path_length = sum(G[u][v].get('length') for u, v in zip(path[:-1], path[1:]))
                if path_length > max_length:
                    max_length = path_length
                    longest_path = path
    print(f"Longest path length: {max_length}")
    print(f"Longest path end nodes: {[longest_path[0], longest_path[-1]]}")
    
    return longest_path, max_length

longest_path, max_length = get_long_path_in_graph_edgelength(G_simplified)

################################################################################

def find_branch_close_other(labeled_mask, mask_other):
    """
    A function that will find the segment that's closest to another mask
    This is intended to find the segment of the root that is closest to the 
    shoot, to use as a starting point for the longest path search.
    """
    # mask_other = mask_firstshoot
    # labeled_mask = labeled_skeleton_no_branchpoints

    # Generate a distance map based on the other mask
    distance_map = distance_transform_edt(~mask_other)
    # plt.imshow(distance_map)

    # set points outside the labeled mask to inf
    distance_map[labeled_mask==0] = np.inf
        # plt.imshow(distance_map)
        
    # now get the coordinate of the pixel with the smallest distance
    closest_pixel = np.unravel_index(np.argmin(distance_map), distance_map.shape)    
    # now get the label corresponding to that lowest value
    closest_label = labeled_mask[closest_pixel]
    
    return closest_label

# create a picture with the closest label highlighted
closest_label = find_branch_close_other(labeled_skeleton_no_branchpoints, mask_firstshoot)
highlighted_mask = (labeled_skeleton_no_branchpoints>0).astype(int)
highlighted_mask[labeled_skeleton_no_branchpoints == closest_label] = 2
# %matplotlib qt
# plt.imshow(highlighted_mask, cmap='viridis')

################################################################################

# Now find the longest possible path within the network
def get_long_path_in_graph_nodearea(G, source=None):
    """
    Find the longest path between any two nodes.
    
    TO DO: 
    - OR, find the longest path, starting from any node that touches the shoot.
    """
    
    longest_path = []
    max_length = 0
    
    # if source is provided, only check paths starting from that source node,
    # otherwise, check all pairs of nodes
    if source is not None:
        source_nodes = [source]
    else:
        source_nodes = G.nodes
    
    # check all pairs of nodes
    for source in source_nodes:
        for target in G.nodes:
            if source != target:
                path = nx.shortest_path(G, source=source, target=target, weight='length')
                # Calculate path length
                # path_length = sum(G[u][v].get('length') for u, v in zip(path[:-1], path[1:]))
                # Using the areas of the involved nodes
                path_length = sum(G.nodes[n].get('area', 1) for n in path)
                if path_length > max_length:
                    max_length = path_length
                    longest_path = path
    print(f"Longest path length: {max_length}")
    print(f"Longest path end nodes: {[longest_path[0], longest_path[-1]]}")
    
    return longest_path, max_length

# longest_path, max_length = get_long_path_in_graph_edgelength(G_simplified)
longest_path, max_length = get_long_path_in_graph_nodearea(G, source=closest_label)

# For reference, print G again
visualize_graph_nodesize(G)

################################################################################

# create a mask with the longest path through segments 
skeleton_longest_path = np.zeros_like(labeled_skeleton_no_branchpoints)
for node_id in longest_path:
    skeleton_longest_path[labeled_skeleton_no_branchpoints == node_id] = 1
# plt.imshow(skeleton_longest_path)

################################################################################

# now use the ids in longest_path to highlight that path in the original skeleton image
skeleton_highlighted = skeleton_firstroot*1
skeleton_highlighted[skeleton_longest_path == 1] = 2
plt.figure()
plt.imshow(skeleton_highlighted, cmap='viridis')
plt.title("Longest Path Highlighted in Skeleton")
%matplotlib qt
plt.show()
    
# plt.imshow(labeled_skeleton_no_branchpoints)

################################################################################

# Kernel with distances for 8-connectivity (diagonal neighbors have distance sqrt(2))    
# The distances are divided by 2, because otherwise lengths are counted twice.
DISTANCE_KERNEL = np.array([[np.sqrt(2), 1, np.sqrt(2)],
                            [1, 0, 1],
                            [np.sqrt(2), 1, np.sqrt(2)]]) / 2

    
def get_length_segment(the_mask, distance_kernel = DISTANCE_KERNEL):
    """
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
    
    # first test wether all pixels are indeed only connected to <3 neighbors
    neighbor_counts = convolve(the_mask.astype(int),
                               KERNEL_NEIGHBOR_COUNT, 
                               mode='constant', cval=0)
    if (np.any(neighbor_counts[the_mask>0] > 2) or
        np.any(neighbor_counts[the_mask>0] < 1)):
        raise ValueError("Mask is not a valid line (isolated pixels or connected to >2 neighbors)")

    # now for each pixel, get the total distance to all neighbors
    neighbor_distances = convolve(the_mask.astype(float),
                                  distance_kernel,
                                  mode='constant',
                                  cval=0)
    neighbor_distances[the_mask==0] = 0

    # now sum up the distances for all pixels in the mask
    line_length = np.sum(neighbor_distances[the_mask>0])
    return line_length

# Now calculate the length of the root
length_root = get_length_segment(skeleton_longest_path)
skeleton_area = np.sum(skeleton_longest_path>0)

# plt.imshow(skeleton_firstroot)

# %%
    
# (removed some stuff here)
