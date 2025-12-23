
#%% 

# 
import sys
dir_output_exploratory = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/exploratory/"
sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/_Projects-bioDSC/2025_Root-length')
import tifffile as tiff

#%% libs #######################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import morphology
from skimage.measure import regionprops
import networkx as nx
# from skan import Skeleton, summarize
from scipy.ndimage import convolve

# custom libs
import custom_functions.remove_large_objects as cflo
    # import importlib; importlib.reload(cflo)

# generate a randomly shuffled rainbow categorical color map
colors_rainbow = plt.cm.rainbow(np.linspace(0, 1, 255))
np.random.shuffle(colors_rainbow)
colors_rainbow = np.vstack(([0, 0, 0, 1], colors_rainbow))
cmap_random_rainbow = ListedColormap(colors_rainbow)

# Let's set a custom color scheme
custom_colors_plantclasses = \
    [   # background black
        '#000000', 
        # shoot light green
        '#90EE90',
        # root white
        '#FFFFFF', 
        # seed brown
        '#A52A2A', 
        # leaf darkgreen
        '#006400' 
        ]
# Create a custom cmap

cmap_plantclasses = ListedColormap(custom_colors_plantclasses)

cm_to_inch = 1/2.54

#%% ############################################################################

# assuming we have some input
img_mask = \
    np.load("/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/ANALYSIS/202510/plots/fullimages_predictions_all/predictedmask_idx001.npy")

# now plot
plt.imshow(img_mask, cmap=cmap_plantclasses)

# binary mask
img_mask_binary = img_mask>0
plt.imshow(img_mask_binary)

# Let's say a typical plant is ±400 pixels high, and 5 pixels wide. That's
# 2000 px area;
TYPICAL_PLANT_SIZE = 2000
MIN_SIZE = TYPICAL_PLANT_SIZE/10
MAX_SIZE = TYPICAL_PLANT_SIZE*10
# remove small objects from the mask
img_mask_cleaned = morphology.remove_small_objects(img_mask_binary, min_size=MIN_SIZE)
# remove large objects from the mask
img_mask_cleaned = cflo.remove_large_objects(img_mask_cleaned, max_size=MAX_SIZE)
plt.imshow(img_mask_cleaned)

# now clean the original mask with labels
img_mask[~img_mask_cleaned] = 0
plt.imshow(img_mask, cmap=cmap_plantclasses)


# %% Now let's look whether I can perform some skeletonization

# First, isolate the root segments
img_mask_roots = img_mask==2
plt.imshow(img_mask_roots)

# again, remove small parts
mask_roots_cleaned = morphology.remove_small_objects(img_mask_roots, min_size=MIN_SIZE)
plt.imshow(mask_roots_cleaned)

# now let's get labeled map and regionprops
label_img_cleanroots = morphology.label(mask_roots_cleaned)
regions_cleanroots = regionprops(label_img_cleanroots)

# now, isolate the first root bbox imgage
first_root = regions_cleanroots[0]
minr, minc, maxr, maxc = first_root.bbox
mask_firstroot = mask_roots_cleaned[minr:maxr, minc:maxc]

plt.imshow(mask_firstroot)

mask_firstroot_realplant = mask_firstroot




# %% Now comes the challenge

# I want to be able to deal with curvy roots
# I want to ignore side-branches
# Probably I want to fit a spline locally
# But how do I determine what the main direction of the root is?
    # --> perhpas some opening/closing to determine the main path
    # nevertheless, if there are "real" side branches, how would I be able to
    # remove those?
    # 
    # Maybe also create an idealized root mask, with some challenges in it,
    # to see if I can deal with that..

testmask_file_path = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/idealized_root_images/root_mask_1.tif"
testmask_file_path = "/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/idealized_root_images/root_mask_2.tif"
# Load it
mask_firstroot = plt.imread(testmask_file_path)

mask_firstroot = mask_firstroot_realplant

# now skeletonize this
skeleton_firstroot = morphology.skeletonize(mask_firstroot)

# side-to-side comparison of mask_firstroot and skeleton_firstroot
fig, axs = plt.subplots(1, 2)
axs[0].imshow(mask_firstroot)
axs[1].imshow(skeleton_firstroot)

# save the skeleton to a tiff image
tiff.imwrite(dir_output_exploratory + "skeleton_firstroot.tif", 
            skeleton_firstroot.astype(np.uint8)*255)



# Let's experiment with simple branching structure

# First let's remove all pixels that touch more than 2 other pixels
# Define a kernel that sums the 8 neighbors
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# Calculate the number of neighbors for each pixel
# We convert the boolean skeleton to integer (0/1) for the convolution
neighbor_counts = convolve(skeleton_firstroot.astype(int), kernel, mode='constant', cval=0)

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
%matplotlib qt
plt.show()

# now convert the mask with no branchpoints to labeled mask
labeled_skeleton_no_branchpoints = morphology.label(skeleton_no_branchpoints)
max_label = labeled_skeleton_no_branchpoints.max()
# now loop over the branchpoint-coordinates (both end and branchpoint nodes)
# and labels, starting at max_label+1, to each pixel
for idx, coord in enumerate(np.vstack((branchpoint_coords, endpoint_coords))):
    labeled_skeleton_no_branchpoints[coord[0], coord[1]] = idx + max_label + 1

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


plt.figure()
plt.imshow(labeled_skeleton_no_branchpoints, cmap=cmap_random_rainbow)
for lbl, coord in zip(labels, zip(rows, cols)):
    plt.text(coord[1], coord[0], str(lbl), color='black', fontsize=6, ha='center', va='center')
%matplotlib qt
plt.show()


# this has worked neatly, now, loop over each of the labels, and gather any 
# label that is in the direct neighborhood of any of these pixels
# Initialize adjacency dictionary
adjacency = {}
sizes     = {}

# Get all unique labels excluding background (0)
unique_labels = np.unique(labeled_skeleton_no_branchpoints)
unique_labels = unique_labels[unique_labels != 0]

# Define a 3x3 structure for 8-connectivity
struct = morphology.square(3)

print("Calculating adjacency...")
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
    
    # Store in dictionary
    adjacency[label_id] = neighbor_labels.tolist()

# Print the results
print("Adjacency list:")
for label, neighbors in adjacency.items():
    print(f"Label {label} connects to: {neighbors}")


# now convert the adjacency to a networkx graph
G = nx.Graph()
G.add_nodes_from(adjacency.keys())

for node, neighbors in adjacency.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Visualize the graph abstractly
plt.figure()
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Connectivity Graph")
plt.show()




# %%

def skeleton_to_graph(skeleton_img):
    """
    Converts a binary skeletonized image into a NetworkX graph.
    
    Nodes correspond to branch points or endpoints.
    Edges correspond to the segments connecting them.
    Edge weights ('weight') represent the number of pixels in the segment (Euclidean length is also available in skan).
    
    Parameters:
    -----------
    skeleton_img : np.ndarray
        Binary skeletonized image.
        
    Returns:
    --------
    G : networkx.MultiGraph
        Graph representation of the skeleton.
    """
    
    # Use skan to analyze the skeleton
    skel = Skeleton(skeleton_img)
    
    # Create a graph
    # skan provides a summary which is very useful, but we can also build a graph directly
    # The skan.Skeleton object already contains a graph representation in CSR format, 
    # but let's convert it to a more standard NetworkX graph for easier manipulation.
    
    # Get the summary dataframe which lists all branches (edges)
    branch_data = summarize(skel)
    
    G = nx.MultiGraph()
    
    # Iterate over the branches identified by skan
    for index, row in branch_data.iterrows():
        # node IDs in skan are based on the flattened index of the pixel in the image
        src_node = int(row['node-id-src'])
        dst_node = int(row['node-id-dst'])
        
        # branch-distance is the euclidean length, branch-type tells us about the topology
        # We can also use 'image-coord-src-0', 'image-coord-src-1' for coordinates if needed
        
        # The number of pixels is roughly the branch distance, but skan calculates 
        # Euclidean distance along the path. If you strictly want pixel count:
        # The skan path coordinates are available via skel.path_coordinates(index)
        path_coords = skel.path_coordinates(index)
        pixel_count = path_coords.shape[0]
        
        # Add edge with attributes
        G.add_edge(src_node, dst_node, 
                   weight=pixel_count, 
                   euclidean_length=row['branch-distance'],
                   branch_id=index)
        
        # Add node attributes (coordinates)
        # We only need to do this once per node, but overwriting is fine
        src_coords = (row['image-coord-src-0'], row['image-coord-src-1'])
        dst_coords = (row['image-coord-dst-0'], row['image-coord-dst-1'])
        
        G.nodes[src_node]['pos'] = src_coords
        G.nodes[dst_node]['pos'] = dst_coords
        
    return G, branch_data

# Example usage:
# G, df = skeleton_to_graph(skeleton_firstroot)
# print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")


