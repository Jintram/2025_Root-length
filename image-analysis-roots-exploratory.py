

# %% ################################################################################

# library to load jpg image, using skimage
from skimage import io
from matplotlib import pyplot as plt
from skimage import filters, morphology
from skimage.measure import label, regionprops
import numpy as np
from scipy import stats
from scipy.ndimage import distance_transform_edt


# %% ################################################################################

image_path = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Jiawen_Root-length/DATA/20250806_col0_meoh_D1.jpg'

# read the image
img_roots = io.imread(image_path)

# display the image
plt.imshow(img_roots)
plt.show()

# let's look at a crop first
(x1, x2) = (260, 380)
(y1, y2) = (600, 860)

# crop
img_crop = img_roots[y1:y2, x1:x2, :]

# display, but now first RGB, then separate color channels
fig, axs = plt.subplots(1, 4, figsize=(12, 6))
axs[0].imshow(img_crop)
axs[1].imshow(img_crop[:,:,0], cmap='Reds')
axs[2].imshow(img_crop[:,:,1], cmap='Greens')
axs[3].imshow(img_crop[:,:,2], cmap='Blues')
plt.show()

# select red channel for some further operations (seems high contrast)
img_crop_red   = img_crop[:,:,0]

# Some statistics of image
bg_mode_red = stats.mode(img_crop_red, axis=None).mode
perc5_red   = np.percentile(img_crop_red, 5)
delta_p5_mode = bg_mode_red - perc5_red

# let's see what triangle thresholding does:
thresh_triangle = filters.threshold_triangle(img_crop_red)
img_crop_masktriangle = img_crop_red > thresh_triangle
# let's see what Otsu thresholding does
thresh_otsu = filters.threshold_otsu(img_crop_red)
img_crop_maskotsu = img_crop_red > thresh_otsu
# threshold minimum
thresh_minimum = filters.threshold_minimum(img_crop_red)
img_crop_maskminimum = img_crop_red > thresh_minimum
# let's see what above background signal filter does
thresh_abovebg = bg_mode_red + 10*(delta_p5_mode)
img_crop_abovebg = img_crop_red > thresh_abovebg

# show the histogram of red image
plt.hist(img_crop_red.ravel(), bins=256)
plt.axvline(thresh_abovebg, color='r', linestyle='dashed', linewidth=2)
plt.show()

# show the masks
fig, axs = plt.subplots(1, 4, figsize=(12, 6))
axs[0].imshow(img_crop_masktriangle, cmap='gray')
axs[0].set_title('triangle thresh: %d' % thresh_triangle)
axs[1].imshow(img_crop_maskotsu, cmap='gray')
axs[1].set_title('Otsu thresh: %d' % thresh_otsu)
axs[2].imshow(img_crop_maskminimum, cmap='gray')
axs[2].set_title('minimum thresh: %d' % thresh_minimum)
axs[3].imshow(img_crop_abovebg, cmap='gray')
axs[3].set_title('above bg thresh: %d' % thresh_abovebg)
plt.show()

# %% ################################################################################
# Now try with Otsu on whole image (but cropped version)

path_img2 = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Jiawen_Root-length/DATA/20250806_col0_meoh_D1_CROPMANUAL.tif'

# read the image
img2 = io.imread(path_img2)

# otsu
img2_red = img2[:,:,0]
thresh_otsu2 = filters.threshold_otsu(img2_red)

# mask
img2_maskotsu = img2_red > thresh_otsu2

# show
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.imshow(img2_red, cmap='Reds')
ax.contour(img2_maskotsu, colors='black', linewidths=0.5)
plt.show()

# now perform some polishing
img2_maskotsu_close = morphology.closing(img2_maskotsu, morphology.disk(10)) 
img2_maskotsu_dilate = morphology.dilation(img2_maskotsu_close, morphology.disk(10))

# get areas
mask_areas = [p.area for p in regionprops(label(img2_maskotsu_dilate))]
# histogram
plt.hist(mask_areas, bins=50)
plt.show()

# remove small objects from the dilated mask
img2_maskotsu_dilate_filt = morphology.remove_small_objects(img2_maskotsu_dilate, min_size=3000)

# now skeletonize the dilated mask
img2_maskotsu_dilate_filt_skel = morphology.skeletonize(img2_maskotsu_dilate_filt)

# now calculate a distance mask for the closed mask
distance_map = distance_transform_edt(img2_maskotsu_close)
# mask by skeletons
distance_map_skel = distance_map * img2_maskotsu_dilate_filt_skel

# label the dilated mask
labeled_mask_dil = label(img2_maskotsu_dilate_filt)
rprops_dil = regionprops(labeled_mask_dil)

# we also want a bounding box, and also one that excludes the top part of the plant 
# first obtain all bounding boxes
bboxes_dil = np.array([r.bbox for r in rprops_dil])
# now determine the average height of the bounding boxes
bbox_avg_height = np.mean(bboxes_dil[:,2] - bboxes_dil[:,0])
bbox_l20pctavg   = bbox_avg_height/5
# now create new bounding boxes, with bbox_l20pctavg subtracted from the top
bboxes_dil_adj = bboxes_dil.copy()
bboxes_dil_adj[:,0] = np.minimum(bboxes_dil_adj[:,0]+bbox_l20pctavg, bboxes_dil_adj[:,2]-1)
# now create a new mask based on these bounding boxes
mask_bboxadj = np.zeros_like(img2_red)
for idx, bbox in enumerate(bboxes_dil_adj):
    minr, minc, maxr, maxc = bbox
    mask_bboxadj[minr:maxr, minc:maxc] = idx+1

# for each of the labeled areas, get the corresponding coordinates of the maximum in the distance map
# now go over each plant
root_start_loc = np.zeros((len(rprops_dil), 2), dtype=int)
for idx, region in enumerate(rprops_dil):
    # idx, region = 0, rprops_dil[0]
        
    # get coordinates of pixels in region
    coords = region.coords
    
    # look only in the plant-specific bbox that doesn't contain the top of the plant
    distance_map_skel_notops = distance_map_skel * (mask_bboxadj==(region.label))
    
    # get distance map values at these coordinates
    dist_values = distance_map_skel_notops[coords[:,0], coords[:,1]] 
    
    # obtain index
    max_idx = np.argmax(dist_values)
    
    # now calculate back to coordinates
    max_coord = coords[max_idx, :]
    
    # store the maximum distance as root length
    root_start_loc[idx,:] = max_coord

# now repeat creating a bunch of bboxes, but cut the top at the respective root_start_loc
bboxes_dil_adj2 = bboxes_dil.copy()
for idx, bbox in enumerate(bboxes_dil_adj2):
    # idx, bbox = 0, bboxes_dil_adj2[0]
    # get the root start location for this plant
    rsl = root_start_loc[idx,0]
    # adjust the top of the bbox
    bboxes_dil_adj2[idx,0] = rsl
# and again create a mask
# now create a new mask based on these bounding boxes
mask_bboxadj2 = np.zeros_like(img2_red)
for idx, bbox in enumerate(bboxes_dil_adj2):
    minr, minc, maxr, maxc = bbox
    mask_bboxadj2[minr:maxr, minc:maxc] = idx+1

# now finally calculate the root distances per plant, by using the skeletons in the root-bboxes
mask_skel_roots = img2_maskotsu_dilate_filt_skel * (mask_bboxadj2>0)
# create mask for roots
mask_roots_cut = np.zeros_like(img2_red)
# loop over the original labeled_mask_dil, to select each root, and calculate total
for idx, region in enumerate(rprops_dil):
    current_root_mask = mask_skel_roots * (labeled_mask_dil==(region.label))
    total_root_pixels = np.sum(current_root_mask)
    # save to mask
    mask_roots_cut[current_root_mask>0] = 1

# dilate mask_roots_cut 
mask_roots_cut = morphology.dilation(mask_roots_cut, morphology.disk(1))

# show
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
_=ax.imshow(img2_red, cmap='Reds')
# ax.contour(img2_maskotsu, colors='white', linewidths=0.5)
_=ax.contour(img2_maskotsu_close, colors='black', linewidths=0.5)
_=ax.contour(img2_maskotsu_dilate_filt, colors='green', linewidths=0.5)
#ax.imshow(img2_maskotsu_dilate_filt_skel, cmap='gray', alpha=(img2_maskotsu_dilate_filt_skel>0)*1.0)
_=ax.imshow(distance_map_skel, cmap='viridis', alpha=(img2_maskotsu_dilate_filt_skel>0)*1.0)
_=ax.plot(root_start_loc[:,1], root_start_loc[:,0], 'o', markersize=5, markerfacecolor='none', markeredgecolor='yellow', markeredgewidth=2)
# show the bbox mask
_=ax.contour(mask_bboxadj, colors='blue', linewidths=1.0)
_=ax.contour(mask_bboxadj2, colors='white', linewidths=1.0)
# show the final roots mask (mask_roots_cut)
_=ax.contour(mask_roots_cut, colors='black', linewidths=2.0)
# now add text with information below each plant
for idx, region in enumerate(rprops_dil):
    # idx, region = 0, rprops_dil[0]
    # get bbox
    minr, minc, maxr, maxc = region.bbox
    # calculate center bottom of bbox
    xtext = int((minc + maxc)/2)
    ytext = maxr + 20
    # get root length
    total_root_pixels = np.sum(mask_skel_roots * (labeled_mask_dil==(region.label)))
    # add text
    ax.text(xtext, ytext, f'#{idx+1}\n{total_root_pixels} px', color='black', fontsize=7, ha='center', va='top')
# and show
plt.show()





