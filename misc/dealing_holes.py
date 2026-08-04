

################################################################################
# %% Libs

import imageio as iio
import os 
import matplotlib.pyplot as plt

# load closing image operation
from skimage.morphology import closing, disk

################################################################################
# %% Load example image

img_test = iio.imread("../example_files/idealized_root_masks/root_with_hole.tif")


################################################################################
# %%

# show example image
plt.imshow(img_test)
plt.show()

# now expand using 
img_test_closed = closing(img_test, disk(25))
plt.imshow(img_test_closed)


# %%
