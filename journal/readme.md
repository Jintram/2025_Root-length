


## 2026-06-17

#### Plan

First run the segmentation pipeline using the new model.

#### Things to do

- The current pre-processing is inconvenient, as there is a specific function
being fed that pre-processes the image for cropping, and the cropped image
won't be saved.
     - Allthough this takes more space, it might be better to first segment
     the images, and then during post-processing algorithmically -- on top
     of the ML segmentation -- identify a background region that will be
     cleared.
     
     
- Perhaps a function that allows 'automatic' updating of root/shoot boundary,
bound in napari, where all "shoot" annotation that falls below the highest
"root" annotation, is automatically converted to root, would be highly
useful.


- Many images are quite well segmented, but many are also more poorly segmented.
In some cases, the previous model was even better at identifying the roots/shoots,
even for the higher resolution cases.
    - A reason for this might be that I've now included more background, ie
    the outline and background of the plate. This might have resulted in 
    the issue that recognizing the plate outline and background now makes
    it more difficult to discern the plants themselves.     
    - A solution for this is to split the pipeline again,
        (1) find "plate" masks of the images. pre-process all images 
        such that only cropped versions remain.
            - Previously, I already had a filter that worked pretty well 
            to pre-process images. But i think it can be further improved.
            - Probably the best way to store this is to store a mask somewhere
            such that data is not duplicated;
                this is already partially in place with the "crop_rect" approach.
        (2) that cropped data set is used both for training of the model
        as well as for segmentation. 
        
        
- additionally, currently, we used two separate training sets for the model;
these two training sets could be combined.