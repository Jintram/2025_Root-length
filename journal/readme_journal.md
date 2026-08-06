


## 2026-06-17

#### Plan

First run the segmentation pipeline using the new model.

#### Things to do

- [ ] The current pre-processing is inconvenient, as there is a specific function
being fed that pre-processes the image for cropping, and the cropped image
won't be saved.
     - Allthough this takes more space, it might be better to first segment
     the images, and then during post-processing algorithmically -- on top
     of the ML segmentation -- identify a background region that will be
     cleared.
     
     
- [ ] Perhaps a function that allows 'automatic' updating of root/shoot boundary,
bound in napari, where all "shoot" annotation that falls below the highest
"root" annotation, is automatically converted to root, would be highly
useful.


- [ ] Many images are quite well segmented, but many are also more poorly segmented.
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

## 2026-06-23

- [ ] Combining root & shoot skeletons
    - Currently, root and shoot are analyzed independently. This causes
    the "length line" to be often not connected between the root and shoot.
    This could be avoided by calculating the skeleton for both the root and shoot
    area at once (assuming they are connected -- allthough might work
    if they aren't), and then select skeleton based on the respective root and
    shoot-masks for root and shoot specific skeletons.
        - Challenge: how to deal with when there's a seed that isolates the two
        separate regions?
            - perhaps using a strong closing on the selected regions does the trick, see "misc/dealing_holes.py".
    
## 2026-08-04

- [ ] See previous entries
    - (better skeleton at boundaries)
    - (cropping)

**TO DO NOW:**
- [ ] Implement way to combine multiple labels and distill separate skeletons 
afterwards.
    - Keep closing operation in mind.
    
##### Notes regarding root/shoot joined area analysis

*I currently have a pipeline where root and shoot masks are determined from the labeled mask per plant, and the length is determined based on the skeleton of either root or shoot, without sharing knowledge between the two masks. I'd like to change this as follows;
the root and shoot mask should be combined, such and a shared skeleton should be determined. then, to determine the root and shoot specific skeletons, the relevant (root/shoot) mask is simply applied to the skeleton.*

- Add a "fix_holes()" function
- Would easiest solution not simply be to create a joined mask and store that
in the tissue class?
- things to check: how is longest length restricted to 

##### Optional features later;

- [ ] Show the crop-recteangle in the Napari viewer (and allow it to be edited)

##### Current test case: 

- projects/20260621_highresmodel-crop/202606_highrescropmodel_batch1_2lengths.py