


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

- [X] Combining root & shoot skeletons
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

**

CONTINUE HERE

CONTINUE WITH 

[X] (1) test run the napari updates
    [X] (1b) maybe also re-organize the button and text positions..
    [X] (1c) !!! git merge into main!!!! <----------------------------------------------
(2) MAKING A "WALK-THROUGH" OF THE CODE, USING PY FILES IN:
projects/20260621_highresmodel-crop-TESTSET

current 1st file:
20250530_OY_09_all-plants-projected.pdf

[ ] CONTINUE WITH
    - [X] documentation_technical.md (note "TO DO"s)
        - [ ] perform final check on documentation_technical?    
    - [ ] README.md    
        - [ ] Finish walkthrough
        - [ ] Remove/edit old parts regarding technical considerations in readme.md
    - [ ] Check differences between models that were trained, get overview of this!
    - [ ] (youtube movie giving "tour"?)

(previously created some additional featuers in the napari GUI)

#### Suggestions Yuzeng

- [ ] Create two kind of output plots for the length, also have the
overlay plot with original image and "offset overlay view"
- [ ] Create example with 1 or two images to show how to run the whole pipeline.

**

- [ ] See previous entries
    - [X] (better skeleton at tissue-tissue boundaries)
    - [ ] (further improve cropping? -- ship first, do this later)

- [X] (Added phase 3 walkthrough for cheeky_cells in that repository)

- [ ] Implementation of a feature to directly show analysis result
    - To do after Claude:
        - [ ] project_results_to_full_image should be maintained, but 
        it should NOT plot the plant mask (already done), instead it should
        plot the tissue-specific skeletons only. 
        (plus their sizes in an annotation label)

- [ ] Implement way to combine multiple labels and distill separate skeletons 
afterwards.
    - [X] Use Claude Code to implement changes.
        - Description by Claude Opus 5: 
        *Root and shoot are no longer skeletonized separately (which distorted both centerlines at
        their shared border); instead they are merged into one binary mask, skeletonized once, and
        the skeleton is split again by assigning each pixel to the nearest tissue. Holes in that mask
        (e.g. a seed interrupting the root) are bridged beforehand by dilating with the smallest
        radius that makes the mask whole, capped by ConfigPipeline.dilation_radius_maximum; plants
        exceeding the cap fall back to the old per-tissue treatment. Over 131 test plants the net
        effect was centered on zero (root: mean -2.4 px, median 0) with 25 plants bridged, and
        ConfigPipeline(shared_skeleton=False) reproduces the previous output exactly.*
        - NOTE: assess myself what happens if this procedure result in multiple skeletons (because there are multiple root or shoot areas).
    - [ ] Test the code.
    - [ ] Double check open issues
        - See journal/refactor_20260806_sharedskeleton-holes_issues.md, which explains a case where it goes wrong.
        - When there are multi-area roots or shoots, the longest stretch is taken (this requires root and shoots to be intermingled, doesn't occur with "seed blockade"), instead of summing the lengths.
    - Keep closing operation in mind.
- [ ] Go over test sample and make description of what happens for documentation
    
##### Notes regarding root/shoot joined area analysis

*I currently have a pipeline where root and shoot masks are determined from the labeled mask per plant, and the length is determined based on the skeleton of either root or shoot, without sharing knowledge between the two masks. I'd like to change this as follows;
the root and shoot mask should be combined, such and a shared skeleton should be determined. then, to determine the root and shoot specific skeletons, the relevant (root/shoot) mask is simply applied to the skeleton.*

- Add a "fix_holes()" function
- Would easiest solution not simply be to create a joined mask and store that
in the tissue class?
- things to check: how is longest skeleton length restricted to path that touches root/shoot boundary?

##### Optional features later;

- [ ] Show the crop-recteangle in the Napari viewer (and allow it to be edited)

##### Current test case: 

- projects/20260621_highresmodel-crop/202606_highrescropmodel_batch1_2lengths.py

## Open nice-to-address issues

- [ ] some functions have a hard-coded label for roots/shoots (ie `== 1`, or 
  `== 2`) (and also of course `root_tissue` or `shoot_tissue` restrict
  the analysis to only root-shoot analysis). This occurs e.g. in `analyze_plate()`
  and `assign_nearest_tissue()`. Should this be made more general?
- [ ] When there are a few pixels of misclassified root along the shoot, or 
vice versa, this will result in not having a joined skeleton (because that 
requires having only two connected components). Can be fixed e.g. by removing 
small parts (e.g. <5 px) automatically on a per-tissue masks.