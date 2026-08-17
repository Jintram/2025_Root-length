

# Technical details

*Walkthrough of the pipeline and hierarchy of functions called.*

Note that this file only discusses the length analysis (part 2) of 
the pipeline, as part 1 is conducted using the 
[Cheeky-cells](https://github.com/Jintram/Cheeky-cells/)
package.


- `root_length/functions_files/filelisting.py`
    - function `gen_metadatafile_segfiles()` 
        - will search for `.npz` files in a given directory and subdirectory
        - stores filelist in pandas metadata dataframe
            - this dataframe simply has columns `basedir`, `subdir` and `filename`.
            - result stored typically as `df_filelist`
    
- call to `compute_and_save_mask_rect_all()` 
    - (defined in `root_length/functions_pipeline/edit_segfiles.py`)
    - Given `df_filelist`, loops over files
        - file info is stored in `fileinfo` dataclass (from `filelisting.py`)
        - segfile contents and related image (required) are loaded
        - **artifact/legacy cropping** previously I allowed cropping at another time, 
        with segfile gettting cropped, but image not. `prepr_info` contains
        contains the rect, and cropping is applied to original img here as well.
            - currently intended usage will keep segfile full-sized, but 
            allow storing of `mask_rect` (determined with this function).
        - `preprocess_getbbox_insideplate2()` is called 
            - (defined in `root_length/functions_pipeline/preprocessing_seg.py`)
            - this is a simple function that detects the background
            around the plate and returns cropped `img_cropped, rect`
            - `rect` is manually adjusted to outline an ROI that
            covers the agar but not artifacts (margins are added). strategy;
                - takes 50th percentile of grayscale img
                    - assumes to be agar background levels
                    - thus includes edges (bright)
                    - **completely** excludes background at side
                - calls `bbox_from_mask_light()`
                    - mask is projected to 1d for x and y using `any()`
                    - first and last non-zero elements are taken as boundary
                    - rect `r0, c0, r1, c1` is returned
                - use full pic if rect `< min_expected_area`.
                - add margins (allowed in `%` or `px`)
        - **artifact/legacy** optionally, outside mask zeroed; not meant as standard practice as 
        whole pipeline will understand the stored `mask_rect`
            - `_save_segfile()` is used to store the mask (`mask_rect`) in original segfile
                    

<img src="figures/technical/plate_detection_1getbbox2.png" width=50%><br>
***Figure.** Thresholded image by 50th percentile.*

<img src="figures/technical/plate_detection_2flatmask.png" width=50%><br>
***Figure.** Resulting mask projected on 1d.*

- call to `edit_all_segfiles()`
    - (defined in `root_length/functions_pipeline/edit_segfiles.py`)
    - again loops over files using `df_filelist` and `curr_file` class
        - calls `edit_segfile_single()`
            - handles loading of segfile and image file
            - (again dealing with legacy crop_rect)
            - calls `edit_annotation_napari()`
                - Large, written by Claude
                - Calls the Napari GUI
                - Defines subfunctions to interact with segmentation
                    - Shift view (visual effect, Claude-written)
                    - `mask_rect` adjustment (with Napari layer, Claude-written)
                    - Remove small/large components (calls `remove_small_foregroundregions()` and `remove_large_foregroundregions()`)
                    - manually draw a new root/shoot boundary with **r**
                        - important function
                        - calls `correct_mask_rootshootline()`
                            - Places a seed pixel at mouse position (row, col) with label 5, then finds the nearest
                            background pixel (value == 0) to the left and right of the seed using
                            distance grid, and draws line between those two (using label 5).
                            
                            <img src="figures/technical/give_boundary_line.png" width=45%>
                            <img src="figures/technical/give_boundary_line_fail.png" width=45%><br>
                            
                            ***Left image.** Illustration of procedure. To the left
                            and right of the black cross (determined by the mouse click) the pixels with the lowest
                            distance (color-coded) outside the plant mask (light green) are located. Those
                            two pixels are connected by a boundary line (red). The **right image** shows a case
                            where this fails.*
                            
                            - Requires updating mask with "u" functionality (see below).
                    - manually draw a new root/shoot boundary with **t**
                        - important function
                        - calls `correct_mask_throughline()` 
                            - Places seed at mouse position, 
                            finds the nearest background pixel (bg1),
                            and then draws a line from bg1 to the seed, 
                            and continues until background is hit, thus
                            creating a new root/shoot boundary line.
                            - The idea is that when placed in the middle of 
                            the tissue, this will identify a line perpendicular 
                            to the curvature.
                                - See the illustration of the previous function for reference (this function will use the overall closest point and the black cross to draw a line as described above).
                            - Requires updating mask with "u" functionality (see below).
                    - **u** functionality allows updating root shoot assignments
                    based on newly drawn root/shoot boundaries (**r** or **t**).
                        - calls `relabel_by_rootshootlines()`
                            - converts mask to binary and loops over blobs
                            - if red line;
                            - find CoM of blob 
                            - separate blob by red line
                            - assign sub-regions to root/shoot based on whether
                            sub-region CoM is above/below overall CoM. 
                            - all red lines will be assigned root identity
                    - Analysis preview; 
                        - invokes `root_length/functions_pipeline/napari_analysis.py`
                        - this performs an analysis round that would usually
                        be done in the rest of the pipeline. 
                        - not saved, because saving would create risk of 
                        all kinds of different settings being used per plate.
                    - Navigation functions, wich use local helpers to
                    allow saving+continue (auto behavior upon closing Napari GUI), 
                    save now, no save+continue, jumping
                    to sample N, quit no save.

- Call to `analyze_all_plates()`
    - (Defined in `root_length/functions_pipeline/analyze_plate/`)
    - This performs the actual length measurements.
    - Again loops over all files using `df_filelist` and `curr_file` class
        - Calls `analyze_plate(curr_file, ..)`
            - Loads labeled segmentation `mask` and `mask_rect`, applies `mask_rect`
            - This calls `analyze_labels(img_mask, config_pipeline=config_pipeline)`
                - calls `clean_mask()` (from `preprocessing.py`)
                    - this removes small and large objects using **hard coded**
                    parameters `TYPICAL_PLANT_SIZE = 2000; MIN_SIZE = TYPICAL_PLANT_SIZE/10; MAX_SIZE = TYPICAL_PLANT_SIZE*10`. 
                - calls `identify_plants()`
                    - this identifies and compiles a list of plant objects
                    - calls `find_individual_plants()` (defined in `preprocessing.py`)
                        - creates binary mask to identify single objects (potentially plants)
                        - loops over objects
                            - lifts ROI, only keeps current object
                            - per tissue label, count separate objects (`>min_count`) and size
                        - loop stores & function returns
                            - img list with plant ROIs
                            - list with label counts per plant (`[plant_idx, tissue_idx]`)
                            - regionprops from the binarized mask (total plant areas etc)
                    - `identify_plants()` now keeps only plants (**filters**) with 
                        - 1 root & at least one other category
                - `analyze_labels()` now loops over the plant img list
                    - creates root mask (stored as `TissueSample` from `determine_length`)
                    - creates shoot mask (stored as `TissueSample` from `determine_length`)
                    - creates overall plant sample (into `PlantSample` from `determine_length`)
                    - adds plant samples to list (`current_sample_all_plants`)
                    - loops over plant sample list
                        - calls `run_default_length_pipeline` (from `determine_length.py`)
                            - Determines length & other data for each plant object
                            - See below for outline
                        - stores all plant data (incl lengths) in `current_sample_all_plants`
                - returns `current_sample_all_plants` (and `img_mask_clean`)
            - Saves output
                - Overview figure 
                - `.csv` file is written **for this plate** with data for all
            the individual plants (root length, shoot length, ..)
    
    

- description of `run_default_length_pipeline()`
    - (defined in `determine_length`)
    - runs on single `PlantSample` and `ConfigPipeline` instance as input
    - runs `prepare_shared_skeleton()`
        - adds joined root/shoot tissue (as `TissueSample` instance) to plant sample `plant` (`PlantSample` instance)
        - then pads, dilates, performs closing
        - calls `dilate_to_connect()`
            - if there are any gaps in the joined mask, dilates by ~`g/2` (`g` being size of largest gap)
                - capped by `config_pipeline.dilation_radius_maximum`
            - returns **dilated** mask on which to base skeleton
        - returns fail if no single object was found (`plant.nosharedskeleton_flag = True`)
        - skeletonizes the joined tissue mask
        - calls `prune_skeleton_outside_mask()` 
            - removes 1-connected pixels outside the original mask
            - keeps >2-connected pixels outside the original mask, which bridge possible gaps
        - calls `get_nearest_tissue_map()`
            - employs `distance_transform_edt()` to map outside-mask pixels
            to closest tissue
            - this information gets returned and stored into `plant.nearest_tissue`
    - calls `prepare_tissue_skeleton()` if shared skeleton failed (`plant.nosharedskeleton_flag = True`) both for root and shoot
        - This keeps largest component, performs closing
        - Calls `generate_skeleton()`, one liner that applies `morphology.skeletonize()` to sample
    - calls `split_shared_skeleton()` if shared skeleton succeeded (`plant.nosharedskeleton_flag = False`) both for root and shoot tissue 
        - This extracts tissue-specific `clean_mask` and `skeleton` for root/shoot only based on the shared skeleton and `nearest_tissue` (see above)
        - Also creates `anchor_mask`, which are parts of the shared skeleton **not** corresponding to this tissue,
        later used to find root/shoot boundary position to anchor longest-path in skeleton.
    - calls `run_tissue_pipeline()` both for `plant.root` and `plant.shoot`
        - (`plant.root` and `plant.shoot` are now either based on single-tissue mask or shared-root/shoot skeletons)
        - calls `sample = analyze_skeleton_branchpoints(sample)`
            - Counts pixel neighbors in mask 
            - Removes branchpoints (`neighbor_counts > 2`)
            - Collect end point (`neighbor_counts == 1`) and branchpoint locations
            - Stores info on sample
            
            <img src=figures/skeleton_no_branchpoints.png width=50%><br>
            
            *Example of a skeleton without branch points, with branch points and 
            end points highlighted in red and white circles, respectively.*
            
        - calls `sample = label_skeleton_segments(sample)`
            - This simply produces a labeled version of the skeleton without branchpoints (each segment gets its own label).
            - Then adds new label at each branchpoint location
            - And new label at each end point location
            
            <img src=figures/labeled_skeleton_no_branchpoints.png width=50%><br>
            
            *Example of labeled skeleton as described above.*
            
        - calls `sample = build_segment_graph(sample)`
            -  Uses the `networkx` library to create a graph representation where
            each segment is one node.
            - Each segment is connected to another one if it touches (dilation based).
            - The segments are assigned a size in area (total pixels) and length by `get_length_segment()`
                - `get_length_segment()` assumed <=2 connectivity, and assigns
                uses distance kernel,
                ```python
                DISTANCE_KERNEL = np.array([[np.sqrt(2), 1, np.sqrt(2)],
                            [1, 0, 1],
                            [np.sqrt(2), 1, np.sqrt(2)]]) / 2
                ```
                - to get the per-pixel distance contribution, which are then summed for total segment length.
                            
        - calls `sample = find_start_labels_close_to_anchor(sample)`
            - Using a distance map determines which skeleton segment lies closest
            to root/shoot boundary (closest to root if self is shoot, and vice versa), using the `anchor_mask`
            - (Does this per connected component, such that the longest path is eventually
            searched for over all connected components -- usually there's only one.)
            - stored in 
        - calls `sample = get_long_path_in_graph_nodearea(sample)`
            - goes over `sample.start_labels` and simply calculates
            path length to each other segment
            - then the path with longest length is selected, stored are
                - `sample.longest_path` (stores longest path, as array of segment labels)
                - `sample.max_length_px_bynx` (length of longest path)
                - `sample.total_length_px_bynx` (if multiple connected segments, sum of respective max lengths)
        - calls `sample = build_longest_path_mask(sample)`
                - creates `sample.mask_longest_path` with mask that represents
                longest path
        - calls `sample = get_length_longestpath(sample)`
                - calculates `sample.length_pixels`, which is taken
                as the final root length.
                    - This is slightly different from `sample.total_length_px_bynx`
                    because segment-to-segment connections are not counted 
                    towards the total length, only the sum of segment lengths is taken for 
                    `sample.total_length_px_bynx`. Therefor, `sample.length_pixels`
                    is the final length.
    - `run_default_length_pipeline` also calculated length in mm if 
    conversion factor is available.

        
