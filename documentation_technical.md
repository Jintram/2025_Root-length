

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
    
- `root_length/functions_pipeline/edit_segfiles.py`
    - `compute_and_save_mask_rect_all()` 
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


