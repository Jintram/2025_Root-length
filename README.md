



### Quantifying root length


## Project description

Color images of plants were taken using a scanner (a basic consumer model), with the aim of determining root lenghts
from these color pictures.

The scripts here 

1. Uses machine learning scripts (from the package [Cheeky-cells](https://github.com/Jintram/Cheeky-cells/), a set of scripts that allow training and 
segmenting of images) to segment the plant images.
2. Use segmentation files as input to determine the length of each plant.

The image below shows the segmentation of the plants:
<img src="figures/example_plant_segmentation.png" width=50%>

The image below shows the length measurement of a plant:
<img src="figures/screenshot_plant-analysis.png" width=50%>


**Image description:** *White is the root, green is the shoot, brown is the seed, and dark green are leaves (as determined by 
the segmentation).*

We want to determine the size of the root and the size of the shoot.


## Setup

I'm using the same environment as the [Cheeky_cells](https://github.com/Jintram/Cheeky-cells) package
to run these scripts. For installation instructions of that environment,
[see that repository](https://github.com/Jintram/Cheeky-cells).

```
conda activate cheeky-all
```

### Download the scripts

To use them, you will need to put the scripts in this repository on your computer.

You can download them using the green "<> Code" button at the right top, and press "download zip".

A more advanced option is to set up git and clone the repository to your local computer, see below.

```
# Navigate to the directory you'd like to install the scripts
cd /path/to/your/directory

# Clone the repository using git
git clone git@github.com:Jintram/2025_Root-length.git
cd 2025_Root-length
```

Then, you need to run the following command (replace `/path/to/script/directory` by the actual path):

```
pip install -e /path/to/script/directory
```

## Workflow and input/output directories

### Expected input folders

- You have your raw image data organized in a directory, which may also have
subdirectories that contain image data.

- `<directory_with_raw_data>`
    - `<some_folder>`
        - `your_file.tif`
        - `(..)`
    - `<some_other_folder>`
        - `<yet_another_folder>`
            - `yet_another_file.tif`
            - `(..)`
    - `(..)`
    - `your_other_file.tif`
    
## Output folders that will be generated

- The script will create two output folders, for which you need to specify a
location. It is convenient to create a new output folder, separate from the 
input folder (referred to below by `your/output/folder/`). The two output folders, `LEN` and `SEG` 
(names can be further customized), then look as follows:

- `your/output/folder/`
    - `SEG/`
        - `SEG/plots`  mirrors the original directory structure, where instead
        of pictures of plates, *there are plots showing the segmentation of the plants
        on top of the original image.*
        - `SEG/segfiles` mirrors the original directory structure, where instead
        of pictures of plates, *there are `.npz` datafiles, which each hold information 
        about the segmentation.
    - `LEN/`
        - `LEN/all_samples_length.xlsx` is a summary file which holds the plant root
        lengths for all analyzed plates from the input folder. You can use this for 
        plotting. Metadata about conditions can be added based on file and subfolder
        names.
        - `LEN/data` mirrors the original directory structure, where instead
        of pictures of plates, *there are `.tsv` files, which list the plant lengths
        in the plate. (Plants are assigned unique IDs.)*
        - `LEN/lenplots` mirrors the original directory structure, where instead
        of pictures of plates, *there are plots showing the plates with projected on top
        the lengths of each plant.*    

Because `LEN/data` contains a `.tsv` file for each plate, you can later
easily combine data from multiple datasets that were analyzed separately.


## Usage of this package






# Installation instructions for developers

- Contributors additionally install: `mamba install -c conda-forge pytest ruff`


# Potential things to improve

- Improve the cropping procedure (use another ML model to detect foreground/background?)


