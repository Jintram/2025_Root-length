


# Type of input and output files

The input files that we have are .npz files, each containing a segmentation of
an agar plate with multiple plants. (Specified by `DIR_INPUTFILES`.)

The final output is an xlsx file with on each row a different plant, with its 
length and from which plate it came. (Will be stored in `DIR_OUTPUTFILES`.)

To process data, intermediate .tsv files are made, one for each plate,
in a subdirectory of `DIR_OUTPUTFILES`, called `length_datafiles/`, that adheres
to precisely the same directory structure as the input segfiles. The subdirectory
`length_plots` holds plots that allow you to identify individual plants, 
and check whether the analysis doesn't contain any errors. Based on these
intermediate files, the final output xlsx can be compiled.

**Note:** Further processing of the data will require you to add metadata
to the final output table (e.g. conditions of each plate). This can be done
using Python as well.

### Functions in this folder

The functions in this folder compile a list (dataframe) which holds all the segmentation files to be processed. 













