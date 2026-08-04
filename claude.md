# claude.md

Guidance for AI coding assistants (Claude Code, Copilot, etc.). 
This file is loaded into the agent every turn, so brevity matters.

---

## Project overview

**root_length** is a Python toolkit that analyzes pictures of plants, for which
segmentation masks are already available. These masks are organized in a folder
which have the same file structure as the original image files.

Datasets are typically organized in multiple images organized in subdirectories.
Image file names identify conditions and dataset details.
Each image contains a picture of an agar plate on which multiple plants are grown.
The goal of the analysis is to quantify plant parameters, of which the most
important ones are root and shoot length.

---

## Repository layout


- [root_length/](root_length/) contains installable python package
    - [functions_files/filelisting.py](root_length/functions_files/filelisting.py) provides functionality to 
    read the directory structure into a dataframe.
    - [edit_segfiles](root_length/functions_pipeline/edit_segfiles.py) the function `edit_all_segfiles` and 
    its helpers allows looping over files (using napari), to correct the segmentation masks.
    - [analyze_plate](root_length/functions_pipeline/analyze_plate.py), `analyze_plate.py` analyzes the 
    segmentation masks. this file makes use of the following functionality:
        - [preprocessing](root_length/functions_pipeline/preprocessing.py), as name says contains pre-processing 
        functions. E.g. identify plants and clean the mask.
        - [determine_length](root_length/functions_pipeline/determine_length.py), offers functionality
        to (using skeletonization and graph representation) find the root lengths. 
        - [determine_length_moreplots](root_length/functions_pipeline/determine_length_moreplots.py), placeholder,
        currently only has imports, but more plotting functionalities could be added here.
    - [preprocessing_seg](root_length/functions_pipeline/preprocessing_seg.py), offers additional preprocessing
    functionality. This is a separate file as these functions can also be plugged directly into another repository
    that does the segmentation itself. [preprocessing.py](root_length/functions_pipeline/preprocessing.py) only
    operates within this repo.
    - [utils.py](root_length/functions_pipeline/utils.py) contains plant color scheme.
    

#### Script that run the pipeline

Python files in which analysis of samples is performed based on this 
repository (root_length) are also found in subfolders of the root dir.

- The [projects](projects/) directory contains files that run the pipeline
to analyze samples.
- The [segmentation_training](segmentation_training/) directory contains
files that use an external library (cheeky_cells) to train a ML algorithm
that will perform the segmentation.
- The file [pipeline_example_v2.py](pipeline_example_v2.py) in the root
dir provides an example pipeline that uses the root_length pipeline. [pipeline_example_v1.py](pipeline_example_v1.py)
as well (but is a previous version).


#### Other folders

- [old-code](old-code/), [misc](misc/), and [devtools](devtools), contain one-off 
scripts, scratch scripts, some obscure helpers, and old python files mostly not 
generally used. Don't consider or touch this code unless specified.
- [journal](journal/) contains some notes about this repo. very detailed
regarding specific issues, so generally don't consider this, unless
relevant for detailed discussion about code history. (might also have outdated
information)
- [figures](/figures/) for figures displayed in readme, can be ignored by LLMs.
- [example_files](example_files), for specific tests, contents can be ignored by LLMs.
- [documentation](documentation/), contains some empty files, intended
to put documentation in. can be ignored for now.


Package metadata lives in [pyproject.toml](pyproject.toml). Dependencies are
**not** declared there — they are installed via conda (see below).

---

## Environment & commands

This project uses **conda**, not pip-managed virtualenvs.

**No test suite, linter, or CI is configured.** Do not invent commands like
`pytest`, `ruff`, or `make test` — they will not exist. Validate changes by
running the relevant pipeline script or by importing the modified module.

This repo runs in `cheeky-all` conda env, ie `conda activate cheeky-all`.

---

## Conventions

- Module imports in pipeline scripts often include a commented
  `# import importlib; importlib.reload(...)` line — this is intentional for
  interactive (Spyder / Jupyter) development.

---

## Working preferences

- If changes are made, claude.md can be updated accordingly.
- Coding style;
    - Code should be relatively offensive (no tests, extensive docstrings). 
    - Code should be modular, but rather a two liner than extensive function definitions.
    - Generally, code is organized in topical blocks, headed by explanatory comment 
    focussed on the why and what, not the how.
    - Do not add type hints, docstrings, or comments to existing code that does not
  already have them, unless that is the task.
- **Do not modify** the following files/folders under  unless explicitly asked:
[old-code](old-code/), [misc](misc/), [devtools](devtools), [journal](journal/), [figures](/figures/), [example_files](example_files), [documentation](documentation/).
- Do not introduce new dependencies without flagging it — the conda recipe in
  the README is the source of truth and must be updated in lockstep.
  (Though this recipe is currently not present.)

---

## Known gotchas

- `pyproject.toml` declares `dependencies = []`. Runtime deps come from conda;
  `pip install -e .` only registers the package.
- The image files can either (or both)
    - be cropped before segmenting them, in which case
the segfiles will be cropped. To translate from the original images, the
stored parameter `prepr_info` in the segfile contains the cropping information. 
    - have a cropping recteangle attached that was determined afterwards, 
    which is stored in the parameter `mask_rect` in the segfile. 
    Depending on settings, the segfile might be cleaned outside that mask
    by the `compute_and_save_mask_rect_all` function.


---

## Pointers for deeper context

- Beginning of documentation (needs further editing from human programmer): [readme.md](readme.md)
