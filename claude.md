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
        - [napari_analysis](root_length/functions_pipeline/napari_analysis.py) adds the "Analyze plate"
        widget to that editor: it runs `analyze_plate.analyze_labels` on the labels currently on screen
        and shows the result as `analysis: ...` layers. Kept out of `edit_segfiles` to keep that file
        about editing (and because it imports the analysis side of the package).
    - [analyze_plate](root_length/functions_pipeline/analyze_plate.py), `analyze_plate.py` analyzes the 
    segmentation masks. `analyze_labels` does the measuring on an in-memory label image, `analyze_plate`
    wraps file loading, the overview plot and the .tsv around it. This file makes use of the following
    functionality:
        - [preprocessing](root_length/functions_pipeline/preprocessing.py), as name says contains pre-processing 
        functions. E.g. identify plants and clean the mask.
        - [determine_length](root_length/functions_pipeline/determine_length.py), offers functionality
        to (using skeletonization and graph representation) find the root lengths. 
        - [determine_length_moreplots](root_length/functions_pipeline/determine_length_moreplots.py), placeholder,
        currently only has imports, but more plotting functionalities could be added here.
        - [config](root_length/functions_pipeline/config.py) holds the `ConfigPipeline` dataclass with all
        pipeline settings. Its own module so `analyze_plate` and `determine_length` can both import it
        without importing each other.
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

**No human-made test suites, only Claude-generated testing.** `pytest` is
installed and can be used sparingly. There's no human-coded test suite: this is
not production-grade code, but an academic project where bugs surface during
manual use. Keep testing light.

Claude-written tests are **kept and expected to pass** — they form a small
regression net.

- Save as `tests/test_byclaude_<somescript>.py`, where `<somescript>` is the
  module under test.
- After changing a module, run its test file
  (`pytest tests/test_byclaude_<somescript>.py`); run `pytest tests/` if the
  change was broad.
- Each file opens with a docstring recording **the date**, **which module and
  functions it covers**, and **the assumed behaviour** it pins down — one line
  per test. Each test function gets a one-line docstring naming the behaviour it
  checks. This documentation is what makes a later session able to judge a
  failure, so it is a deliberate exception to the "don't write docstrings"
  preference below.

When a test fails, decide which case applies before changing anything:

1. *The behaviour is still intended* → the recent code change is a bug. Fix the
   code.
2. *The behaviour was deliberately changed* → the test is stale. Update it only
   if the new intended behaviour is unambiguous. Otherwise mark it
   `@pytest.mark.skip(reason="STALE <date>: <what changed>")`, leave it in
   place, and report it to the user for updating or removal.

Never weaken an assertion or delete a test just to get green — that silently
discards the regression net.

`ruff` is installed as well, but is not intended to be used, as code is
non-compliant (and doesn't aim to).

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
    - Code should be modular, but rather a two liner than many function definitions.
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
    which is stored in the parameter `mask_rect` in the segfile 
    (4 numbers `(r0, r1, c0, c1)`, never a raster mask; in the same coordinate
    frame as `img_pred_lbls`, so already cropped by `prepr_info`).
    `mask_rect` is applied non-destructively: `analyze_plate` discards labels
    outside it when loading (switch: `ConfigPipeline.apply_mask_rect`), so the
    segfile keeps everything and the rectangle stays correctable.
    The napari editor shows the same filtered view, lets the rectangle be
    dragged, and saves it along with the labels; labels it hid are put back on
    save, so widening the rectangle later brings them back. Use its
    "Save + reload plate" button to see the effect of a changed rectangle.
    Only `compute_and_save_mask_rect_all(clear_outside_mask=True)` still
    clears labels destructively, and it is off by default.
    - Reading either rect from a segfile goes through
    `preprocessing.normalize_rect`, which unwraps the np.savez quirks and
    clamps to the image (a negative coordinate would silently wrap when
    slicing, rather than raise). Apply with `preprocessing.apply_mask_rect`,
    and merge edits back into the complete labels with
    `preprocessing.paste_inside_rect`.
    - `edit_annotation_napari` is also handed to the external cheeky_cells
    library (from [segmentation_training](segmentation_training/)), which calls
    it with 3 positional arguments and unpacks 2 return values. Scripts must
    pass `edit_annotation_napari_cheekycells`, the adapter that keeps that
    older shape; drop it once cheeky_cells is updated.
- The editor's analysis button ("Preview analysis result", in the "Analysis
  preview" box) measures the labels *on screen*, cropped
  to the rectangle as it is at that moment (so unlike the rest of the editor it
  does not need a reload after dragging), and never touches disk: no .tsv and no
  plot are written, and unsaved edits are measured as they are. It blocks the
  viewer while running, on purpose — a worker thread would let the labels be
  edited halfway through measuring them. Its settings live in
  `napari_analysis.LAST_CONFIG` (not in `EditorSessionState`, which cannot be
  read at closing time), updated on every widget change, so they carry over to
  the next file of a session.
- Every action of the napari editor is declared once in the `ACTIONS` table in
  `edit_annotation_napari` — `(key, caption, callback, gets_button, needs_mouse)`
  — which is what builds the keybindings, the buttons and the help text, so add
  new actions there rather than writing a `@viewer.bind_key` by hand. Buttons are
  captioned `"Caption [k]"`, so each one advertises its own shortcut.
  `needs_mouse` (not `not gets_button`) is what puts an action in the info box:
  'w' also loses its button when `curr_file` is None, but is not a mouse action.
  The two mouse-position actions ('r', 't') deliberately have no button:
  clicking in the dock takes the pointer off the canvas, and
  `viewer.cursor.position` then holds wherever the mouse last crossed the canvas
  edge. They are advertised in an info box (a `magicgui` `Label` with HTML) at
  the top of the Tools panel instead. Every button hands keyboard focus back to
  the canvas when it is done (`_focus_canvas`), otherwise focus stays in a
  spinbox and the shortcuts appear to stop working.
- The Tools panel is stacked in a plain `QVBoxLayout` inside a `QScrollArea`,
  not a magicgui `Container`, because it mixes magicgui widgets with `QGroupBox`
  sections and is taller (~860px) than a laptop screen; napari's dock does not
  scroll by itself. Sections are numbered in the order a plate is worked
  through: 1. View, 2. Plate area, 3. Improve segmentation, 4. Analysis preview,
  5. Save & continue. Each holds its own inputs, button *and* hint, so hints sit
  next to what they describe rather than in one legend at the top.
  Build one with the local `_group_box(title, entries)` helper (entries may be
  magicgui widgets or raw QWidgets) and `_hint(text)` for the explanations —
  italic rather than coloured, since napari has light and dark themes.
  napari's stylesheet themes `QGroupBox` already, so don't style it.
  Widgets inside a box are not in `panel_entries`, so the focus-restore loop
  lists them explicitly — extend that list when adding one.


---

## Pointers for deeper context

- Beginning of documentation (needs further editing from human programmer): [README.md](README.md)
