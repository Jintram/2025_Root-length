
"""
Run the length analysis from within the napari editor, and show it as layers.

This is what turns the editor into an edit -> measure -> check loop: correct a
mask, press the button, and see whether the centerlines the pipeline traces
through it are the ones you meant.

Lives apart from `edit_segfiles` so that module stays about editing: it only
asks `make_analysis_widgets` for widgets and docks them, everything else
(settings, running the pipeline, drawing the result) is here.
"""

################################################################################
# %% libraries

import traceback

import numpy as np

from magicgui import magicgui
from napari.utils.colormaps import DirectLabelColormap

import root_length.functions_pipeline.analyze_plate as plana
    # import importlib; importlib.reload(plana)
import root_length.functions_pipeline.determine_length as pllen
    # import importlib; importlib.reload(pllen)
from root_length.functions_pipeline.config import ConfigPipeline


################################################################################
# %% the overlay layers

# The result gets its own layers, wiped and rebuilt on every run. Named, so
# they can be found back between the layers that are being edited.
LAYER_NAME_TRACES = 'analysis: traces'
LAYER_NAME_PLANTS = 'analysis: plants'

# Same colors as the matplotlib overview plot, keyed by the values that
# `project_results_to_full_image` writes. `None` is the fallback for values not
# listed, which napari requires to be present.
TRACES_COLORMAP = DirectLabelColormap(color_dict={
    None: 'transparent',
    0: 'transparent',
    1: 'gray',      # skeleton
    2: 'red',       # root centerline
    3: 'orange',    # shoot centerline
})


def _format_length(tissue):
    """Length of one tissue as shown next to a plant, in pixels like the plot."""

    if tissue.length_pixels is None:
        return "n/a"

    return f"{tissue.length_pixels:.1f}px"


def remove_analysis_layers(viewer):
    """Drop the overlay layers of a previous run, if there are any."""

    for layer_name in (LAYER_NAME_TRACES, LAYER_NAME_PLANTS):
        if layer_name in viewer.layers:
            viewer.layers.remove(layer_name)


def add_analysis_layers(viewer, plant_results, shape):
    """
    Show centerlines, plant boxes and lengths as layers on top of the labels.

    Shows the same things as `determine_length.plot_all_plants_projected`, but
    as layers that can be toggled and zoomed along with the labels being edited.
    They are made non-editable and the previously active layer is selected
    again afterwards, so that painting keeps landing on the segmentation.
    """

    remove_analysis_layers(viewer)
    active_before = viewer.layers.selection.active

    # the traced centerlines of all plants, in one label image
    traces_layer = viewer.add_labels(
        pllen.project_results_to_full_image(plant_results, shape),
        name=LAYER_NAME_TRACES, colormap=TRACES_COLORMAP, opacity=1.0)

    # a box per plant, labeled with the lengths that were found for it
    plant_boxes, plant_labels = [], []
    for idx, plant in enumerate(plant_results):

        if plant is None or plant.bbox is None:
            continue

        minr, minc, maxr, maxc = plant.bbox
        plant_boxes.append(
            np.array([[minr, minc], [minr, maxc], [maxr, maxc], [maxr, minc]]))
        plant_labels.append(f"({idx}) root {_format_length(plant.root)}, "
                            f"shoot {_format_length(plant.shoot)}")

    plants_layer = viewer.add_shapes(
        plant_boxes, shape_type='rectangle', name=LAYER_NAME_PLANTS,
        edge_color='#ffc900', face_color='transparent', edge_width=4,
        features={'lengths': plant_labels},
        text={'string': '{lengths}', 'size': 8, 'color': 'white',
              'anchor': 'upper_left', 'translation': [-8, 0]},
    )

    for layer in (traces_layer, plants_layer):
        layer.editable = False
    if active_before is not None:
        viewer.layers.selection.active = active_before

    return traces_layer, plants_layer


################################################################################
# %% the widget docked in the editor

# Settings outlive the viewer, so the choices made on one plate still apply to
# the next file of an editing session.
LAST_CONFIG = ConfigPipeline()


def make_analysis_widgets(viewer, get_labels):
    """
    Build the analysis widgets for the napari editor.

    `get_labels` is called (without arguments) when the button is pressed and
    should hand back the label image to measure; `edit_segfiles` passes the
    labels as they are on screen, cropped to the plate rectangle as it is now,
    so that what is measured here is what `analyze_plate` measures later.

    Returns a list of magicgui widgets, for the caller to dock.
    """

    @magicgui(
        smoothing_diskradius={
            "widget_type": "SpinBox", "min": 0, "max": 50,
            "value": LAST_CONFIG.smoothing_diskradius or 0,
            "label": "Smoothing radius (0=off)"},
        dilation_radius_maximum={
            "widget_type": "SpinBox", "min": 0, "max": 100,
            "value": LAST_CONFIG.dilation_radius_maximum,
            "label": "Max bridging dilation"},
        shared_skeleton_flag={
            "value": LAST_CONFIG.shared_skeleton_flag,
            "label": "Shared root+shoot skeleton"},
        call_button="Preview analysis result",
    )
    def analyze_widget(smoothing_diskradius: int, dilation_radius_maximum: int,
                       shared_skeleton_flag: bool):

        # The viewer is frozen for as long as this runs (seconds to a minute),
        # so the prints of the pipeline are the only sign of life. Doing it in a
        # worker thread instead would mean the labels can be edited halfway
        # through a measurement of those same labels.
        print("____\nAnalyzing plate — the viewer is unresponsive until this ends.")
        img_mask = get_labels()
        try:
            _, plant_results = plana.analyze_labels(img_mask, LAST_CONFIG)
        except Exception:
            traceback.print_exc()
            print("  Analysis FAILED (see traceback above), layers not updated.\n____")
            return

        add_analysis_layers(viewer, plant_results, img_mask.shape)
        print(f"  Analyzed {len(plant_results)} plants; "
              "showing them in the 'analysis: ...' layers.\n____")

    def _remember_settings(*_):
        """
        Copy the widget values into `LAST_CONFIG`, which is where they are read
        from: both by the button below and by the next viewer that opens.

        Hooked on every change rather than on pressing the button, so a setting
        that is dialed in and then not used is still there for the next plate.
        Reading the widgets when the window closes is not an option: Qt has
        deleted them by then (see the snapshot dance in `edit_segfiles`).

        `apply_mask_rect` is left at its default, since here the cropping
        already happened in `get_labels`.
        """
        global LAST_CONFIG
        LAST_CONFIG = ConfigPipeline(
            smoothing_diskradius=analyze_widget.smoothing_diskradius.value or None,
            shared_skeleton_flag=analyze_widget.shared_skeleton_flag.value,
            dilation_radius_maximum=analyze_widget.dilation_radius_maximum.value)

    analyze_widget.changed.connect(_remember_settings)

    return [analyze_widget]
