"""Configuration for the plate analysis pipeline.

Lives in its own module so both `analyze_plate` (which creates the config and
reads the plate-level settings) and `determine_length` (which reads the
per-plant settings) can import it without importing each other.
"""

################################################################################
# %% Libraries

from dataclasses import dataclass


################################################################################
# %% Settings that steer the pipeline

@dataclass
class ConfigPipeline:
    smoothing_diskradius: int | None = None
    # derive root and shoot centerlines from one shared skeleton
    shared_skeleton_flag: bool = True
    # largest dilation radius allowed to bridge holes in the root+shoot mask
    dilation_radius_maximum: int = 15
    # ignore labels outside the stored plate area (`mask_rect`) when loading
    apply_mask_rect: bool = True
    # dpi for plots
    dpi_plots: int = 300
