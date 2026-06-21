"""
This is training a model using the combined annotated images of Martijn
and Yuzeng.
All these images are cropped.
"""

# %% ###########################################################################
# Libraries

import cheeky_cells.orchestrators.orchestrate_phase2_clean as o2
    # import importlib; importlib.reload(o2)


# %% ###########################################################################
# Configuration

# Colormap for display of annotation
custom_colormap = {
    0: 'transparent', # bg
    1: '#90EE90', # shoot
    2: '#FFFFFF', # root
    3: '#A52A2A', # seed
    4: '#006400', # leaf
    5: '#FF0000', # 5 = bright red (for corrections)
}


config2 = o2.Phase2Config(
    training_dir = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-1n2_20260618_cleaned/',
    metadata_customized_filename = 'metadata_imagefiles_manual.xlsx',
    target_device = 'mps',
    nr_classes = 5,
    img_suffix = '_img_enhanced',
    lbl_suffix = '_seg',
    learning_rate = 1e-4, # 1e-3
    batch_size = 8,
    # for test run
    # artificial_n= 2, # 5000, # 1000
    # epochs = 2, # 42, # 24
    # for actual run
    artificial_n = 5000, # lower this during tests
    epochs = 42,
    cmap_custom = custom_colormap,
    # 12-epoch warm-up: log-spaced from 0.01 -> 1.0 (~1.52x per step),
    # then 10/10/10 plateaus at 1.0 / 0.1 / 0.01.
    lr_schedule_relative = [
        (1, 0.0100), (1, 0.0152), (1, 0.0231), (1, 0.0351),
        (1, 0.0534), (1, 0.0811), (1, 0.1233), (1, 0.1874),
        (1, 0.2848), (1, 0.4329), (1, 0.6579), (1, 1.0000),
        (10, 1.0), (10, 0.1), (10, 0.01),
    ],
)
    # Timing notes;
    # batch 8, epoch 24, n 1000, took 2 hrs

dataset_train, dataset_test, model_unet = o2.phase2_setup(config2)

# %% ###########################################################################
# Running it

saved_model_path = o2.phase2_train(config2, dataset_train, dataset_test, model_unet)
print(f'Model saved to: {saved_model_path}')

# %%
