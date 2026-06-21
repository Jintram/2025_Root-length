"""
In phase 2, the annotated tiles from phase 1 are used to train
a U-Net segmentation model.
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
    training_dir = '/Users/m.wehrens/Data_UVA/2025_10_hypocotyl-root-length/TRAININGDIR_SET-2_20260605/',
    metadata_customized_filename = 'metadata_imagefiles_manual.xlsx',
    target_device = 'mps',
    nr_classes = 5,
    img_suffix = '_img_enhanced',
    lbl_suffix = '_seg',
    learning_rate = 1e-3,
    batch_size = 8,
    # for test run
    artificial_n=5000, # 1000
    epochs = 24, # 24
    # for actual run
    #artificial_n = 1000, # lower this during tests
    #epochs = 24,
    cmap_custom = custom_colormap, 
)
    # Timing notes;
    # batch 8, epoch 24, n 1000, took 2 hrs

dataset_train, dataset_test, model_unet = o2.phase2_setup(config2)

# %% ###########################################################################
# Running it

saved_model_path = o2.phase2_train(config2, dataset_train, dataset_test, model_unet)
print(f'Model saved to: {saved_model_path}')

# %%
