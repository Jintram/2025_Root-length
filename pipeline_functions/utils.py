
from matplotlib.colors import ListedColormap

# Custom color map for the labeled masks of the plants
custom_colors_plantclasses = \
    [   # 0 = background black
        '#000000', 
        # 1 = shoot light green
        '#90EE90',
        # 2 = root white
        '#FFFFFF', 
        # 3 = seed brown
        '#A52A2A', 
        # 4 = leaf darkgreen
        '#006400' 
        ]
cmap_plantclasses = ListedColormap(custom_colors_plantclasses)