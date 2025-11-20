# automatic_watershed.py (With Color Key Visualization & MANUAL THRESHOLD)

import os
from turtle import mode
import higra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Import patches for color key
from PIL import Image
from matplotlib import colormaps
from skimage import io, color, measure
from skimage.filters import threshold_otsu, threshold_local # We still need this for the default case
from skimage.morphology import closing, erosion, dilation, disk, remove_small_objects, reconstruction # Added remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from scipy import ndimage as ndi # Keep ndi import if needed elsewhere, though distance transform is removed
# from skimage.feature import peak_local_max # peak_local_max removed

CUSTOM_DISTINCT_COLORS_RGB = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Lime
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (0, 128, 0),     # Green
    (0, 128, 255),   # Sky Blue
    (255, 153, 153), # Light Pink
    (153, 76, 0),    # Brown
    (160, 160, 160), # Gray
    (0, 204, 102),   # Tealish Green
    (255, 204, 0),   # Gold
    (204, 0, 102),   # Cerise
    (102, 102, 255), # Light Purple
    (102, 255, 102), # Light Green
    (255, 102, 102), # Light Red
    (178, 102, 255), # Lavender
    (51, 153, 102),  # Dark Teal
    (255, 178, 102), # Peach
    (102, 178, 255), # Cornflower Blue
    (218, 112, 214), # Orchid
    (75, 0, 130),    # Indigo
    (240, 230, 140), # Khaki
    (0, 128, 128),   # Teal
    (255, 20, 147),  # Deep Pink
    (127, 255, 0),   # Chartreuse
    (255, 215, 0),   # Gold (Slightly different)
    (139, 0, 0),     # Dark Red
    (0, 100, 0),     # Dark Green
    (0, 0, 139),     # Dark Blue
    (255, 165, 0),   # Orange (Standard)
    (188, 143, 143), # Rosy Brown
    (70, 130, 180),  # Steel Blue
    (255, 250, 205), # Lemon Chiffon (Very Light Yellow)
    (147, 112, 219), # Medium Purple
    (0, 255, 127),   # Spring Green
    (210, 105, 30),  # Chocolate
]
NUM_CUSTOM_COLORS = len(CUSTOM_DISTINCT_COLORS_RGB)

output_dir = "output/connected_filtering"
    
def save_image(segmentation, filename):
    """Saves the segmentation label image as a color-mapped PNG file."""
    fig = plt.figure(frameon=False)
    fig.set_size_inches(segmentation.shape[1] / 100, segmentation.shape[0] / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Mascara o fundo (label 1)
    masked_seg = np.ma.masked_where(segmentation == 1, segmentation)
    ax.imshow(masked_seg, cmap='nipy_spectral', interpolation='nearest')
    
    # Salva
    fig.savefig(filename, dpi=100)
    plt.close(fig)

def run_automatic_watershed(color_image, vol_min_area, area_min_area, dynamics_min_area):
    """
    Performs automatic seeded watershed segmentation using sure foreground/background markers.

    Args:
        color_image (np.ndarray): The input color image (RGB).
        min_area_threshold (int): Minimum area threshold for identifying main components
                                      and for final component filtering.
        global_threshold (float, optional): A user-defined threshold (0.0-1.0). 
                                            If None, Otsu's method is used.

    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray) Figure with visualizations and the final segmented label image.
    """

    # === Block 1: Initial Image Conversion ===
    # Convert input image to grayscale for processing.
    image_rgb = np.array(color_image)
    print("Format of input image:", image_rgb.shape, image_rgb.dtype) # Verbose
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]
    array_grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
    array_grayscale8bit = array_grayscale.astype(np.uint8)
    img_grayscale = Image.fromarray(array_grayscale8bit)
    inverted_image = 255 - array_grayscale8bit
    image_size = inverted_image.shape
    # print(f"Image loaded. Size: {image_size}") # Verbose

    # === Block 2: HIGRA Graph and Edge Weight Setup ===
    # Create the graph representation of the image grid.
    # print("Creating adjacency graph...") # Verbose
    graph = higra.get_4_adjacency_graph(image_size)
    # Calculate edge weights based on L1 difference on the ORIGINAL grayscale image.
    # print("Calculating edge weights on original image...") # Verbose
    #gradient = higra.weight_graph(graph, array_grayscale8bit, higra.WeightFunction.L1)

    tree, altitudes = higra.component_tree_max_tree(graph, inverted_image)

    vertex_area = higra.attribute_vertex_area(graph)

    modes = ['Volume', 'Area', 'Dynamics']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes.ravel()
    
    ax[0].imshow(inverted_image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    io.imsave(os.path.join(output_dir, "00_original.png"), array_grayscale8bit)

    for i, mode in enumerate(modes):
        # print(f"Generating markers using extinction mode: {mode}...") # Verbose
        if mode == 'Volume':
            attr_value = higra.attribute_volume(tree, altitudes)
            max_vol = attr_value.max()

            threshold_ratio = 0.0025 
            calculated_threshold = max_vol * threshold_ratio

            print(f"   Limiar Aplicado: {calculated_threshold} (Objetos menores que isso somem)")

            deleted_nodes = attr_value < calculated_threshold
        elif mode == 'Area':
            attr_value = higra.attribute_area(tree)
            max_vol = attr_value.max()
        
            threshold_ratio = 0.0025 
            calculated_threshold = max_vol * threshold_ratio

            print(f"   Limiar Aplicado: {calculated_threshold} (Objetos menores que isso somem)")

            deleted_nodes = attr_value < calculated_threshold
        elif mode == 'Dynamics':
            attr_value = higra.attribute_dynamics(tree, altitudes)
            deleted_nodes = attr_value < dynamics_min_area
        
        new_tree, node_map = higra.simplify_tree(tree, deleted_nodes)

        new_altitude = altitudes[node_map]

        filtred_image = higra.reconstruct_leaf_data(new_tree, new_altitude)


        is_extrema = higra.attribute_extrema(new_tree, new_altitude)

        markers_mask = higra.reconstruct_leaf_data(new_tree, is_extrema)

        markers = label(markers_mask).astype(np.int32)

        thresh_val = threshold_otsu(filtred_image)

        is_background = filtred_image < thresh_val

        background_label = markers.max() + 1
        markers[is_background] = background_label

        #background_threshold = 0.3
        #is_background = filtred_image < background_threshold * 255

        #background_label = markers.max() + 1

        #markers[is_background] = background_label

        gradient = higra.weight_graph(graph, filtred_image, higra.WeightFunction.L1)

        segmentation = higra.labelisation_seeded_watershed(graph, gradient, markers)

        segmentation[segmentation == background_label] = 0

        segmentation = closing(segmentation, disk(2))

        vis_filename = os.path.join(output_dir, f"visualizacao_{mode}.png")
        save_image(segmentation, vis_filename)

        idx = i + 1
        masked_lbl = np.ma.masked_where(segmentation == 1, segmentation)
        ax[idx].imshow(image_rgb, alpha=0.3)
        ax[idx].imshow(masked_lbl, cmap='nipy_spectral', alpha=0.7, interpolation='nearest')
        ax[idx].set_title(f"Segmentation: {mode} (Saved)")
        ax[idx].axis('off')

    summary_path = os.path.join(output_dir, "resumo_comparativo.png")
    plt.tight_layout()
    plt.savefig(summary_path)
    print(f"--- EXPERIMENTO FINALIZADO ---")
    print(f"Todas as amostras foram arquivadas na pasta: '{output_dir}'")
    plt.show()

    return fig, segmentation

    
