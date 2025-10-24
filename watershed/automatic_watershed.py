# automatic_watershed.py

import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, dilation, disk
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes

def run_automatic_watershed(color_image, min_area_threshold):
    gray_image = color.rgb2gray(color_image)


    # --- 2. HIGRA SETUP: Create Graph and Edge Weights ---
    image_size = gray_image.shape
    graph = hg.get_4_adjacency_graph(image_size)
    edge_weights = hg.weight_graph(graph, gray_image, hg.WeightFunction.L1)


    # --- 3. PRE-PROCESSING: Create a Clean Binary Mask ---
    otsu_threshold = threshold_otsu(gray_image)
    binary_mask = gray_image < otsu_threshold
    selem = disk(1)
    opened_mask = opening(binary_mask, selem)


    # --- 4. FILTERING: Remove Small Unwanted Objects by Area ---
    labeled_mask = measure.label(opened_mask)
    regions_props = measure.regionprops(labeled_mask)

    # min_area_threshold = 180.0 

    filtered_mask = np.copy(labeled_mask)
    for prop in regions_props:
        if prop.area < min_area_threshold:
            filtered_mask[filtered_mask == prop.label] = 0

    binary_filtered_mask = filtered_mask > 0


    # --- 5. MARKER CREATION: The Key to a Perfect Watershed ---
    sure_foreground = binary_fill_holes(binary_filtered_mask)
    dilated_foreground = dilation(sure_foreground, selem)
    sure_background = (dilated_foreground == 0)
    foreground_labels = measure.label(sure_foreground)

    final_markers = np.zeros(gray_image.shape, dtype=np.int32)
    final_markers[sure_background] = 1
    final_markers[foreground_labels > 0] = foreground_labels[foreground_labels > 0] + 1


    # --- 6. FINAL SEGMENTATION: Run the Guided Watershed ---
    final_partition = hg.watershed.labelisation_seeded_watershed(graph, edge_weights, final_markers)
    if hasattr(final_partition, 'to_label_image'):
        final_segmentation = final_partition.to_label_image(image_size)
    else:
        final_segmentation = final_partition


    # --- 7. AREA ANALYSIS ---
    labels, area = np.unique(final_segmentation, return_counts=True)
    total_tissue_area = 0
    component_areas = {}

    for label, area in zip(labels, area):
        if label > 1:
            component_number = label - 1
            component_areas[component_number] = area
            total_tissue_area += area

    # --- 8. VISUALIZATION ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tissue Segmentation and Area Analysis (Automatic Higra Method)', fontsize=20)
    ax = axes.ravel()

    ax[0].imshow(color_image)
    ax[0].set_title("1. Original Image")
    ax[0].axis('off')

    ax[1].imshow(binary_filtered_mask, cmap='gray')
    ax[1].set_title("2. Cleaned & Filtered Mask")
    ax[1].axis('off')

    ax[2].imshow(final_segmentation, cmap='nipy_spectral')
    ax[2].set_title("3. Final Segmentation")
    ax[2].axis('off')

    ax[3].axis('off')
    ax[3].set_title("4. Area Analysis")

    sorted_items = sorted(component_areas.items())
    mid_point = len(sorted_items) // 2 + (len(sorted_items) % 2)

    left_column_items = sorted_items[:mid_point]
    right_column_items = sorted_items[mid_point:]

    left_column_str = "\n".join([f"Comp {num}: {area} px" for num, area in left_column_items])
    right_column_str = "\n".join([f"Comp {num}: {area} px" for num, area in right_column_items])

    ax[3].text(0.05, 0.80, left_column_str, fontsize=10, fontfamily='monospace', verticalalignment='top')
    ax[3].text(0.55, 0.80, right_column_str, fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    total_area_str = f"\n\nTotal Tissue Area: {total_tissue_area} pixels"
    ax[3].text(0.30, 0.85, total_area_str, fontsize=10, fontweight='bold', verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, final_segmentation