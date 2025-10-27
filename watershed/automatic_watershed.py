# automatic_watershed.py (With Color Key Visualization)

import higra as hg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Import patches for color key
from skimage import io, color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, dilation, disk, remove_small_objects # Added remove_small_objects
# from skimage.measure import regionprops # regionprops not needed anymore
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi # Keep ndi import if needed elsewhere, though distance transform is removed
# from skimage.feature import peak_local_max # peak_local_max removed

# Removed blur_ksize and min_peak_dist parameters as they are not used in this version
def run_automatic_watershed(color_image, min_area_threshold):
    """
    Performs automatic seeded watershed segmentation using sure foreground/background markers.

    Args:
        color_image (np.ndarray): The input color image (RGB).
        min_area_threshold (int): Minimum area threshold for identifying main components
                                   and for final component filtering.

    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray) Figure with visualizations and the final segmented label image.
    """

    # === Block 1: Initial Image Conversion ===
    # Convert input image to grayscale for processing.
    gray_image = color.rgb2gray(color_image)
    image_size = gray_image.shape
    # print(f"Image loaded. Size: {image_size}") # Verbose

    # === Block 2: HIGRA Graph and Edge Weight Setup ===
    # Create the graph representation of the image grid.
    # print("Creating adjacency graph...") # Verbose
    graph = hg.get_4_adjacency_graph(image_size)
    # Calculate edge weights based on L1 difference on the ORIGINAL grayscale image.
    # print("Calculating edge weights on original image...") # Verbose
    edge_weights = hg.weight_graph(graph, gray_image, hg.WeightFunction.L1)


    # === Block 3: Pre-processing - Create Initial Binary Mask ===
    # Separate potential foreground (tissue) from background using Otsu's threshold.
    otsu_threshold = threshold_otsu(gray_image)
    binary_mask = gray_image < otsu_threshold
    # Apply morphological opening to remove small noise from the initial mask.
    selem = disk(1) # Small structuring element for noise removal
    opened_mask = opening(binary_mask, selem)


    # === Block 4: Filtering - Remove Small Objects ===
    # Label connected components in the opened mask.
    labeled_mask = measure.label(opened_mask)
    # Filter out components smaller than the specified threshold.
    # print(f"Filtering initial components smaller than {min_area_threshold} pixels...") # Verbose
    # Use remove_small_objects for efficiency
    filtered_mask_labels = remove_small_objects(labeled_mask, min_size=min_area_threshold)
    binary_filtered_mask = filtered_mask_labels > 0 # Final mask of significant foreground regions


    # === Block 5: Marker Creation (Sure Foreground/Background) ===
    # Automatically generate markers for the seeded watershed based on the filtered mask.
    # print("Generating watershed markers...") # Verbose
    # Sure Foreground: Fill holes in the filtered mask to get solid regions.
    sure_foreground = binary_fill_holes(binary_filtered_mask)
    # Sure Background: Dilate the foreground slightly; everything outside is sure background.
    dilated_foreground = dilation(sure_foreground, selem) # Use the same small selem
    sure_background = (dilated_foreground == 0)
    # Label the sure foreground regions to create initial seeds (labels 1, 2, ...).
    foreground_labels, num_fg_labels = measure.label(sure_foreground, return_num=True)
    # print(f"Found {num_fg_labels} foreground markers.") # Verbose

    # Combine markers: Background = 1, Foreground = 2, 3, ...
    final_markers = np.zeros(gray_image.shape, dtype=np.int32)
    final_markers[sure_background] = 1
    # Add foreground labels, offsetting them by 1.
    final_markers[foreground_labels > 0] = foreground_labels[foreground_labels > 0] + 1


    # === Block 6: Final Segmentation - Run Seeded Watershed ===
    # Execute Higra's seeded watershed using the graph, edge weights (from original gray),
    # and the automatically generated combined markers.
    # print("Executing seeded watershed...") # Verbose
    final_partition = watershed(gray_image, markers=final_markers, mask=binary_filtered_mask)
    # Convert result to a standard NumPy array if necessary.
    if hasattr(final_partition, 'to_label_image'):
        final_segmentation = final_partition.to_label_image(image_size)
    else:
        final_segmentation = final_partition
    # Remove the explicit background label (1) assigned during marker creation.
    final_segmentation[final_segmentation == 1] = 0
    # print("Final labels after watershed (BG removed):", np.unique(final_segmentation)) # Verbose


    # === Block 7: Area Analysis ===
    # Calculate the area for each final segmented component (label > 0).
    labels, areas = np.unique(final_segmentation, return_counts=True)
    total_tissue_area = 0
    component_areas = {}
    valid_labels = [] # Store labels of components that actually exist
    for label, area in zip(labels, areas):
        # We process labels > 0 because background label 1 was removed in step 6
        if label > 0:
            component_areas[label] = area
            total_tissue_area += area
            valid_labels.append(label)
    # print("Final calculated areas:", component_areas) # Verbose

    # === Block 8: Visualization with Color Key ===
    # Generate the output figure including the color key legend.
    print("Generating final visualization with color key...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)) # Back to 2x2 layout
    fig.suptitle('Tissue Segmentation (Automatic - Sure FG/BG Markers)', fontsize=16) # Updated title
    ax = axes.ravel() # Flatten axes array

    # Plot 1: Original Image
    ax[0].imshow(color_image)
    ax[0].set_title("1. Original Image")
    ax[0].axis('off')

    # Plot 2: Cleaned Foreground Mask (input for sure_foreground)
    ax[1].imshow(binary_filtered_mask, cmap='gray')
    ax[1].set_title("2. Cleaned Foreground Mask")
    ax[1].axis('off')

    # Plot 3: Final Segmented Image (Colored)
    num_final_labels = len(valid_labels)
    cmap = plt.cm.nipy_spectral # Define colormap

    if num_final_labels > 0:
        min_label = min(valid_labels)
        max_label = max(valid_labels)
        # Adjust normalization to prevent darkest color for the lowest label
        if min_label == max_label:
            norm = plt.Normalize(vmin=min_label - 1, vmax=max_label + 1)
        else:
            norm = plt.Normalize(vmin=min_label - 0.5, vmax=max_label)

        # Apply colormap
        colored_segmentation = cmap(norm(final_segmentation))
        colored_segmentation[final_segmentation == 0] = [0, 0, 0, 1] # Ensure background is black
        ax[2].imshow(colored_segmentation)
    else:
        ax[2].imshow(final_segmentation, cmap='gray') # Show black if no components

    ax[2].set_title("3. Final Segmentation")
    ax[2].axis('off')

    # Plot 4: Area Analysis Text with Color Key (Two Columns)
    ax[3].axis('off')
    ax[3].set_title("4. Final Area Analysis (with Color Key)")
    ax[3].set_ylim(0, 1)
    ax[3].set_xlim(0, 1)

    if component_areas and num_final_labels > 0:
        # Get the same colormap and normalization used for the image plot
        cmap_func = cmap
        norm_func = norm # Use the adjusted norm created above

        sorted_labels = sorted(component_areas.keys())

        # --- Logic for Two Columns ---
        mid_point = len(sorted_labels) // 2 + (len(sorted_labels) % 2)
        items_col1 = sorted_labels[:mid_point]
        items_col2 = sorted_labels[mid_point:]

        y_pos_start = 0.95
        y_step = 0.045 # Adjust vertical spacing if needed
        color_box_size = 0.04

        # Column 1
        x_pos_color1 = 0.05
        x_pos_text1 = 0.12
        current_y = y_pos_start
        for label in items_col1:
            area = component_areas[label]
            color_val = cmap_func(norm_func(label)) # Get color using norm
            rect = patches.Rectangle((x_pos_color1, current_y - color_box_size),
                                     color_box_size, color_box_size, facecolor=color_val)
            ax[3].add_patch(rect)
            # Display actual label number
            text = f"Label {label}: {area} px"
            ax[3].text(x_pos_text1, current_y - color_box_size/2, text,
                       fontsize=14, fontfamily='monospace', verticalalignment='center')
            current_y -= y_step
            # if current_y < 0.15: break # Optional break if list is very long

        # Column 2
        x_pos_color2 = 0.55
        x_pos_text2 = 0.62
        current_y = y_pos_start # Reset Y for the second column
        for label in items_col2:
            area = component_areas[label]
            color_val = cmap_func(norm_func(label)) # Get color using norm
            rect = patches.Rectangle((x_pos_color2, current_y - color_box_size),
                                     color_box_size, color_box_size, facecolor=color_val)
            ax[3].add_patch(rect)
            text = f"Label {label}: {area} px"
            ax[3].text(x_pos_text2, current_y - color_box_size/2, text,
                       fontsize=14, fontfamily='monospace', verticalalignment='center')
            current_y -= y_step
            # if current_y < 0.15: break # Optional break if list is very long
        # --- End of Two Column Logic ---

        # Display Total Area
        total_area_str = f"Total Segmented Area: {total_tissue_area} pixels"
        ax[3].text(0.5, 0.05, total_area_str, fontsize=12, fontweight='bold', ha='center', verticalalignment='bottom')

    else:
        # Message if no components remain after filtering
        ax[3].text(0.5, 0.5, "No components found\nafter filtering.", ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust overall layout

    # Return the figure and the final segmented mask
    # Return final_filtered_image (or output_image_for_display if defined) if needed
    # For now, matching the original return value
    return fig, final_segmentation