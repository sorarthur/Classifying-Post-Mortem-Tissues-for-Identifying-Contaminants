# automatic_watershed.py (With Color Key Visualization & MANUAL THRESHOLD)

import higra as hg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Import patches for color key
from matplotlib import colormaps
from skimage import io, color, measure
from skimage.filters import threshold_otsu # We still need this for the default case
from skimage.morphology import opening, closing, dilation, disk, remove_small_objects # Added remove_small_objects
# from skimage.measure import regionprops # regionprops not needed anymore
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
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

# Removed blur_ksize and min_peak_dist parameters as they are not used in this version
#
# !!! --- MODIFICATION 1: Added 'manual_threshold=None' --- !!!
#
def run_automatic_watershed(color_image, min_area_threshold, manual_threshold=None, watershed_method='skimage'):
    """
    Performs automatic seeded watershed segmentation using sure foreground/background markers.

    Args:
        color_image (np.ndarray): The input color image (RGB).
        min_area_threshold (int): Minimum area threshold for identifying main components
                                      and for final component filtering.
        manual_threshold (float, optional): A user-defined threshold (0.0-1.0). 
                                            If None, Otsu's method is used.

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
    # Separate potential foreground (tissue) from background.
    
    if manual_threshold is not None:
        threshold_value = manual_threshold
        print(f"--- Using MANUAL threshold: {threshold_value} ---")
    else:
        # Otherwise, calculate using Otsu's method
        threshold_value = threshold_otsu(gray_image)
        print(f"--- Using AUTOMATIC (Otsu) threshold: {threshold_value} ---")
        
    # Pixels LESS than the threshold are foreground (dark objects)
    # This is what you want: dark purple (low value) < threshold
    binary_mask = gray_image < threshold_value
    
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
    if(watershed_method == 'skimage'):
        final_partition = watershed(gray_image, markers=final_markers, mask=binary_filtered_mask)
    else:
        final_partition = hg.watershed.labelisation_seeded_watershed(graph, edge_weights, final_markers)
    # Convert result to a standard NumPy array if necessary.
    if hasattr(final_partition, 'to_label_image'):
        output_image_for_display = final_partition.to_label_image(image_size)
    else:
        output_image_for_display = final_partition
    # Remove the explicit background label (1) assigned during marker creation.
    output_image_for_display[output_image_for_display == 1] = 0
    # print("Final labels after watershed (BG removed):", np.unique(output_image_for_display)) # Verbose

    # === Block 7: Post-processing - Remove Small Components ===
    post_processed_filled = np.copy(output_image_for_display)
    component_labels = np.unique(output_image_for_display)
    if len(component_labels) > 0 and component_labels[0] == 0: component_labels = component_labels[1:] # Exclude background
    print(f"Filling holes for {len(component_labels)} components...")
    for label in component_labels:
        component_mask = (output_image_for_display == label)
        filled_component_mask = binary_fill_holes(component_mask)
        post_processed_filled[filled_component_mask] = label # Re-apply label to filled areas

    # Remove final components smaller than `min_area_threshold` (e.g., noise, small artifacts).
    print(f"Filtering final components smaller than {min_area_threshold} pixels...")
    # `remove_small_objects` efficiently removes labeled regions below the size threshold.
    final_filtered_image = remove_small_objects(post_processed_filled, min_size=min_area_threshold)
    final_labels_after_filter = np.unique(final_filtered_image)
    print("Final labels after area filtering:", final_labels_after_filter)
    output_image_for_display = final_filtered_image # Final result


    # === Block 8: Area Analysis ===
    # Calculate the area for each final segmented component (label > 0).
    labels, areas = np.unique(output_image_for_display, return_counts=True)
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

    # === Block 9: Visualization with Color Key ===
    # Generate the output figure including the color key legend.
    print("Generating final visualization with custom distinct color key...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tissue Segmentation (Manual Threshold Test)', fontsize=16) # Title Updated
    ax = axes.ravel()

    # Plot 1: Original Image
    ax[0].imshow(color_image)
    ax[0].set_title("1. Original Image")
    ax[0].axis('off')

    # Plot 2: Cleaned Foreground Mask
    # We plot 'opened_mask' now to see the direct result of the thresholding
    ax[1].imshow(opened_mask, cmap='gray')
    ax[1].set_title(f"2. Binary Mask (Thresh={threshold_value:.3f})")
    ax[1].axis('off')

    # --- Plot 3: Final Segmented Image (CUSTOM Distinct Colors) ---
    num_final_labels = len(valid_labels)
    # Create an empty RGB image (black background)
    colored_segmentation_rgb = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    # Dictionary to store the mapping: label -> color
    color_map_dict = {}

    if num_final_labels > 0:
        print(f"Mapping {num_final_labels} labels to {NUM_CUSTOM_COLORS} custom colors (cycling)...")

        # Assign colors to each valid label using the custom list
        for i, label in enumerate(valid_labels):
            color_index = i % NUM_CUSTOM_COLORS # Cycle through custom colors
            rgb_color = CUSTOM_DISTINCT_COLORS_RGB[color_index] # Get RGB tuple
            color_map_dict[label] = np.array(rgb_color) # Store as numpy array
            # Apply color to the corresponding pixels
            colored_segmentation_rgb[output_image_for_display == label] = rgb_color

        ax[2].imshow(colored_segmentation_rgb)
    else:
        ax[2].imshow(output_image_for_display, cmap='gray') # Show black if no components

    ax[2].set_title("3. Final Segmentation (Custom Colors)") # Title Updated
    ax[2].axis('off')

    # --- Plot 4: Area Analysis Text with CUSTOM Distinct Color Key (Two Columns) ---
    ax[3].axis('off')
    ax[3].set_title("4. Final Area Analysis (with Color Key)")
    ax[3].set_ylim(0, 1)
    ax[3].set_xlim(0, 1)

    if component_areas and num_final_labels > 0:
        sorted_labels = sorted(component_areas.keys())

        # --- Logic for Two Columns ---
        mid_point = len(sorted_labels) // 2 + (len(sorted_labels) % 2)
        items_col1 = sorted_labels[:mid_point]
        items_col2 = sorted_labels[mid_point:]

        y_pos_start = 0.95
        y_step = 0.045
        color_box_size = 0.04

        # Column 1
        x_pos_color1 = 0.05
        x_pos_text1 = 0.12
        current_y = y_pos_start
        for label in items_col1:
            area = component_areas[label]
            # Get color from CUSTOM dictionary (convert 0-255 to 0-1 for patch)
            rgb_color_0_255 = color_map_dict.get(label, np.array([0,0,0]))
            color_val_0_1 = rgb_color_0_255 / 255.0
            rect = patches.Rectangle((x_pos_color1, current_y - color_box_size),
                                     color_box_size, color_box_size, facecolor=color_val_0_1)
            ax[3].add_patch(rect)
            text = f"Label {label}: {area} px" # Using "Label" for automatic mode
            ax[3].text(x_pos_text1, current_y - color_box_size/2, text,
                       fontsize=14, fontfamily='monospace', verticalalignment='center')
            current_y -= y_step

        # Column 2
        x_pos_color2 = 0.55
        x_pos_text2 = 0.62
        current_y = y_pos_start
        for label in items_col2:
            area = component_areas[label]
            # Get color from CUSTOM dictionary (convert 0-255 to 0-1 for patch)
            rgb_color_0_255 = color_map_dict.get(label, np.array([0,0,0]))
            color_val_0_1 = rgb_color_0_255 / 255.0
            rect = patches.Rectangle((x_pos_color2, current_y - color_box_size),
                                     color_box_size, color_box_size, facecolor=color_val_0_1)
            ax[3].add_patch(rect)
            text = f"Label {label}: {area} px" # Using "Label"
            ax[3].text(x_pos_text2, current_y - color_box_size/2, text,
                       fontsize=14, fontfamily='monospace', verticalalignment='center')
            current_y -= y_step
        # --- End of Two Column Logic ---

        # Display Total Area
        total_area_str = f"Total Segmented Area: {total_tissue_area} pixels"
        ax[3].text(0.5, 0.05, total_area_str, fontsize=12, fontweight='bold', ha='center', verticalalignment='bottom')

    else:
        # Message if no components remain
        ax[3].text(0.5, 0.5, "No components found\nafter filtering.", ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust overall layout

    # Return the figure and the final postprocessed segmented mask
    return fig, output_image_for_display