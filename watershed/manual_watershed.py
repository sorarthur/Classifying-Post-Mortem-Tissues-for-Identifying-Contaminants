# manual_watershed.py
# Implements manual seeded watershed using Higra, incorporating a "divide and conquer"
# strategy by processing major components individually.

import higra as hg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For color key in plot
from matplotlib import colormaps
from skimage import color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk, dilation, remove_small_objects
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi
import cv2 # For GaussianBlur

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

def run_manual_watershed(color_image, min_area_threshold, markers_df, blur_ksize=5):
    """
    Performs manual seeded watershed segmentation using a divide-and-conquer approach.

    Args:
        color_image (np.ndarray): The input color image (RGB).
        min_area_threshold (int): Minimum area threshold for identifying major components
                                   and for final component filtering.
        markers_df (pd.DataFrame): DataFrame with marker coordinates ('x', 'y', 'type') from the interface.
        blur_ksize (int): Kernel size for Gaussian blur applied before edge weight calculation.

    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray) Figure with visualizations and the final segmented label image.
    """

    # === Block 1: Initial Image Conversion ===
    # Convert the input RGB image to grayscale for intensity-based processing.
    # Store image size for graph creation and coordinate validation.
    gray_image = color.rgb2gray(color_image)
    image_size = gray_image.shape
    print(f"Image loaded. Size: {image_size}")

    # === Block 2: Higra Graph and Edge Weight Setup ===
    # Apply Gaussian blur to the grayscale image. This smooths noise and texture,
    # potentially helping the watershed algorithm flood regions more consistently.
    # The edge weights will be calculated on this blurred image.
    ksize = max(1, int(blur_ksize))
    if ksize % 2 == 0: ksize += 1 # Ensure kernel size is odd for cv2.GaussianBlur
    print(f"Applying Gaussian Blur (kernel size: {ksize}x{ksize})...")
    blurred_gray = cv2.GaussianBlur(gray_image.astype(np.float32), (ksize, ksize), 0)

    # Create a 4-adjacency graph where each pixel is a node connected to its immediate neighbors.
    print("Creating adjacency graph...")
    graph = hg.get_4_adjacency_graph(image_size)
    # Calculate weights for each edge in the graph based on the absolute intensity difference (L1 norm)
    # between connected pixels in the blurred image. Lower weights represent 'easier' paths for flooding.
    print("Calculating edge weights on blurred image...")
    edge_weights = hg.weight_graph(graph, blurred_gray, hg.WeightFunction.L1)

    # === Block 3: Pre-processing - Identify Major Components ===
    # Identify the main, large, disconnected regions of tissue that will be processed individually.
    # This helps prevent watershed flooding from leaking between major sections.
    print("Identifying major components for processing...")
    # Apply Otsu's threshold to the original (non-blurred) grayscale image to get an initial binary mask
    # separating potential tissue from background.
    otsu_threshold = threshold_otsu(gray_image)
    binary_mask = gray_image < otsu_threshold
    # Apply morphological opening (erosion followed by dilation) to remove small noise artifacts
    # (e.g., single pixels or tiny specks) from the binary mask.
    selem_open = disk(1) # Structuring element (disk shape, radius 2) for opening
    opened_mask = opening(binary_mask, selem_open)
    # Label each connected region in the opened mask with a unique integer.
    labeled_mask_initial = measure.label(opened_mask)

    # Filter the labeled regions, keeping only those larger than the specified `min_area_threshold`.
    # This defines the "major components" that are significant enough to process.
    main_component_mask = np.copy(labeled_mask_initial)
    regions_props = measure.regionprops(labeled_mask_initial) # Get region properties
    for prop in regions_props:
        if prop.area < min_area_threshold:
            main_component_mask[main_component_mask == prop.label] = 0 # Set small regions to 0

    # Re-label the remaining components to ensure consecutive labels (1, 2, 3...).
    main_components, num_components = measure.label(main_component_mask > 0, return_num=True)
    print(f"Found {num_components} major components to process.")

    # Handle the case where no significant components are found.
    if num_components == 0:
        print("Error: No major components found after initial filtering.")
        fig, _ = plt.subplots()
        plt.text(0.5, 0.5, 'Error: No major components found.', ha='center', va='center')
        return fig, np.zeros(image_size, dtype=np.int32)

    # === Block 4: Marker Validation ===
    # Filter the user-provided markers (from `markers_df`) to keep only those
    # that fall within the boundaries of the identified major components.
    valid_markers_coords = []
    if not markers_df.empty:
        print("Validating received markers...")
        for index, row in markers_df.iterrows():
            # Process only markers identified as 'circle' (from the canvas 'point' tool)
            if row['type'] == 'circle':
                x, y = int(row['x']), int(row['y'])
                # Check if coordinates are within the image dimensions
                if 0 <= y < image_size[0] and 0 <= x < image_size[1]:
                    # Check if the marker coordinate falls within a labeled major component (label > 0)
                    if main_components[y, x] > 0:
                         valid_markers_coords.append({'x': x, 'y': y, 'label': -1}) # Store valid coords
                         # Verbose logging can be added here if needed
                # Markers outside bounds or major components are ignored
        print(f"Total valid markers (within major components): {len(valid_markers_coords)}")

    # Handle the case where no valid markers were provided or found within components.
    if not valid_markers_coords:
         print("Error: No valid markers found within the major components.")
         fig, _ = plt.subplots()
         plt.text(0.5, 0.5, 'Error: No valid markers found.', ha='center', va='center')
         return fig, np.zeros(image_size, dtype=np.int32)

    # === Block 5: "Divide and Conquer" Segmentation Loop ===
    # Process each major component individually to perform detailed segmentation within it.
    final_segmented_image = np.zeros(image_size, dtype=np.int32) # Accumulates results
    label_offset = 0 # Ensures globally unique labels for sub-components
    processed_labels = set() # Tracks used global labels

    print(f"Starting segmentation loop for {num_components} components...")
    for i in range(1, num_components + 1): # Loop through major components (labeled 1 to num_components)
        # --- 5a. Isolate Component and Local Markers ---
        # Create a boolean mask isolating the current major component.
        current_mask = (main_components == i)
        # print(f"\n--- Processing Major Component {i} ---") # Verbose

        # Select only the validated markers that fall within this specific component.
        local_markers_coords = []
        for marker in valid_markers_coords:
            if current_mask[marker['y'], marker['x']]: # Check against the component's mask
                local_markers_coords.append((marker['y'], marker['x'])) # Store as (y, x) tuples
        # print(f"  Markers for this component: {len(local_markers_coords)}") # Verbose

        # --- 5b. Create Local Marker Image ---
        # Generate a labeled image containing seeds (markers) only for this component.
        if len(local_markers_coords) > 0:
            # Create a boolean image with points at marker locations.
            local_peak_markers = np.zeros(image_size, dtype=bool)
            for y, x in local_markers_coords:
                local_peak_markers[y, x] = True

            # Label individual points *before* dilation to assign unique IDs.
            labeled_individual_local, num_local_initial = ndi.label(local_peak_markers)
            marker_radius = 2 # Dilation radius
            local_labeled_markers_fg = np.zeros(image_size, dtype=np.int32) # Initialize local labeled marker image
            # Dilate each labeled point individually into a small region.
            for lbl_local in range(1, num_local_initial + 1):
                 single_marker_mask = (labeled_individual_local == lbl_local)
                 dilated_single_marker = dilation(single_marker_mask, disk(marker_radius))
                 # Assign the unique label only within the current component's mask and where no label exists yet.
                 target_mask = dilated_single_marker & (local_labeled_markers_fg == 0) & current_mask
                 local_labeled_markers_fg[target_mask] = lbl_local
                 local_labeled_markers_fg[single_marker_mask & current_mask] = lbl_local # Ensure center has label

            # Count effective seeds within the current component mask.
            num_local_regions = len(np.unique(local_labeled_markers_fg[current_mask])) - 1 # Exclude 0
            # Handle cases where dilation might fail or go outside the mask.
            if num_local_regions <= 0:
                 print(f"  Warning: No effective local markers for component {i}. Treating as single region.")
                 local_labeled_markers_fg[current_mask] = 1 # Fallback: label the whole component as 1
                 num_local_regions = 1
            # print(f"  Number of local seeds: {num_local_regions}") # Verbose

        else:
            # If the user didn't place markers in this major component, treat it as one region.
            print(f"  No specific markers for component {i}. Treating as single region.")
            local_labeled_markers_fg = np.zeros(image_size, dtype=np.int32)
            local_labeled_markers_fg[current_mask] = 1 # Label the whole component area as 1
            num_local_regions = 1

        # --- 5c. Apply Local Seeded Watershed ---
        # Run Higra's watershed algorithm using the graph, edge weights (from blurred image),
        # and the locally generated labeled markers (local_labeled_markers_fg).
        # print("  Executing local watershed...") # Verbose
        partition_local = watershed(gray_image, markers=local_labeled_markers_fg, mask=current_mask, connectivity=1)
        segmentation_local_full = partition_local.to_label_image(image_size) if hasattr(partition_local, 'to_label_image') else partition_local

        # --- 5d. Mask Watershed Result ---
        # Restrict the watershed segmentation result strictly to the area of the current major component.
        # This prevents flooding from leaking outside the intended region.
        local_segmentation_masked = np.zeros(image_size, dtype=np.int32)
        local_segmentation_masked[current_mask] = segmentation_local_full[current_mask]

        # --- 5e. Add to Final Image with Offset ---
        # Integrate the segmented sub-components into the global result image,
        # ensuring all sub-components have unique labels across the entire image.
        new_labels = np.unique(local_segmentation_masked)
        if len(new_labels) > 0 and new_labels[0] == 0: new_labels = new_labels[1:] # Get local labels (1, 2, ...)

        # Calculate the appropriate offset to avoid label collisions with previously processed components.
        current_offset = label_offset
        while any((lbl + current_offset) in processed_labels for lbl in new_labels if (lbl + current_offset) != 0):
             current_offset += num_local_regions # Find next available block of labels

        # print(f"  Applying offset: {current_offset}. Original local labels: {new_labels}") # Verbose

        # Apply the offset to the non-zero labels and add them to the final image.
        mask_to_update = local_segmentation_masked > 0
        # Use += here assumes final_segmented_image is initially all zeros.
        final_segmented_image[mask_to_update] += (local_segmentation_masked[mask_to_update] + current_offset)

        # Record the newly used global labels and update the base offset for the next component.
        for lbl in new_labels:
             processed_labels.add(lbl + current_offset)
        label_offset = max(processed_labels) if processed_labels else 0
        # print(f"  Global labels after component {i}: {np.unique(final_segmented_image)}") # Verbose


    # === Block 6: Post-processing on Combined Image ===
    # Apply final cleanup steps to the composite image containing results from all components.
    print("\nStarting final post-processing...")

    # --- 6a. Hole Filling ---
    # Fill any holes that might exist within the final segmented regions.
    post_processed_filled = np.copy(final_segmented_image)
    component_labels = np.unique(final_segmented_image)
    if len(component_labels) > 0 and component_labels[0] == 0: component_labels = component_labels[1:] # Exclude background
    print(f"Filling holes for {len(component_labels)} components...")
    for label in component_labels:
        component_mask = (final_segmented_image == label)
        filled_component_mask = binary_fill_holes(component_mask)
        post_processed_filled[filled_component_mask] = label # Re-apply label to filled areas

    # --- 6b. Minimum Area Filtering ---
    # Remove final components smaller than `min_area_threshold` (e.g., noise, small artifacts).
    print(f"Filtering final components smaller than {min_area_threshold} pixels...")
    # `remove_small_objects` efficiently removes labeled regions below the size threshold.
    final_filtered_image = remove_small_objects(post_processed_filled, min_size=min_area_threshold)
    final_labels_after_filter = np.unique(final_filtered_image)
    print("Final labels after area filtering:", final_labels_after_filter)
    output_image_for_display = final_filtered_image # Final result


    # === Block 7: Area Analysis ===
    # Calculate the area (pixel count) for each component remaining after all processing.
    labels, areas = np.unique(output_image_for_display, return_counts=True)
    component_areas = {}
    total_tissue_area = 0
    valid_labels = [] # Store labels of components that actually exist after filtering
    for label, area in zip(labels, areas):
        if label > 0: # Ignore background
            component_areas[label] = area
            total_tissue_area += area
            valid_labels.append(label)
    print("Final calculated areas:", component_areas)


    # === Block 8: Visualization with Color Key ===
    # Generate the output figure displaying intermediate steps and the final result with area analysis.
    # === Block 8: Visualization with Color Key and Two Columns ===
    print("Generating final visualization with custom distinct color key...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tissue Segmentation (Manual - Custom Colors)', fontsize=16) # Title Updated
    ax = axes.ravel()

    # Plot 1: Original Image
    ax[0].imshow(color_image)
    ax[0].set_title("1. Original Image")
    ax[0].axis('off')

    # Plot 2: Identified Major Components
    ax[1].imshow(main_components, cmap='nipy_spectral') # Keep spectral here, just identifies regions
    ax[1].set_title("2. Major Components Identified")
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
            # Cycle through the custom color list
            color_index = i % NUM_CUSTOM_COLORS
            # Get the RGB color tuple (0-255)
            rgb_color = CUSTOM_DISTINCT_COLORS_RGB[color_index]

            # Store in dictionary
            color_map_dict[label] = np.array(rgb_color) # Store as numpy array

            # Apply color to the corresponding pixels in the RGB image
            colored_segmentation_rgb[output_image_for_display == label] = rgb_color

        ax[2].imshow(colored_segmentation_rgb)
    else:
        # Show black image if no components remain
        ax[2].imshow(output_image_for_display, cmap='gray')

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
        y_step = 0.045 # Adjust vertical spacing if needed
        color_box_size = 0.04

        # Column 1
        x_pos_color1 = 0.05
        x_pos_text1 = 0.12
        current_y = y_pos_start
        for label in items_col1:
            area = component_areas[label]
            # Get color from the CUSTOM dictionary (divide by 255.0 for patch)
            rgb_color_0_255 = color_map_dict.get(label, np.array([0,0,0]))
            color_val_0_1 = rgb_color_0_255 / 255.0
            rect = patches.Rectangle((x_pos_color1, current_y - color_box_size),
                                     color_box_size, color_box_size, facecolor=color_val_0_1)
            ax[3].add_patch(rect)
            text = f"Comp {label}: {area} px"
            ax[3].text(x_pos_text1, current_y - color_box_size/2, text,
                       fontsize=14, fontfamily='monospace', verticalalignment='center')
            current_y -= y_step

        # Column 2
        x_pos_color2 = 0.55
        x_pos_text2 = 0.62
        current_y = y_pos_start # Reset Y
        for label in items_col2:
            area = component_areas[label]
            # Get color from the CUSTOM dictionary (divide by 255.0 for patch)
            rgb_color_0_255 = color_map_dict.get(label, np.array([0,0,0]))
            color_val_0_1 = rgb_color_0_255 / 255.0
            rect = patches.Rectangle((x_pos_color2, current_y - color_box_size),
                                     color_box_size, color_box_size, facecolor=color_val_0_1)
            ax[3].add_patch(rect)
            text = f"Comp {label}: {area} px"
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Return the figure and the final segmented mask
    return fig, output_image_for_display