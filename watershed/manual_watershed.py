# manual_watershed.py
# Implements the manual seeded watershed segmentation using a "divide and conquer" approach.

import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure
from skimage.filters import threshold_otsu
# from skimage.segmentation import watershed 
from skimage.morphology import opening, disk, dilation, remove_small_objects, binary_erosion # Added binary_erosion if needed later
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi
import cv2 # For GaussianBlur

def run_manual_watershed(color_image, min_area_threshold, markers_df, blur_ksize=5):
    """
    Performs manual seeded watershed segmentation.

    Args:
        color_image (np.ndarray): The input color image (RGB).
        min_area_threshold (int): Minimum area to keep components after final filtering.
        markers_df (pd.DataFrame): DataFrame containing marker coordinates ('x', 'y') from the interface.
        blur_ksize (int): Kernel size for Gaussian blur applied before edge weight calculation.

    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray) The figure with visualization plots and the final segmented label image.
    """
    
    # === 1. Initial Image Conversion ===
    # Convert the input color image to grayscale for intensity-based operations.
    gray_image = color.rgb2gray(color_image)
    image_size = gray_image.shape
    print(f"Image loaded. Size: {image_size}")

    # === 2. HIGRA Graph and Edge Weight Setup ===
    # Apply Gaussian blur to the grayscale image to reduce noise and internal texture variations,
    # which helps the watershed flood regions more completely.
    ksize = max(1, int(blur_ksize))
    if ksize % 2 == 0: ksize += 1 # Ensure kernel size is odd
    print(f"Applying Gaussian Blur (kernel size: {ksize}x{ksize})...")
    blurred_gray = cv2.GaussianBlur(gray_image.astype(np.float32), (ksize, ksize), 0)

    # Create a 4-adjacency graph representing pixel connectivity.
    print("Creating adjacency graph...")
    graph = hg.get_4_adjacency_graph(image_size)
    # Calculate edge weights based on the L1 difference (absolute intensity difference)
    # between adjacent pixels in the *blurred* image. Lower weights mean easier flooding.
    print("Calculating edge weights on blurred image...")
    edge_weights = hg.weight_graph(graph, blurred_gray, hg.WeightFunction.L1)

    # === 3. Pre-processing: Identify Major Processing Components ===
    # This step identifies the main, large tissue regions where the watershed will be applied locally.
    print("Identifying major components for processing...")
    # Apply Otsu's threshold on the *original* grayscale image to get a sharp initial mask.
    otsu_threshold = threshold_otsu(gray_image)
    binary_mask = gray_image < otsu_threshold
    # Apply morphological opening to remove small noise/particles from the binary mask.
    selem_open = disk(2) # Structuring element for opening
    opened_mask = opening(binary_mask, selem_open)
    # Label the connected regions in the opened mask.
    labeled_mask_initial = measure.label(opened_mask)

    # Filter out small regions based on the min_area_threshold from the interface
    # to define the main components that will guide the segmentation loop.
    main_component_mask = np.copy(labeled_mask_initial)
    regions_props = measure.regionprops(labeled_mask_initial) # Get properties for area filtering
    for prop in regions_props:
        if prop.area < min_area_threshold:
            main_component_mask[main_component_mask == prop.label] = 0 # Remove small components

    # Re-label the final main components ensuring consecutive labels starting from 1.
    main_components, num_components = measure.label(main_component_mask > 0, return_num=True)
    print(f"Found {num_components} major components to process.")

    # Error handling if no major components are found.
    if num_components == 0:
        print("Error: No major components found after initial filtering.")
        fig, _ = plt.subplots()
        plt.text(0.5, 0.5, 'Error: No major components found.', ha='center', va='center')
        return fig, np.zeros(image_size, dtype=np.int32) # Return empty results

    # === 4. Marker Validation ===
    # Validate the marker coordinates received from the interface (markers_df).
    # Keep only markers that fall within the bounds of the identified major components.
    valid_markers_coords = []
    if not markers_df.empty:
        print("Validating received markers...")
        for index, row in markers_df.iterrows():
            if row['type'] == 'circle': # Check if the drawn object is a 'circle' (used by 'point' tool)
                x, y = int(row['x']), int(row['y'])
                # Check if coordinates are within image dimensions
                if 0 <= y < image_size[0] and 0 <= x < image_size[1]:
                    # Check if the marker falls within any labeled major component region
                    if main_components[y, x] > 0:
                         valid_markers_coords.append({'x': x, 'y': y, 'label': -1}) # Store valid coords
                         # print(f"  Marker {index} (Coord: {x},{y}) - VALID (Inside Component {main_components[y, x]})") # Verbose logging
                    # else:
                         # print(f"  Marker {index} (Coord: {x},{y}) - IGNORED (Outside major components)") # Verbose logging
                # else:
                    # print(f"  Marker {index} (Coord: {x},{y}) - IGNORED (Outside image bounds)") # Verbose logging
        print(f"Total valid markers (within major components): {len(valid_markers_coords)}")

    # Error handling if no valid markers are found inside major components.
    if not valid_markers_coords:
         print("Error: No valid markers found within the major components.")
         fig, _ = plt.subplots()
         plt.text(0.5, 0.5, 'Error: No valid markers found.', ha='center', va='center')
         return fig, np.zeros(image_size, dtype=np.int32) # Return empty results

    # === 5. "Divide and Conquer" Segmentation Loop ===
    # Process each major component individually using the watershed algorithm.
    final_segmented_image = np.zeros(image_size, dtype=np.int32) # Initialize final output image
    label_offset = 0 # Offset to ensure unique labels across all components
    processed_labels = set() # Keep track of labels used so far

    print(f"Starting segmentation loop for {num_components} components...")
    for i in range(1, num_components + 1):
        # --- 5a. Isolate Current Component and its Markers ---
        # Create a mask for the current major component being processed.
        current_mask = (main_components == i)
        # print(f"\n--- Processing Major Component {i} ---") # Verbose logging

        # Filter the validated markers to get only those within the current component mask.
        local_markers_coords = []
        for marker in valid_markers_coords:
            if current_mask[marker['y'], marker['x']]:
                local_markers_coords.append((marker['y'], marker['x']))
        # print(f"  Markers found for this component: {len(local_markers_coords)}") # Verbose logging

        # --- 5b. Create Local Marker Image ---
        # Generate a labeled image containing seeds only for the current component.
        if len(local_markers_coords) > 0:
            local_peak_markers = np.zeros(image_size, dtype=bool)
            for y, x in local_markers_coords:
                local_peak_markers[y, x] = True

            # Label individual marker points *before* dilation to prevent merging.
            labeled_individual_local, num_local_initial = ndi.label(local_peak_markers)
            marker_radius = 2 # Radius for dilating markers into small regions
            local_labeled_markers_fg = np.zeros(image_size, dtype=np.int32)
            # Dilate each marker individually and assign its unique label within the current component mask.
            for lbl_local in range(1, num_local_initial + 1):
                 single_marker_mask = (labeled_individual_local == lbl_local)
                 dilated_single_marker = dilation(single_marker_mask, disk(marker_radius))
                 target_mask = dilated_single_marker & (local_labeled_markers_fg == 0) & current_mask
                 local_labeled_markers_fg[target_mask] = lbl_local
                 local_labeled_markers_fg[single_marker_mask & current_mask] = lbl_local # Reinforce center

            # Verify the number of effective seeds created within the mask.
            num_local_regions = len(np.unique(local_labeled_markers_fg[current_mask])) - 1 # Exclude 0
            if num_local_regions == 0:
                 print(f"  Warning: No effective local markers after dilation/masking for component {i}. Treating as single region.")
                 local_labeled_markers_fg[current_mask] = 1 # Treat component as one region if markers fail
                 num_local_regions = 1
            # print(f"  Number of local seeds (after dilation): {num_local_regions}") # Verbose logging

        else:
            # If no markers were clicked within this specific component, treat it as a single region.
            print(f"  No specific markers clicked for component {i}. Treating as single region.")
            local_labeled_markers_fg = np.zeros(image_size, dtype=np.int32)
            local_labeled_markers_fg[current_mask] = 1 # Assign label 1 to the whole component area
            num_local_regions = 1

        # --- 5c. Apply Local Seeded Watershed ---
        # Run Higra's watershed using the graph, calculated edge weights, and the local markers.
        # No explicit background marker is needed here because the result will be masked.
        # print("  Executing local watershed...") # Verbose logging
        partition_local = hg.watershed.labelisation_seeded_watershed(graph, edge_weights, local_labeled_markers_fg)
        # Convert the partition result (if needed) to a standard labeled NumPy array.
        segmentation_local_full = partition_local.to_label_image(image_size) if hasattr(partition_local, 'to_label_image') else partition_local

        # --- 5d. Mask Watershed Result ---
        # Crucially, restrict the watershed result to only the area of the current major component.
        local_segmentation_masked = np.zeros(image_size, dtype=np.int32)
        local_segmentation_masked[current_mask] = segmentation_local_full[current_mask]

        # --- 5e. Add to Final Image with Offset ---
        # Combine the masked local segmentation into the final global image, ensuring unique labels.
        new_labels = np.unique(local_segmentation_masked)
        if len(new_labels) > 0 and new_labels[0] == 0: new_labels = new_labels[1:] # Get only non-zero local labels (1, 2, ...)

        # Find the next available block of labels in the global image.
        current_offset = label_offset
        while any((lbl + current_offset) in processed_labels for lbl in new_labels if (lbl + current_offset) != 0):
             current_offset += num_local_regions # Increment offset if labels collide

        # print(f"  Applying offset: {current_offset}. Original local labels: {new_labels}") # Verbose logging

        # Apply the calculated offset to the non-zero labels of the local segmentation
        # and add them to the final composite image.
        mask_to_update = local_segmentation_masked > 0
        final_segmented_image[mask_to_update] += (local_segmentation_masked[mask_to_update] + current_offset)

        # Update the set of used labels and the base for the next offset.
        for lbl in new_labels:
             processed_labels.add(lbl + current_offset)
        label_offset = max(processed_labels) if processed_labels else 0 # Next offset starts after the highest label used
        # print(f"  Global labels after component {i}: {np.unique(final_segmented_image)}") # Verbose logging


    # === 6. Post-processing on Combined Image ===
    # Apply final cleanup steps to the fully combined segmented image.
    print("\nStarting final post-processing...")

    # --- 6a. Hole Filling ---
    # Fill any remaining holes within each final labeled component.
    post_processed_filled = np.copy(final_segmented_image)
    component_labels = np.unique(final_segmented_image)
    if len(component_labels) > 0 and component_labels[0] == 0: component_labels = component_labels[1:] # Exclude background
    print(f"Filling holes for {len(component_labels)} components...")
    for label in component_labels:
        component_mask = (final_segmented_image == label)
        filled_component_mask = binary_fill_holes(component_mask)
        post_processed_filled[filled_component_mask] = label # Re-apply the label to the filled area

    # --- 6b. Minimum Area Filtering ---
    # Remove small components (noise, artifacts) using the threshold from the interface.
    print(f"Filtering final components smaller than {min_area_threshold} pixels...")
    final_filtered_image = remove_small_objects(post_processed_filled, min_size=min_area_threshold)
    final_labels_after_filter = np.unique(final_filtered_image)
    print("Final labels after area filtering:", final_labels_after_filter)
    output_image_for_display = final_filtered_image # This is the final result mask


    # === 7. Area Analysis ===
    # Calculate the area (in pixels) for each remaining component in the final filtered image.
    labels, areas = np.unique(output_image_for_display, return_counts=True)
    component_areas = {}
    total_tissue_area = 0
    # print("Analyzing final areas for labels:", labels) # Verbose logging
    for label, area in zip(labels, areas):
        if label > 0: # Ignore background
            component_areas[label] = area
            total_tissue_area += area
    print("Final calculated areas:", component_areas)


    # === 8. Visualization ===
    # Generate a matplotlib figure showing the key steps and the final results.
    print("Generating final visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)) # Simple 2x2 layout
    fig.suptitle('Tissue Segmentation (Manual - Divide & Conquer)', fontsize=16)
    ax = axes.ravel() # Flatten axes array for easy indexing

    # Plot 1: Original Image
    ax[0].imshow(color_image)
    ax[0].set_title("1. Original Image")
    ax[0].axis('off')

    # Plot 2: Identified Major Components (the regions processed in the loop)
    ax[1].imshow(main_components, cmap='nipy_spectral')
    ax[1].set_title("2. Major Components Identified")
    ax[1].axis('off')

    # Plot 3: Final Segmented and Filtered Image
    ax[2].imshow(output_image_for_display, cmap='nipy_spectral')
    ax[2].set_title("3. Final Filtered Segmentation")
    ax[2].axis('off')

    # Plot 4: Area Analysis Text
    ax[3].axis('off')
    ax[3].set_title("4. Final Area Analysis")
    if component_areas:
        # Format area text nicely, potentially split into columns if many components
        display_areas = {f"Comp {lbl}": area for lbl, area in component_areas.items()}
        sorted_items = sorted(display_areas.items())
        max_items_per_col = 15 # Max items per column in the text display
        col_strs = []
        current_col = []
        for i, (name, area) in enumerate(sorted_items):
             current_col.append(f"{name}: {area} px")
             if len(current_col) >= max_items_per_col or i == len(sorted_items) - 1:
                  col_strs.append("\n".join(current_col))
                  current_col = []
        # Position text columns dynamically
        x_pos_step = 0.9 / max(1, len(col_strs))
        for i, col_str in enumerate(col_strs):
             ax[3].text(0.05 + i * x_pos_step, 0.95, col_str, fontsize=9, fontfamily='monospace', verticalalignment='top')
        # Add total area text
        total_area_str = f"\n\nTotal Segmented Area: {total_tissue_area} pixels"
        ax[3].text(0.5, 0.1, total_area_str, fontsize=10, fontweight='bold', ha='center', verticalalignment='bottom')
    else:
        # Display message if no components remain after filtering
        ax[3].text(0.5, 0.5, "No components found\nafter filtering.", ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Return the figure and the final segmented mask
    return fig, output_image_for_display