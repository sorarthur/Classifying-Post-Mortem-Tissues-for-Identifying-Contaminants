# interface.py
# Main script for the Streamlit application.

import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import base64
import cv2 # Ensure cv2 is imported if needed by validation_utils

# === Module Imports ===
# Imports watershed functions from the 'watershed' subfolder.
# Assumes 'automatic_watershed.py' and 'manual_watershed.py' are in './watershed/'
# and the folder contains an '__init__.py' file.
try:
    from watershed.automatic_watershed import run_automatic_watershed
    from watershed.manual_watershed import run_manual_watershed
except ModuleNotFoundError:
    st.error("Error: Ensure 'automatic_watershed.py' and 'manual_watershed.py' are in the 'watershed' subfolder and it contains an '__init__.py' file.")
    st.stop()
except ImportError as e:
    st.error(f"Erro ao importar 'run_automatic_watershed'. A sua função foi atualizada para aceitar 'global_threshold'? Erro: {e}")
    st.stop()


# Imports utility functions for validation from 'validation_utils.py'
# Assumes 'validation_utils.py' is in the same folder as this script.
try:
    from validation_utils import calculate_metrics, create_overlay, pil_to_cv2, cv2_to_rgb, maybe_downscale
except ModuleNotFoundError:
    st.error("Error: 'validation_utils.py' not found in the same folder as 'interface.py'.")
    st.stop()
except ImportError as e:
    st.error(f"Error importing from 'validation_utils.py': {e}. Check the functions within the file.")
    st.stop()

# === Helper Function ===
# Converts a PIL Image to a Base64 data URL.
# Necessary for compatibility with st_canvas background_image in some Streamlit versions.
def pil_to_data_url(pil_img: Image.Image) -> str:
    """Converts a PIL image to a Base64 data URL for web display."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# === Streamlit Page Configuration ===
# Sets the title and layout for the Streamlit application page.
st.set_page_config(page_title="Post-Mortem Tissue Analysis", layout="wide")
st.title("🔬 Classifying Post-mortem Tissues for Identifying Contaminants")

# === Sidebar Definition ===
# Defines the controls and parameters available in the sidebar.
st.sidebar.title("⚙️ Controls")

# --- Processing Controls ---
# Controls related to the 'Processing' tab.
st.sidebar.subheader("🔬 Processing")
processing_method = st.sidebar.selectbox(
    "Select Method",
    ("Automatic Watershed", "Interactive Manual Watershed", "AI Model (coming soon)"),
    key="processing_selector"
)
# Conditional sliders shown only if a Watershed method is selected.
min_area = 180 # Default value
blur_kernel_size = 5 # Default value

# !!! --- MODIFICAÇÃO 1: Adicionar controles de Threshold --- !!!
# Estas variáveis precisam ser definidas FORA do 'if' para estarem no escopo
use_manual_thresh = False
manual_thresh_val = 0.5 # Um padrão, não importa se 'use_manual_thresh' for Falso

ws_lib = 'skimage' # Default library for watershed

if "Watershed" in processing_method:
    min_area = st.sidebar.slider("Minimum Component Area (px²)", 50, 5000, 180, 10, key="ws_min_area")
    
    # O Blur só é usado no Manual, mas podemos deixar aqui ou mover para dentro do 'if manual'
    blur_kernel_size = st.sidebar.slider("Gaussian Smoothing (Kernel)", 1, 21, 5, 2, key="ws_blur_ksize", help="Controls blur intensity before watershed. Increase if flooding stops early.")

    # Apenas mostrar controles de threshold para o método Automático
    if processing_method == "Automatic Watershed":
        st.sidebar.markdown("---") # Separator
        st.sidebar.subheader("Thresholding (Auto Watershed)")
        use_manual_thresh = st.sidebar.checkbox(
            "Set Manual Threshold", 
            value=False, 
            key="ws_use_manual_thresh",
            help="If unchecked, the automatic (Otsu) threshold will be used."
        )
        
        if use_manual_thresh:
            manual_thresh_val = st.sidebar.slider(
                "Manual Threshold Value", 
                0.0, 1.0, 0.5, 0.01, 
                key="ws_manual_thresh_val",
                help="0.0 = Black, 1.0 = White. Pixels *darker* than this value are kept."
            )
            
    st.sidebar.markdown("---") # Separador
    ws_lib = st.sidebar.radio(
        "Watershed Algorithm",
        ('skimage', 'higra'),
        key='ws_lib_select',
        index=0, # 'skimage' é o padrão
        help="Escolha o motor do algoritmo. 'skimage' (baseado em gradiente) é rápido. 'higra' (baseado em grafo) pode ser mais preciso para formas complexas."
    )


st.sidebar.markdown("---")

# --- Validation Controls ---
# Controls related to the 'Validation' tab.
st.sidebar.subheader("📊 Validation")
max_side = st.sidebar.slider("Resize Validation Image (px)", 0, 3000, 1024, 50, help="0 = do not resize.", key="val_max_side")
threshold = st.sidebar.slider("Validation Binarization Threshold", 0, 255, 127, 1, help="Converts grayscale predictions to binary.", key="val_threshold")

# === Main Application Tabs ===
# Defines the main tabs for the application interface.
tab_process, tab_validate, tab_results = st.tabs(["🔬 Processing", "📊 Validation", "🖼️ Visual Results"])

# ===========================
# === PROCESSING TAB LOGIC ===
# ===========================
with tab_process:
    st.header("Run Segmentation Algorithms")

    # --- Image Upload ---
    # Widget for uploading the tissue image to be processed.
    uploaded_file = st.file_uploader("Upload a tissue image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="process_uploader")

    # --- Image Loading and Preprocessing ---
    # Executes only if a file has been uploaded.
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            color_image = np.array(pil_image)
            img_width, img_height = pil_image.size
        except Exception as e:
            st.error(f"Error loading or converting image: {e}")
            st.stop()

        # --- Canvas Dimension Calculation ---
        # Calculates the proportional width for the canvas based on a fixed height.
        CANVAS_HEIGHT = 400 # Fixed canvas height
        aspect_ratio = img_width / img_height if img_height != 0 else 1
        canvas_width = int(aspect_ratio * CANVAS_HEIGHT)

        # --- Automatic Watershed Execution ---
        # Logic for running the automatic watershed algorithm.
        if processing_method == "Automatic Watershed":
            st.subheader("Mode: Automatic Watershed")
            current_min_area = min_area # Get value from sidebar slider
            run_auto = st.button("▶️ Run", type="primary", key="run_auto")
            
            if run_auto:
                with st.spinner("Processing..."):
                    
                    threshold_to_pass = None
                    if use_manual_thresh:
                        threshold_to_pass = manual_thresh_val 
                    
                    if threshold_to_pass is not None:
                        st.info(f"Running with MANUAL THRESHOLD: {threshold_to_pass}")
                    else:
                        st.info("Running with AUTOMATIC (Otsu) THRESHOLD...")
                    
                    fig_auto, mask_auto = run_automatic_watershed(
                        color_image, 
                        current_min_area, 
                        global_threshold=threshold_to_pass,
                        watershed_method=ws_lib
                    )
                    st.pyplot(fig_auto) # Display the resulting matplotlib figure

        # --- Manual Watershed Execution ---
        # Logic for running the interactive manual watershed algorithm.
        elif processing_method == "Interactive Manual Watershed":
            st.subheader("Mode: Interactive Manual Watershed")
            st.info("Use the 'Point' tool to mark the center of each region, then click 'Run with Markers'.")

            # --- Canvas State Initialization ---
            # Ensures a placeholder exists in session state for canvas data.
            if 'canvas_data_manual' not in st.session_state:
                st.session_state.canvas_data_manual = None

            # --- Drawable Canvas Widget ---
            # Creates the canvas for user interaction (drawing markers).
            canvas_width_manual = canvas_width
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2, stroke_color="#FF0000",
                background_image=pil_image, # Pass original PIL image
                update_streamlit=True,      # Update on every interaction to save state
                height=CANVAS_HEIGHT,
                width=canvas_width_manual, # Use calculated width
                drawing_mode="point",
                key="canvas"
            )

            # --- Save Canvas State ---
            # Persists the drawing data between Streamlit reruns.
            if canvas_result is not None and canvas_result.json_data is not None:
                st.session_state.canvas_data_manual = canvas_result.json_data

            # --- Run Button and Processing Logic ---
            run_manual = st.button("▶️ Run with Markers", type="primary", key="run_manual")

            if run_manual:
                 # Reads marker data saved in session state.
                if 'canvas_data_manual' in st.session_state and st.session_state.canvas_data_manual is not None:
                    canvas_data = st.session_state.canvas_data_manual
                    if "objects" in canvas_data and canvas_data["objects"]:
                        try:
                            # --- Marker Data Processing ---
                            # Converts canvas JSON data to a pandas DataFrame.
                            markers_df = pd.json_normalize(canvas_data["objects"])
                            # Filters for 'circle' type objects (used by the 'point' tool) and renames columns.
                            point_markers_df = markers_df[markers_df['type'] == 'circle'].rename(columns={'left': 'x', 'top': 'y'})

                            if not point_markers_df.empty:
                                # --- Coordinate Rescaling with Correction ---
                                # Converts canvas coordinates (relative to the displayed, possibly resized image)
                                # back to original image coordinates, including the manual X-offset correction.
                                canvas_height_actual = CANVAS_HEIGHT
                                canvas_width_actual = canvas_width_manual

                                img_aspect_actual = img_width / img_height if img_height != 0 else 1
                                canvas_aspect_actual = canvas_width_actual / canvas_height_actual if canvas_height_actual != 0 else 1

                                # Calculate scale and offsets based on aspect ratios
                                if img_aspect_actual > canvas_aspect_actual: # Image wider than canvas space
                                    scale = canvas_width_actual / img_width if img_width != 0 else 0
                                    offset_x = 0
                                    offset_y = (canvas_height_actual - (img_height * scale)) / 2 if scale != 0 else 0
                                else: # Image taller than or same aspect as canvas space
                                    scale = canvas_height_actual / img_height if img_height != 0 else 0
                                    offset_x = (canvas_width_actual - (img_width * scale)) / 2 if scale != 0 else 0
                                    offset_y = 0

                                x_correction_pixels = 5 # Manual offset correction based on debugging

                                # Apply inverse transformation
                                if scale != 0:
                                    point_markers_df['original_x'] = (((point_markers_df['x'] - offset_x) / scale) + x_correction_pixels).astype(int)
                                    point_markers_df['original_y'] = ((point_markers_df['y'] - offset_y) / scale).astype(int)
                                else:
                                    point_markers_df['original_x'] = 0
                                    point_markers_df['original_y'] = 0

                                # Filter out points that fall outside original image bounds after rescaling
                                point_markers_df_filtered = point_markers_df[
                                    (point_markers_df['original_x'] >= 0) & (point_markers_df['original_x'] < img_width) &
                                    (point_markers_df['original_y'] >= 0) & (point_markers_df['original_y'] < img_height)
                                ].copy()
                                # Prepare DataFrame for the watershed function
                                point_markers_df_filtered = point_markers_df_filtered[['original_x', 'original_y', 'type']].rename(columns={'original_x':'x', 'original_y':'y'})

                                # --- Execute Watershed ---
                                if not point_markers_df_filtered.empty:
                                     current_min_area = min_area # Get value from sidebar
                                     current_blur = blur_kernel_size # Get value from sidebar
                                     with st.spinner("Processing..."):
                                         # Call the imported manual watershed function
                                         fig_manual, mask_manual = run_manual_watershed(color_image, 
                                                                                        current_min_area, 
                                                                                        point_markers_df_filtered, 
                                                                                        current_blur,
                                                                                        watershed_method=ws_lib)
                                         st.pyplot(fig_manual) # Display result
                                else:
                                    st.warning("Markers were outside image bounds after rescaling.")
                            else:
                                st.warning("No 'circle' type markers found in canvas data.") # Corrected type
                        except Exception as e:
                            st.error(f"Error processing canvas data: {e}")
                            st.json(canvas_data if 'canvas_data' in locals() else "No canvas data in state")
                    else:
                        st.warning("No markers drawn in the saved state.")
                else:
                     st.warning("Canvas data not found in session state.")

        # --- Placeholder for AI Model ---
        elif processing_method == "AI Model (coming soon)":
            st.info("Module to load and run Deep Learning models (UNETR, etc.) will be implemented here.")
            st.image(pil_image) # Show the uploaded image as a placeholder
    else:
        st.info("Waiting for an image upload to begin.")


# ============================
# === VALIDATION TAB LOGIC ===
# ============================
with tab_validate:
    st.header("Segmentation Mask Validation")
    st.info("Upload original images, ground-truth masks, and prediction masks to calculate validation metrics. Files will be matched by name.")

    # --- File Uploaders for Validation ---
    col1, col2, col3 = st.columns(3)
    with col1:
        original_files = st.file_uploader("1. Original Images (RGB)", accept_multiple_files=True, key="val_orig", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    with col2:
        gt_files = st.file_uploader("2. Ground-Truth Masks (Binary)", accept_multiple_files=True, key="val_gt", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    with col3:
        pred_files = st.file_uploader("3. Prediction Masks (Grayscale/Binary)", accept_multiple_files=True, key="val_pred", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

    # --- Run Validation Button ---
    run_val_btn = st.button("▶️ Run Validation", type="primary", disabled=not (original_files and gt_files and pred_files), key="run_val")

    # --- Validation Processing Loop ---
    if run_val_btn:
        # Get validation parameters from sidebar
        current_max_side = max_side
        current_threshold = threshold

        # Create dictionaries to easily find files by name
        original_map = {f.name: f for f in original_files}
        gt_map = {f.name: f for f in gt_files}
        pred_map = {f.name: f for f in pred_files}

        # Find common filenames across all three upload groups
        common_names = set(original_map.keys()) & set(gt_map.keys()) & set(pred_map.keys())

        if not common_names:
            st.error("No matching filenames found across the three upload groups.")
        else:
            rows, vis_list = [], [] # Initialize lists for results
            progress_bar = st.progress(0, text=f"Processing {len(common_names)} images...")

            # Iterate through each set of matched files
            for i, name in enumerate(sorted(list(common_names))):
                try:
                    # --- Load Images ---
                    pil_orig = Image.open(io.BytesIO(original_map[name].read())).convert("RGB")
                    pil_gt = Image.open(io.BytesIO(gt_map[name].read())).convert("L") # Grayscale
                    pil_pred = Image.open(io.BytesIO(pred_map[name].read())).convert("L") # Grayscale

                    # Convert PIL to OpenCV format (BGR for color, NumPy array for masks)
                    bgr, gt_mask, pred_mask_gray = pil_to_cv2(pil_orig), np.array(pil_gt), np.array(pil_pred)

                    # --- Preprocess (Resize) ---
                    # Resize original image if max_side is set
                    bgr = maybe_downscale(bgr, current_max_side)
                    h, w = bgr.shape[:2]
                    # Resize masks to match the (potentially resized) original image dimensions
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST) # Use NEAREST for masks
                    pred_mask_gray = cv2.resize(pred_mask_gray, (w, h), interpolation=cv2.INTER_LINEAR) # Linear might be okay for grayscale pred

                    # --- Binarize Prediction ---
                    # Apply threshold to convert grayscale prediction mask to binary
                    _, pred_mask = cv2.threshold(pred_mask_gray, current_threshold, 255, cv2.THRESH_BINARY)

                    # --- Calculate Metrics ---
                    # Use the imported function to compute Dice, IoU, etc.
                    metrics = calculate_metrics(gt_mask, pred_mask)
                    if metrics:
                        metrics["image"] = name # Add filename to metrics dictionary
                        rows.append(metrics)
                    else:
                         st.warning(f"Could not calculate metrics for {name}. Check mask sizes or content.")

                    # --- Create Visual Overlay ---
                    # Generate an image showing TP (Green), FP (Red), FN (Blue)
                    overlay_img = create_overlay(bgr, gt_mask, pred_mask)

                    # Store images for the results tab (convert BGR back to RGB for display)
                    vis_list.append({
                        "name": name, "original": cv2_to_rgb(bgr), "ground_truth": gt_mask,
                        "prediction": pred_mask, "overlay": cv2_to_rgb(overlay_img)
                    })
                except Exception as e:
                    st.error(f"Error processing {name}: {e}")

                # Update progress bar
                progress_bar.progress((i + 1) / len(common_names), text=f"Processing {name}...")

            # --- Save Results to Session State ---
            # Store calculated metrics and visual results for access in the Results tab.
            st.session_state.bench_rows = rows
            st.session_state.vis_images = vis_list
            st.success("✅ Validation complete! Go to the **Visual Results** tab.")


# =================================
# === VISUAL RESULTS TAB LOGIC ===
# =================================
with tab_results:
    st.header("Validation Results")

    # --- Display Metrics Table ---
    # Check if results exist in session state before displaying.
    if 'bench_rows' not in st.session_state or not st.session_state.get('bench_rows'):
        st.info("Run the validation on the '📊 Validation' tab to see results here.")
    else:
        st.subheader("Metrics Table")
        try:
            df_results = pd.DataFrame(st.session_state.bench_rows)
            st.dataframe(df_results, use_container_width=True)

            # --- Export Metrics ---
            # Provides a button to download the metrics as a CSV file.
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Metrics (CSV)", data=csv_bytes, file_name="validation_metrics.csv", mime="text/csv", key="dl_csv_val")

        except Exception as e:
             st.error(f"Error creating or displaying metrics DataFrame: {e}")
             st.write("Raw metrics data:")
             st.json(st.session_state.bench_rows)

        # --- Display Visual Results ---
        # Iterates through the stored visual data and displays images for each file.
        if 'vis_images' in st.session_state and st.session_state.get('vis_images'):
            st.subheader("Visual Results")
            for item in st.session_state.vis_images:
                try:
                    st.markdown(f"#### {item.get('name', 'Unknown Filename')}")
                    # Use columns for side-by-side image comparison
                    cols = st.columns(4)
                    cols[0].image(item.get("original"), caption="Original", use_container_width=True)
                    cols[1].image(item.get("ground_truth"), caption="Ground-Truth", use_container_width=True)
                    cols[2].image(item.get("prediction"), caption="Prediction (Binary)", use_container_width=True)
                    cols[3].image(item.get("overlay"), caption="Overlay (TP:G, FP:R, FN:B)", use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying images for {item.get('name', 'Unknown Filename')}: {e}")
        else:
             st.warning("No visual result images found in session state.")