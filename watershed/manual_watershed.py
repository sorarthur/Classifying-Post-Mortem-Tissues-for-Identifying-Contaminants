# manual_watershed.py (Versão Revertida)

import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk, dilation # Dilation ainda pode ser útil
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi

# Esta versão usa um marcador de fundo explícito e dilata os marcadores FG
def run_manual_watershed(color_image, min_area_threshold, markers_df):
    gray_image = color.rgb2gray(color_image)
    image_size = gray_image.shape

    # === 2. HIGRA SETUP ===
    graph = hg.get_4_adjacency_graph(image_size)
    edge_weights = hg.weight_graph(graph, gray_image, hg.WeightFunction.L1)

    # === 3. PRE-PROCESSING (Foreground Mask - Usando Abertura) ===
    otsu_threshold = threshold_otsu(gray_image)
    binary_mask = gray_image < otsu_threshold
    opened_mask = opening(binary_mask, disk(2)) # Usando abertura novamente

    # === 4. PROCESSAMENTO DOS MARCADORES (COM MARCADOR DE FUNDO EXPLÍCITO) ===

    # a) Marcadores de Foreground (baseados nos cliques)
    peak_markers_fg = np.zeros(image_size, dtype=bool)
    valid_marker_points_count = 0
    if not markers_df.empty:
        for index, row in markers_df.iterrows():
            if row['type'] == 'circle':
                x, y = int(row['x']), int(row['y'])
                # Valida apenas pelos limites da imagem nesta versão
                if 0 <= y < image_size[0] and 0 <= x < image_size[1]:
                     # Não valida contra binary_mask aqui nesta versão revertida
                     peak_markers_fg[y, x] = True
                     valid_marker_points_count += 1

    if valid_marker_points_count == 0:
         print("Erro: Nenhum marcador válido foi criado.")
         fig, _ = plt.subplots()
         plt.text(0.5, 0.5, 'Erro: Nenhum marcador válido.', ha='center', va='center')
         return fig, np.zeros(image_size, dtype=np.int32)

    # b) Dilatação e Rotulagem (Pode fundir marcadores próximos)
    marker_radius = 2
    dilated_markers_fg = dilation(peak_markers_fg, disk(marker_radius))
    labeled_markers_fg, num_regions = ndi.label(dilated_markers_fg)

    # c) Marcador de Fundo Explícito (Baseado na máscara ABERTA)
    background_marker = (opened_mask == 0)

    # d) Combinar Marcadores
    final_markers = np.zeros(image_size, dtype=np.int32)
    final_markers[background_marker] = 1 # Fundo = 1
    final_markers[labeled_markers_fg > 0] = labeled_markers_fg[labeled_markers_fg > 0] + 1 # FG = 2, 3...

    # === 5. WATERSHED SEGMENTATION ===
    print("Executando watershed com marcadores FG e BG...")
    partition = hg.watershed.labelisation_seeded_watershed(graph, edge_weights, final_markers)
    segmentation = partition.to_label_image(image_size) if hasattr(partition, 'to_label_image') else partition
    print("Unique labels após watershed:", np.unique(segmentation))

    # Remove o label do fundo (1)
    segmentation[segmentation == 1] = 0
    print("Unique labels após remover fundo:", np.unique(segmentation))

    # === 6. POST-PROCESSING (HOLE FILLING E FILTRO DE ÁREA) ===
    post_processed_image = np.copy(segmentation)
    component_labels = np.unique(segmentation)
    if len(component_labels)>0 and component_labels[0] == 0: component_labels = component_labels[1:]

    print(f"Preenchendo buracos para labels: {component_labels}")
    for label in component_labels:
        component_mask = (segmentation == label)
        filled_component_mask = binary_fill_holes(component_mask)
        post_processed_image[filled_component_mask] = label

    # Filtro de Área Mínima
    final_filtered_image = np.copy(post_processed_image)
    component_labels_final = np.unique(final_filtered_image)
    if len(component_labels_final)>0 and component_labels_final[0] == 0: component_labels_final = component_labels_final[1:]

    small_area_threshold = min_area_threshold # Usa o valor da interface
    print(f"Filtrando componentes menores que {small_area_threshold} pixels...")
    removed_count = 0
    for label in component_labels_final:
        component_mask = (final_filtered_image == label)
        area = np.sum(component_mask)
        if area < small_area_threshold:
            final_filtered_image[component_mask] = 0
            removed_count += 1
    print(f"Removidos {removed_count} componentes pequenos.")
    print("Unique labels após filtro de área:", np.unique(final_filtered_image))

    output_image_for_display = final_filtered_image

    # === 7. AREA ANALYSIS ===
    labels, areas = np.unique(output_image_for_display, return_counts=True)
    component_areas = {}
    total_tissue_area = 0
    for label, area in zip(labels, areas):
        if label > 0:
            component_areas[label] = area
            total_tissue_area += area

    # === 8. VISUALIZATION (Layout 2x2 original) ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tissue Segmentation (Manual - Reverted)', fontsize=16) # Título atualizado
    ax = axes.ravel()
    ax[0].imshow(color_image)
    ax[0].set_title("1. Original Image")
    ax[0].axis('off')
    ax[1].imshow(final_markers, cmap='nipy_spectral') # Mostra marcadores combinados
    ax[1].set_title("2. Combined Markers (BG=1)")
    ax[1].axis('off')
    ax[2].imshow(output_image_for_display, cmap='nipy_spectral')
    ax[2].set_title("3. Final Filtered Segmentation")
    ax[2].axis('off')
    ax[3].axis('off')
    ax[3].set_title("4. Final Area Analysis")
    # ... (código de plotagem da área inalterado) ...
    if component_areas:
        display_areas = {f"Comp {lbl}": area for lbl, area in component_areas.items()}
        sorted_items = sorted(display_areas.items())
        mid_point = len(sorted_items) // 2 + (len(sorted_items) % 2)
        left_column_items = sorted_items[:mid_point]
        right_column_items = sorted_items[mid_point:]
        left_column_str = "\n".join([f"{name}: {area} px" for name, area in left_column_items])
        right_column_str = "\n".join([f"{name}: {area} px" for name, area in right_column_items])
        ax[3].text(0.05, 0.80, left_column_str, fontsize=10, fontfamily='monospace', verticalalignment='top')
        ax[3].text(0.55, 0.80, right_column_str, fontsize=10, fontfamily='monospace', verticalalignment='top')
        total_area_str = f"\n\nTotal Segmented Area: {total_tissue_area} pixels"
        ax[3].text(0.30, 0.85, total_area_str, fontsize=10, fontweight='bold', verticalalignment='bottom')
    else:
        ax[3].text(0.5, 0.5, "Nenhum componente\nencontrado após filtragem.", ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, output_image_for_display