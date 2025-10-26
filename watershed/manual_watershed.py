# manual_watershed.py (Reintroduzindo Loop Divide and Conquer)

import higra as hg
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, measure
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk, dilation, remove_small_objects
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from scipy import ndimage as ndi
import cv2 # Para blur

def run_manual_watershed(color_image, min_area_threshold, markers_df, blur_ksize=5):
    gray_image = color.rgb2gray(color_image)
    image_size = gray_image.shape

    # === 2. HIGRA SETUP (COM SUAVIZAÇÃO) ===
    ksize = max(1, int(blur_ksize))
    if ksize % 2 == 0: ksize += 1
    print(f"Aplicando Gaussian Blur ({ksize}x{ksize})...")
    blurred_gray = cv2.GaussianBlur(gray_image.astype(np.float32), (ksize, ksize), 0)
    
    print("Criando grafo e pesos (na imagem suavizada)...")
    graph = hg.get_4_adjacency_graph(image_size)
    edge_weights = hg.weight_graph(graph, blurred_gray, hg.WeightFunction.L1)

    # === 3. PRE-PROCESSING: IDENTIFICAR COMPONENTES PRINCIPAIS ===
    # (Similar ao código antigo - Step 3)
    print("Identificando componentes principais...")
    otsu_threshold = threshold_otsu(gray_image) # Usa imagem original para Otsu
    binary_mask = gray_image < otsu_threshold
    # Usa uma abertura talvez menor aqui, só para limpar ruído inicial
    selem_open = disk(2) 
    opened_mask = opening(binary_mask, selem_open) 
    # Filtra por área para obter os componentes principais onde clicaremos
    labeled_mask_initial = measure.label(opened_mask)
    regions_props = measure.regionprops(labeled_mask_initial)
    
    # Usa o min_area_threshold da interface para definir componentes principais
    main_component_mask = np.copy(labeled_mask_initial)
    for prop in regions_props:
        if prop.area < min_area_threshold:
            main_component_mask[main_component_mask == prop.label] = 0
    
    # Rotula os componentes principais finais
    main_components, num_components = measure.label(main_component_mask > 0, return_num=True)
    print(f"Encontrados {num_components} componentes principais para processar.")
    
    if num_components == 0:
        print("Erro: Nenhum componente principal encontrado após filtragem inicial.")
        fig, _ = plt.subplots()
        plt.text(0.5, 0.5, 'Erro: Nenhum componente principal encontrado.', ha='center', va='center')
        return fig, np.zeros(image_size, dtype=np.int32)

    # === 4. PROCESSAMENTO DOS MARCADORES (Recebidos da interface) ===
    # Apenas valida e armazena os marcadores válidos
    valid_markers_coords = []
    if not markers_df.empty:
        print("Validando marcadores recebidos...")
        for index, row in markers_df.iterrows():
            if row['type'] == 'circle':
                x, y = int(row['x']), int(row['y'])
                # Valida pelos limites da imagem
                if 0 <= y < image_size[0] and 0 <= x < image_size[1]:
                    # Verifica se o marcador cai DENTRO de algum componente principal
                    if main_components[y, x] > 0:
                         valid_markers_coords.append({'x': x, 'y': y, 'label': -1}) # Label inicial inválido
                         print(f"  Marker {index} (Coord: {x},{y}) - VÁLIDO (Dentro do Componente {main_components[y, x]})")
                    else:
                         print(f"  Marker {index} (Coord: {x},{y}) - IGNORADO (Fora dos componentes principais)")
                else:
                    print(f"  Marker {index} (Coord: {x},{y}) - IGNORADO (Fora da imagem)")
        print(f"Total de marcadores válidos: {len(valid_markers_coords)}")
    
    if not valid_markers_coords:
         print("Erro: Nenhum marcador válido dentro dos componentes principais.")
         fig, _ = plt.subplots()
         plt.text(0.5, 0.5, 'Erro: Nenhum marcador válido.', ha='center', va='center')
         return fig, np.zeros(image_size, dtype=np.int32)

    # === 5. SEGMENTAÇÃO "DIVIDE AND CONQUER" (Loop do Código Antigo) ===
    final_segmented_image = np.zeros(image_size, dtype=np.int32)
    label_offset = 0
    processed_labels = set() # Para acompanhar quais labels já foram usados

    print(f"Iniciando loop de segmentação para {num_components} componentes...")
    for i in range(1, num_components + 1):
        current_mask = (main_components == i)
        print(f"\n--- Processando Componente Principal {i} ---")

        # a) Filtra os marcadores válidos que pertencem a ESTE componente
        local_markers_coords = []
        for marker in valid_markers_coords:
            # Re-verifica, pois o clique pode estar na borda exata
            if current_mask[marker['y'], marker['x']]: 
                local_markers_coords.append((marker['y'], marker['x'])) 
        
        print(f"  Marcadores encontrados para este componente: {len(local_markers_coords)}")

        # b) Cria a imagem de marcadores LOCAIS (dilatados e rotulados)
        if len(local_markers_coords) > 0:
            local_peak_markers = np.zeros(image_size, dtype=bool)
            for y, x in local_markers_coords:
                local_peak_markers[y, x] = True

            # Dilatação e Rotulagem (sem merge, como na última tentativa)
            labeled_individual_local, num_local_initial = ndi.label(local_peak_markers)
            marker_radius = 2
            local_labeled_markers_fg = np.zeros(image_size, dtype=np.int32)
            for lbl_local in range(1, num_local_initial + 1):
                 single_marker_mask = (labeled_individual_local == lbl_local)
                 dilated_single_marker = dilation(single_marker_mask, disk(marker_radius))
                 # Aplica o label DENTRO DA MÁSCARA DO COMPONENTE ATUAL
                 target_mask = dilated_single_marker & (local_labeled_markers_fg == 0) & current_mask
                 local_labeled_markers_fg[target_mask] = lbl_local
                 # Reforça o centro original
                 local_labeled_markers_fg[single_marker_mask & current_mask] = lbl_local

            # Conta as regiões efetivamente criadas DENTRO da máscara
            num_local_regions = len(np.unique(local_labeled_markers_fg[current_mask])) -1 # -1 para o 0
            if num_local_regions == 0: # Segurança: se a dilatação falhar ou sair da máscara
                 print("  Aviso: Nenhum marcador local efetivo após dilatação/mascaramento. Tratando como região única.")
                 local_labeled_markers_fg[current_mask] = 1 # Trata como 1 região
                 num_local_regions = 1
            print(f"  Número de sementes locais (após dilatação): {num_local_regions}")

        else:
            # Se nenhum marcador foi clicado DENTRO deste componente, trata como uma região única
            print("  Nenhum marcador específico clicado. Tratando como região única.")
            local_labeled_markers_fg = np.zeros(image_size, dtype=np.int32)
            local_labeled_markers_fg[current_mask] = 1 # Label 1 para este componente
            num_local_regions = 1

        # c) Aplica o watershed usando os marcadores LOCAIS
        # Não precisa de marcador de fundo aqui, pois vamos mascarar depois
        print("  Executando watershed local...")
        partition_local = watershed(blurred_gray, markers=local_labeled_markers_fg, mask=current_mask)
        segmentation_local_full = partition_local.to_label_image(image_size) if hasattr(partition_local, 'to_label_image') else partition_local

        # d) Mascara o resultado para o componente atual (Crucial!)
        local_segmentation_masked = np.zeros(image_size, dtype=np.int32)
        local_segmentation_masked[current_mask] = segmentation_local_full[current_mask]

        # e) Adiciona ao resultado final com offset
        # Garante que os novos labels sejam únicos globalmente
        new_labels = np.unique(local_segmentation_masked)
        if len(new_labels) > 0 and new_labels[0] == 0: new_labels = new_labels[1:] # Remove 0
        
        current_offset = label_offset
        # Encontra o próximo offset disponível que não colida com labels já usados
        while any((lbl + current_offset) in processed_labels for lbl in new_labels if (lbl + current_offset) != 0):
             current_offset += num_local_regions # Tenta o próximo bloco de labels

        print(f"  Aplicando offset: {current_offset}. Labels originais locais: {new_labels}")
        
        # Aplica o offset apenas aos pixels > 0
        mask_to_update = local_segmentation_masked > 0
        final_segmented_image[mask_to_update] += (local_segmentation_masked[mask_to_update] + current_offset)
        
        # Atualiza o conjunto de labels usados e o próximo offset base
        for lbl in new_labels:
             processed_labels.add(lbl + current_offset)
        label_offset = max(processed_labels) if processed_labels else 0 
        print(f"  Labels globais após componente {i}: {np.unique(final_segmented_image)}")


    # === 6. POST-PROCESSING (Aplicado à imagem combinada FINAL) ===
    print("\nIniciando pós-processamento final...")
    
    # a) Preenchimento de Buracos
    post_processed_filled = np.copy(final_segmented_image)
    component_labels = np.unique(final_segmented_image)
    if len(component_labels) > 0 and component_labels[0] == 0: component_labels = component_labels[1:]
    print(f"Preenchendo buracos para {len(component_labels)} componentes...")
    for label in component_labels:
        component_mask = (final_segmented_image == label)
        filled_component_mask = binary_fill_holes(component_mask)
        post_processed_filled[filled_component_mask] = label

    # b) Filtro de Área Mínima
    # Usa o min_area_threshold original AQUI para limpar fragmentos do resultado final
    final_filtered_image = remove_small_objects(post_processed_filled, min_size=min_area_threshold)
    final_labels_after_filter = np.unique(final_filtered_image)
    print("Labels finais após filtro de área:", final_labels_after_filter)
    output_image_for_display = final_filtered_image

    # === 7. AREA ANALYSIS (Baseado na imagem FINAL filtrada) ===
    labels, areas = np.unique(output_image_for_display, return_counts=True)
    component_areas = {}
    total_tissue_area = 0
    # ... (código inalterado) ...
    for label, area in zip(labels, areas):
        if label > 0:
            component_areas[label] = area
            total_tissue_area += area

    # === 8. VISUALIZATION (Layout 2x2 para simplicidade) ===
    print("Gerando visualização final...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tissue Segmentation (Manual - Divide & Conquer)', fontsize=16)
    ax = axes.ravel()
    ax[0].imshow(color_image)
    ax[0].set_title("1. Original Image")
    ax[0].axis('off')
    
    # Mostra os componentes principais identificados
    ax[1].imshow(main_components, cmap='nipy_spectral') 
    ax[1].set_title("2. Main Components Identified")
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
        max_items_per_col = 15 
        col_strs = []
        current_col = []
        for i, (name, area) in enumerate(sorted_items):
             current_col.append(f"{name}: {area} px")
             if len(current_col) >= max_items_per_col or i == len(sorted_items) - 1:
                  col_strs.append("\n".join(current_col))
                  current_col = []
        x_pos_step = 0.9 / max(1, len(col_strs))
        for i, col_str in enumerate(col_strs):
             ax[3].text(0.05 + i * x_pos_step, 0.95, col_str, fontsize=9, fontfamily='monospace', verticalalignment='top')
        total_area_str = f"\n\nTotal Segmented Area: {total_tissue_area} pixels"
        ax[3].text(0.5, 0.1, total_area_str, fontsize=10, fontweight='bold', ha='center', verticalalignment='bottom')
    else:
        ax[3].text(0.5, 0.5, "Nenhum componente\nencontrado após filtragem.", ha='center', va='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, output_image_for_display