# interface.py
# Run: streamlit run interface.py
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import base64

# Importa as funções dos módulos de processamento
# Assume que automatic_watershed.py e manual_watershed.py estão em uma subpasta 'algoritmos'
# Se estiverem na mesma pasta, remova 'algoritmos.'
try:
    from watershed.automatic_watershed import run_automatic_watershed
    from watershed.manual_watershed import run_manual_watershed
except ModuleNotFoundError:
    st.error("Erro: Certifique-se que os arquivos 'automatic_watershed.py' e 'manual_watershed.py' estão na subpasta 'algoritmos' e que a pasta contém um arquivo '__init__.py'.")
    st.stop()


# Importa as funções de validação (do arquivo validation_utils.py na mesma pasta)
try:
    from validation_utils import calculate_metrics, create_overlay, pil_to_cv2, cv2_to_rgb, maybe_downscale
except ModuleNotFoundError:
    st.error("Erro: Arquivo 'validation_utils.py' não encontrado na mesma pasta que 'interface.py'.")
    st.stop()
except ImportError as e:
    st.error(f"Erro ao importar de 'validation_utils.py': {e}. Verifique as funções dentro do arquivo.")
    st.stop()


# --- FUNÇÃO DE AJUDA PARA CORRIGIR INCOMPATIBILIDADE ---
# (Necessária se usar Streamlit > 1.17.0, mas mantida por segurança)
def pil_to_data_url(pil_img: Image.Image) -> str:
    """Converte uma imagem PIL para uma URL de dados Base64 para a web."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


st.set_page_config(page_title="Análise de Tecidos Post-Mortem", layout="wide")
st.title("🔬 Classifying Post-mortem Tissues for Identifying Contaminants")

# ======================================================================
# SIDEBAR (Simplificada e Incondicional)
# ======================================================================
st.sidebar.title("⚙️ Controles")

# --- Controles de Processamento ---
st.sidebar.subheader("🔬 Processamento")
processing_method = st.sidebar.selectbox(
    "Escolha o Método",
    # Adiciona modo de depuração
    ("Watershed Automático", "Watershed Manual Interativo", "Debug Coordenadas", "Modelo de IA (em breve)"),
    key="processing_selector"
)
# Mostra o slider de área mínima apenas para os métodos Watershed
min_area = 180 # Default value
if "Watershed" in processing_method:
    min_area = st.sidebar.slider("Área Mínima (px²)", 50, 5000, 180, 10, key="ws_min_area")

st.sidebar.markdown("---")

# --- Controles de Validação ---
st.sidebar.subheader("📊 Validação")
max_side = st.sidebar.slider("Redimensionar Imagem Validação (px)", 0, 3000, 1024, 50, help="0 = não redimensiona.", key="val_max_side")
threshold = st.sidebar.slider("Limiar Binarização Validação", 0, 255, 127, 1, help="Converte predições em grayscale para binário.", key="val_threshold")


# ======================================================================
# ABAS
# ======================================================================
tab_process, tab_validate, tab_results = st.tabs(["🔬 Processamento", "📊 Validação", "🖼️ Resultados Visuais"])

with tab_process:
    st.header("Execução de Algoritmos / Depuração") # Título atualizado

    uploaded_file = st.file_uploader("Faça o upload de uma imagem de tecido", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="process_uploader") # Adicionado Tiff

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            color_image = np.array(pil_image)
            img_width, img_height = pil_image.size
        except Exception as e:
            st.error(f"Erro ao carregar ou converter a imagem: {e}")
            st.stop()


        # --- Lógica movida para fora dos ifs específicos de método ---
        CANVAS_HEIGHT = 600 # Altura fixa do canvas
        # Calcula a largura proporcional do canvas para manter o aspect ratio da imagem
        aspect_ratio = img_width / img_height if img_height != 0 else 1
        canvas_width = int(aspect_ratio * CANVAS_HEIGHT)

        # --- NOVO MODO DE DEPURAÇÃO ---
        if processing_method == "Debug Coordenadas":
            st.subheader("Modo: Depuração de Coordenadas")
            st.info("Clique em pontos identificáveis na imagem. As coordenadas do Canvas e as calculadas para a imagem original serão exibidas abaixo.")

            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.3)", # Cor verde para depuração
                stroke_width=3,
                stroke_color="#00FF00",
                background_image=pil_image, # Passa imagem PIL
                update_streamlit=True,      # ATUALIZA A CADA CLIQUE
                height=CANVAS_HEIGHT,
                width=canvas_width,        # Usa largura calculada
                drawing_mode="point",
                key="debug_canvas"
            )

            # Mostra os dados crus do canvas sempre que atualiza
            if canvas_result is not None and canvas_result.json_data is not None and "objects" in canvas_result.json_data and canvas_result.json_data["objects"]:
                st.write("--- Dados do Canvas (Tempo Real) ---")
                st.json(canvas_result.json_data["objects"][-1]) # Mostra apenas o último ponto

                try:
                    markers_df = pd.json_normalize(canvas_result.json_data["objects"])
                    point_markers_df = markers_df[markers_df['type'] == 'circle'].rename(columns={'left': 'x', 'top': 'y'}).tail(1) # Pega só o último

                    if not point_markers_df.empty:
                        # --- REESCALONAMENTO COM CORREÇÃO APLICADA (DEBUG) ---
                        canvas_height_actual = CANVAS_HEIGHT
                        canvas_width_actual = canvas_width # Usa a largura calculada

                        img_aspect_actual = img_width / img_height if img_height != 0 else 1
                        canvas_aspect_actual = canvas_width_actual / canvas_height_actual if canvas_height_actual != 0 else 1

                        if img_aspect_actual > canvas_aspect_actual:
                            scale = canvas_width_actual / img_width if img_width != 0 else 0
                            offset_x = 0
                            offset_y = (canvas_height_actual - (img_height * scale)) / 2 if scale != 0 else 0
                        else:
                            scale = canvas_height_actual / img_height if img_height != 0 else 0
                            offset_x = (canvas_width_actual - (img_width * scale)) / 2 if scale != 0 else 0
                            offset_y = 0

                        # Define o fator de correção
                        x_correction_pixels = 5

                        # Calcula as coordenadas originais estimadas com correção
                        if scale != 0: # Evita divisão por zero
                            point_markers_df['original_x'] = (((point_markers_df['x'] - offset_x) / scale) + x_correction_pixels).astype(int) # Correção aplicada
                            point_markers_df['original_y'] = ((point_markers_df['y'] - offset_y) / scale).astype(int)
                        else:
                            point_markers_df['original_x'] = 0
                            point_markers_df['original_y'] = 0
                        # --- FIM DO REESCALONAMENTO ---

                        st.write("--- Comparação de Coordenadas (Último Ponto Corrigido) ---")
                        st.write(f"Scale: {scale:.4f}, OffsetX: {offset_x:.2f}, OffsetY: {offset_y:.2f}, X_Correction: +{x_correction_pixels}")
                        st.dataframe(point_markers_df[['x', 'y', 'original_x', 'original_y']])

                except Exception as e:
                    st.error(f"Erro ao processar coordenadas: {e}")
            else:
                st.write("Aguardando clique...")

        # --- FIM DO NOVO MODO ---

        elif processing_method == "Watershed Automático":
            st.subheader("Modo: Watershed Automático")
            # Garante que 'min_area' esteja definido
            current_min_area = min_area if 'min_area' in locals() or 'min_area' in globals() else 180 
            run_auto = st.button("▶️ Executar", type="primary", key="run_auto")
            if run_auto:
                with st.spinner("Processando..."):
                    fig_auto, mask_auto = run_automatic_watershed(color_image, current_min_area)
                    st.pyplot(fig_auto)

        elif processing_method == "Watershed Manual Interativo":
            st.subheader("Modo: Watershed Manual Interativo")
            st.info("Use a ferramenta 'Ponto' para marcar o centro de cada região. Depois, clique em executar.")

            if 'canvas_data_manual' not in st.session_state:
                st.session_state.canvas_data_manual = None

            canvas_width_manual = canvas_width

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2, stroke_color="#FF0000",
                background_image=pil_image, # Passa imagem PIL
                update_streamlit=True,      # Mantém True para salvar no state
                height=CANVAS_HEIGHT,
                width=canvas_width_manual, # Usa largura calculada
                drawing_mode="point",
                key="canvas"
            )

            # Salva no state
            if canvas_result is not None and canvas_result.json_data is not None:
                st.session_state.canvas_data_manual = canvas_result.json_data

            run_manual = st.button("▶️ Executar com Marcadores", type="primary", key="run_manual") # use_container_width removido

            if run_manual:
                 # Lê do state e aplica a mesma lógica de reescalonamento da depuração
                if 'canvas_data_manual' in st.session_state and st.session_state.canvas_data_manual is not None:
                    canvas_data = st.session_state.canvas_data_manual
                    if "objects" in canvas_data and canvas_data["objects"]:
                        try:
                            markers_df = pd.json_normalize(canvas_data["objects"])
                            point_markers_df = markers_df[markers_df['type'] == 'circle'].rename(columns={'left': 'x', 'top': 'y'}) # Usa 'circle'

                            if not point_markers_df.empty:
                                # --- REESCALONAMENTO COM CORREÇÃO APLICADA (MANUAL) ---
                                canvas_height_actual = CANVAS_HEIGHT
                                canvas_width_actual = canvas_width_manual # Usa a largura do canvas manual

                                img_aspect_actual = img_width / img_height if img_height != 0 else 1
                                canvas_aspect_actual = canvas_width_actual / canvas_height_actual if canvas_height_actual != 0 else 1

                                if img_aspect_actual > canvas_aspect_actual:
                                    scale = canvas_width_actual / img_width if img_width != 0 else 0
                                    offset_x = 0
                                    offset_y = (canvas_height_actual - (img_height * scale)) / 2 if scale != 0 else 0
                                else:
                                    scale = canvas_height_actual / img_height if img_height != 0 else 0
                                    offset_x = (canvas_width_actual - (img_width * scale)) / 2 if scale != 0 else 0
                                    offset_y = 0

                                # Define o fator de correção
                                x_correction_pixels = 5

                                if scale != 0: # Evita divisão por zero
                                    point_markers_df['original_x'] = (((point_markers_df['x'] - offset_x) / scale) + x_correction_pixels).astype(int) # Correção aplicada
                                    point_markers_df['original_y'] = ((point_markers_df['y'] - offset_y) / scale).astype(int)
                                else:
                                    point_markers_df['original_x'] = 0
                                    point_markers_df['original_y'] = 0

                                point_markers_df_filtered = point_markers_df[
                                    (point_markers_df['original_x'] >= 0) & (point_markers_df['original_x'] < img_width) &
                                    (point_markers_df['original_y'] >= 0) & (point_markers_df['original_y'] < img_height)
                                ].copy()
                                point_markers_df_filtered = point_markers_df_filtered[['original_x', 'original_y', 'type']].rename(columns={'original_x':'x', 'original_y':'y'})
                                # --- FIM DO REESCALONAMENTO ---

                                if not point_markers_df_filtered.empty:
                                     # Garante que 'min_area' esteja definido
                                     current_min_area = min_area if 'min_area' in locals() or 'min_area' in globals() else 180
                                     with st.spinner("Processando..."):
                                         fig_manual, mask_manual = run_manual_watershed(color_image, current_min_area, point_markers_df_filtered)
                                         st.pyplot(fig_manual)
                                else:
                                    st.warning("Marcadores fora da imagem após reescalonamento.")
                            else:
                                st.warning("Nenhum marcador 'círculo' encontrado.")
                        except Exception as e:
                            st.error(f"Erro ao processar dados do canvas: {e}")
                            st.json(canvas_data if 'canvas_data' in locals() else "Nenhum dado de canvas no estado")
                    else:
                        st.warning("Nenhum marcador desenhado no estado salvo.")
                else:
                     st.warning("Dados do canvas não encontrados no estado da sessão.")

        elif processing_method == "Modelo de IA (em breve)":
            st.info("Módulo para carregar e executar modelos de Deep Learning (UNETR, etc.) será implementado aqui.")
            st.image(pil_image)
    else:
        st.info("Aguardando o upload de uma imagem para começar.")


# ======================================================================
# ABA 2: VALIDAÇÃO
# ======================================================================
with tab_validate:
    st.header("Validação de Máscaras de Segmentação")
    st.info("Faça o upload das imagens originais, das máscaras de ground-truth e das predições para calcular as métricas de validação.")

    col1, col2, col3 = st.columns(3)
    with col1:
        original_files = st.file_uploader("1. Imagens Originais (RGB)", accept_multiple_files=True, key="val_orig", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    with col2:
        gt_files = st.file_uploader("2. Máscaras Ground-Truth (Binárias)", accept_multiple_files=True, key="val_gt", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    with col3:
        pred_files = st.file_uploader("3. Máscaras da Predição", accept_multiple_files=True, key="val_pred", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

    run_val_btn = st.button("▶️ Executar Validação", type="primary", disabled=not (original_files and gt_files and pred_files), key="run_val") # use_container_width removido

    if run_val_btn:
        # Verifica se os parâmetros da sidebar de validação existem
        current_max_side = max_side if 'max_side' in locals() or 'max_side' in globals() else 1024 # Valor padrão
        current_threshold = threshold if 'threshold' in locals() or 'threshold' in globals() else 127 # Valor padrão

        original_map = {f.name: f for f in original_files}
        gt_map = {f.name: f for f in gt_files}
        pred_map = {f.name: f for f in pred_files}

        common_names = set(original_map.keys()) & set(gt_map.keys()) & set(pred_map.keys())

        if not common_names:
            st.error("Nenhum arquivo com nome correspondente encontrado entre os três grupos de upload.")
        else:
            rows, vis_list = [], []
            progress_bar = st.progress(0, text=f"Processando {len(common_names)} imagens...")

            for i, name in enumerate(sorted(list(common_names))):
                try:
                    pil_orig = Image.open(io.BytesIO(original_map[name].read())).convert("RGB")
                    pil_gt = Image.open(io.BytesIO(gt_map[name].read())).convert("L")
                    pil_pred = Image.open(io.BytesIO(pred_map[name].read())).convert("L")

                    bgr, gt_mask, pred_mask_gray = pil_to_cv2(pil_orig), np.array(pil_gt), np.array(pil_pred)

                    bgr = maybe_downscale(bgr, current_max_side) # Usa current_max_side
                    h, w = bgr.shape[:2]
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_mask_gray = cv2.resize(pred_mask_gray, (w, h), interpolation=cv2.INTER_LINEAR)

                    _, pred_mask = cv2.threshold(pred_mask_gray, current_threshold, 255, cv2.THRESH_BINARY) # Usa current_threshold

                    metrics = calculate_metrics(gt_mask, pred_mask)
                    if metrics:
                        metrics["image"] = name
                        rows.append(metrics)
                    else:
                         st.warning(f"Não foi possível calcular métricas para {name}. Verifique os tamanhos das máscaras.")

                    overlay_img = create_overlay(bgr, gt_mask, pred_mask)
                    vis_list.append({
                        "name": name, "original": cv2_to_rgb(bgr), "ground_truth": gt_mask,
                        "prediction": pred_mask, "overlay": cv2_to_rgb(overlay_img)
                    })
                except Exception as e:
                    st.error(f"Erro ao processar {name}: {e}")

                progress_bar.progress((i + 1) / len(common_names), text=f"Processando {name}...")

            # Salva no estado da sessão APENAS após o loop terminar com sucesso
            st.session_state.bench_rows = rows
            st.session_state.vis_images = vis_list
            st.success("✅ Validação executada! Vá para a aba **Resultados Visuais**.")


# ======================================================================
# ABA 3: RESULTADOS VISUAIS (da Validação)
# ======================================================================
with tab_results:
    st.header("Resultados da Validação")
    # Verifica se as chaves existem E se as listas não estão vazias
    if 'bench_rows' not in st.session_state or not st.session_state.get('bench_rows'):
        st.info("Rode a validação na aba '📊 Validação' para ver os resultados aqui.")
    else:
        st.subheader("Tabela de Métricas")
        try:
            df_results = pd.DataFrame(st.session_state.bench_rows)
            st.dataframe(df_results, use_container_width=True)

            # Exportar CSV
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar Métricas (CSV)", data=csv_bytes, file_name="validation_metrics.csv", mime="text/csv", key="dl_csv_val")

        except Exception as e:
             st.error(f"Erro ao criar ou exibir DataFrame de métricas: {e}")
             st.write("Dados brutos das métricas:")
             st.json(st.session_state.bench_rows)


        if 'vis_images' in st.session_state and st.session_state.get('vis_images'):
            st.subheader("Resultados Visuais")
            for item in st.session_state.vis_images:
                try:
                    st.markdown(f"#### {item.get('name', 'Nome Desconhecido')}")
                    cols = st.columns(4)
                    cols[0].image(item.get("original"), caption="Original", use_container_width=True)
                    cols[1].image(item.get("ground_truth"), caption="Ground-Truth", use_container_width=True)
                    cols[2].image(item.get("prediction"), caption="Predição Binarizada", use_container_width=True)
                    cols[3].image(item.get("overlay"), caption="Sobreposição (TP:Verde, FP:Vermelho, FN:Azul)", use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao exibir imagens para {item.get('name', 'Nome Desconhecido')}: {e}")
        else:
             st.warning("Nenhuma imagem de resultado visual encontrada no estado da sessão.")