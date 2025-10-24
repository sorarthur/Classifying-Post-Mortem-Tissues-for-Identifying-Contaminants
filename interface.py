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
from watershed.automatic_watershed import run_automatic_watershed
from watershed.manual_watershed import run_manual_watershed

# Importa as funções de validação
from validation_utils import calculate_metrics, create_overlay, pil_to_cv2, cv2_to_rgb, maybe_downscale

# --- FUNÇÃO DE AJUDA PARA CORRIGIR INCOMPATIBILIDADE ---
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

# --- Controles de Processamento (Sempre visíveis) ---
st.sidebar.subheader("🔬 Processamento")
processing_method = st.sidebar.selectbox(
    "Escolha o Método",
    ("Watershed Automático", "Watershed Manual Interativo", "Modelo de IA (em breve)"),
    key="processing_selector"
)
if "Watershed" in processing_method:
    min_area = st.sidebar.slider("Área Mínima (px²)", 50, 5000, 180, 10, key="ws_min_area")

st.sidebar.markdown("---")

# --- Controles de Validação (Sempre visíveis) ---
st.sidebar.subheader("📊 Validação")
max_side = st.sidebar.slider("Redimensionar Imagem (px)", 0, 3000, 1024, 50, help="0 = não redimensiona.", key="val_max_side")
threshold = st.sidebar.slider("Limiar de Binarização", 0, 255, 127, 1, help="Converte predições em grayscale para binário.", key="val_threshold")


# ======================================================================
# ABAS
# ======================================================================
tab_process, tab_validate, tab_results = st.tabs(["🔬 Processamento", "📊 Validação", "🖼️ Resultados Visuais"])


with tab_process:
    st.header("Execução de Algoritmos de Segmentação")
    
    uploaded_file = st.file_uploader("Faça o upload de uma imagem de tecido", type=["png", "jpg", "jpeg", "tif"], key="process_uploader")

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        color_image = np.array(pil_image)

        # Agora lê o valor do selectbox diretamente
        if processing_method == "Watershed Automático":
            st.subheader("Modo: Watershed Automático")
            run_auto = st.button("▶️ Executar", type="primary", key="run_auto")
            if run_auto:
                with st.spinner("Processando..."):
                    fig_auto, mask_auto = run_automatic_watershed(color_image, min_area)
                    st.pyplot(fig_auto)
        
        elif processing_method == "Watershed Manual Interativo":
            st.subheader("Modo: Watershed Manual Interativo")
            st.info("Use a ferramenta 'Ponto' para marcar o centro de cada região. Depois, clique em executar.")

            # --- INÍCIO DAS ALTERAÇÕES ---

            # 1. Inicializa uma variável no session_state para guardar os dados do canvas
            if 'canvas_data_manual' not in st.session_state:
                st.session_state.canvas_data_manual = None

            CANVAS_HEIGHT = 400
            img_width, img_height = pil_image.size
            aspect_ratio = img_width / img_height
            new_width = int(aspect_ratio * CANVAS_HEIGHT)
            # Não precisamos redimensionar aqui, passamos a imagem original
            # resized_pil_image = pil_image.resize((new_width, CANVAS_HEIGHT))

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=pil_image, # Passa a imagem PIL original
                update_streamlit=True,       # <--- VOLTA PARA TRUE
                height=CANVAS_HEIGHT,
                # width=new_width,
                drawing_mode="point",
                key="canvas" # Mantém a key
            )

            # 2. Sempre que o canvas atualizar (update_streamlit=True),
            #    salva os dados JSON na nossa variável de estado
            if canvas_result is not None and canvas_result.json_data is not None:
                st.session_state.canvas_data_manual = canvas_result.json_data

            # --- FIM DAS ALTERAÇÕES ---

            # --- Início da seção do botão ---
            # --- Início da seção do botão ---
            # --- Início da seção do botão ---
            run_manual = st.button("▶️ Executar com Marcadores", type="primary", key="run_manual")

            if run_manual:
                # --- DEPURAÇÃO FINAL ---
                st.write("--- DEBUG: Checking saved state on button click ---")
                if 'canvas_data_manual' in st.session_state:
                    st.write("`canvas_data_manual` exists in session state:")
                    st.json(st.session_state.canvas_data_manual) # Mostra o que foi salvo
                else:
                    st.write("`canvas_data_manual` NOT FOUND in session state.")
                st.write("--- END DEBUG ---")
                # --- FIM DEPURAÇÃO ---

                # Lê os dados da nossa variável de estado
                if 'canvas_data_manual' in st.session_state and st.session_state.canvas_data_manual is not None:
                    canvas_data = st.session_state.canvas_data_manual

                    if "objects" in canvas_data and canvas_data["objects"]:
                        try:
                            markers_df = pd.json_normalize(canvas_data["objects"])

                            # Depuração anterior (pode remover se quiser)
                            # st.write("--- DEBUG: Raw Markers DataFrame (Before Filtering) ---")
                            # st.dataframe(markers_df)
                            # st.write("--- END DEBUG ---")

                            if 'type' in markers_df.columns:
                                point_markers_df = markers_df[markers_df['type'] == 'circle'].rename(columns={'left': 'x', 'top': 'y'}) # Usa 'circle'
                            else:
                                st.error("Erro: Coluna 'type' não encontrada.")
                                st.json(canvas_data["objects"])
                                point_markers_df = pd.DataFrame()

                            if not point_markers_df.empty:

                                # --- REESCALONAMENTO PRECISO ---
                                img_width, img_height = pil_image.size # Tamanho original
                                canvas_height = CANVAS_HEIGHT
                                # Recalcula a largura do canvas baseada na proporção da imagem original
                                canvas_width = int((img_width / img_height) * canvas_height) 

                                # Calcula as proporções
                                img_aspect = img_width / img_height
                                canvas_aspect = canvas_width / canvas_height # Proporção do espaço onde a imagem é desenhada

                                # Determina a escala real e os offsets
                                if img_aspect > canvas_aspect:
                                    # Imagem mais larga que o espaço do canvas -> ajustada pela largura
                                    scale = canvas_width / img_width
                                    final_render_height = img_height * scale
                                    offset_x = 0
                                    offset_y = (canvas_height - final_render_height) / 2
                                else:
                                    # Imagem mais alta ou proporção igual -> ajustada pela altura
                                    scale = canvas_height / img_height
                                    final_render_width = img_width * scale
                                    offset_x = (canvas_width - final_render_width) / 2
                                    offset_y = 0

                                # --- DEPURAÇÃO (Manter para verificar) ---
                                st.write("--- DEBUG: Rescaling V3 ---")
                                st.write(f"Image WxH: {img_width}x{img_height} (Aspect: {img_aspect:.2f})")
                                st.write(f"Canvas WxH (calculated): {canvas_width}x{canvas_height} (Aspect: {canvas_aspect:.2f})")
                                st.write(f"Scale applied: {scale:.4f}")
                                st.write(f"Offsets (X, Y): {offset_x:.2f}, {offset_y:.2f}")
                                st.write("Canvas Coords (Raw):")
                                st.dataframe(point_markers_df[['x', 'y']].head())
                                # --- FIM DEPURAÇÃO ---

                                # Aplica a transformação inversa:
                                # 1. Subtrai o offset do canvas
                                # 2. Divide pela escala para obter coords da imagem original
                                point_markers_df['original_x'] = ((point_markers_df['x'] - offset_x) / scale).astype(int)
                                point_markers_df['original_y'] = ((point_markers_df['y'] - offset_y) / scale).astype(int)

                                # --- DEPURAÇÃO (Manter para verificar) ---
                                st.write("Rescaled Coords (Original Image):")
                                st.dataframe(point_markers_df[['original_x', 'original_y']].head().rename(columns={'original_x':'x', 'original_y':'y'}))
                                # --- FIM DEPURAÇÃO ---

                                # Filtra pontos fora da imagem original
                                point_markers_df_filtered = point_markers_df[
                                    (point_markers_df['original_x'] >= 0) & (point_markers_df['original_x'] < img_width) &
                                    (point_markers_df['original_y'] >= 0) & (point_markers_df['original_y'] < img_height)
                                ].copy()

                                # Renomeia as colunas para o formato esperado por manual_watershed.py ('x', 'y')
                                point_markers_df_filtered = point_markers_df_filtered[['original_x', 'original_y', 'type']].rename(columns={'original_x':'x', 'original_y':'y'})


                                # --- DEPURAÇÃO (Manter para verificar) ---
                                st.write(f"Markers remaining after filtering: {len(point_markers_df_filtered)} / {len(point_markers_df)}")
                                st.write("--- END DEBUG ---")
                                # --- FIM DEPURAÇÃO ---

                                if not point_markers_df_filtered.empty:
                                    with st.spinner("Processando..."):
                                        # Passa os marcadores filtrados e reescalados corretamente
                                        fig_manual, mask_manual = run_manual_watershed(color_image, min_area, point_markers_df_filtered)
                                        st.pyplot(fig_manual)
                                else:
                                    st.warning("Nenhum marcador válido encontrado dentro dos limites da imagem após o reescalonamento.")

                            else:
                                st.warning("Nenhum marcador do tipo 'círculo' foi encontrado nos dados do canvas.")

                        except Exception as e:
                            st.error(f"Erro ao processar os dados do canvas: {e}")
                            st.write("Dados salvos no session_state:")
                            st.json(st.session_state.canvas_data_manual)

                    else: # A chave 'objects' não existe ou está vazia
                        st.warning("Nenhum marcador foi desenhado no canvas.") # <<< A MENSAGEM QUE VOCÊ ESTÁ VENDO

                else: # A variável 'canvas_data_manual' não existe ou é None
                     st.warning("Nenhum dado de desenho recebido do canvas no estado da sessão.")
            # --- Fim da lógica do botão ---

        # ... (resto do código) ...
            # --- Fim da lógica do botão ---

        elif processing_method == "Modelo de IA (em breve)":
             # ... (código inalterado) ...
            st.info("Módulo para carregar e executar modelos de Deep Learning (UNETR, etc.) será implementado aqui.")
            st.image(pil_image)


with tab_validate:
    st.header("Validação de Máscaras de Segmentação")
    st.info("Faça o upload das imagens originais, das máscaras de ground-truth e das predições para calcular as métricas de validação.")

    col1, col2, col3 = st.columns(3)
    with col1:
        original_files = st.file_uploader("1. Imagens Originais (RGB)", accept_multiple_files=True, key="val_orig")
    with col2:
        gt_files = st.file_uploader("2. Máscaras Ground-Truth (Binárias)", accept_multiple_files=True, key="val_gt")
    with col3:
        pred_files = st.file_uploader("3. Máscaras da Predição", accept_multiple_files=True, key="val_pred")

    run_val_btn = st.button("▶️ Executar Validação", type="primary", disabled=not (original_files and gt_files and pred_files), key="run_val")

    if run_val_btn:
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
                pil_orig = Image.open(io.BytesIO(original_map[name].read())).convert("RGB")
                pil_gt = Image.open(io.BytesIO(gt_map[name].read())).convert("L")
                pil_pred = Image.open(io.BytesIO(pred_map[name].read())).convert("L")
                
                bgr, gt_mask, pred_mask_gray = pil_to_cv2(pil_orig), np.array(pil_gt), np.array(pil_pred)
                
                bgr = maybe_downscale(bgr, max_side)
                h, w = bgr.shape[:2]
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_mask_gray = cv2.resize(pred_mask_gray, (w, h), interpolation=cv2.INTER_LINEAR)

                _, pred_mask = cv2.threshold(pred_mask_gray, threshold, 255, cv2.THRESH_BINARY)
                
                metrics = calculate_metrics(gt_mask, pred_mask)
                if metrics:
                    metrics["image"] = name
                    rows.append(metrics)
                
                overlay_img = create_overlay(bgr, gt_mask, pred_mask)
                vis_list.append({
                    "name": name, "original": cv2_to_rgb(bgr), "ground_truth": gt_mask,
                    "prediction": pred_mask, "overlay": cv2_to_rgb(overlay_img)
                })
                progress_bar.progress((i + 1) / len(common_names), text=f"Processando {name}...")

            st.session_state.bench_rows = rows
            st.session_state.vis_images = vis_list
            st.success("✅ Validação executada! Vá para a aba **Resultados Visuais**.")


with tab_results:
    st.header("Resultados da Validação")
    if 'bench_rows' not in st.session_state or not st.session_state.bench_rows:
        st.info("Rode a validação na aba '📊 Validação' para ver os resultados aqui.")
    else:
        st.subheader("Tabela de Métricas")
        st.dataframe(pd.DataFrame(st.session_state.bench_rows), use_container_width=True)

        st.subheader("Resultados Visuais")
        for item in st.session_state.vis_images:
            st.markdown(f"#### {item['name']}")
            cols = st.columns(4)
            cols[0].image(item["original"], caption="Original", use_container_width=True)
            cols[1].image(item["ground_truth"], caption="Ground-Truth", use_container_width=True)
            cols[2].image(item["prediction"], caption="Predição Binarizada", use_container_width=True)
            cols[3].image(item["overlay"], caption="Sobreposição (TP:Verde, FP:Vermelho, FN:Azul)", use_container_width=True)

        st.subheader("Exportar")
        csv_bytes = pd.DataFrame(st.session_state.bench_rows).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar CSV", data=csv_bytes, file_name="validation_metrics.csv", mime="text/csv")