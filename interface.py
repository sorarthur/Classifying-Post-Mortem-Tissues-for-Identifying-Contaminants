# interface_segmentacao_app.py
# Run: streamlit run interface_segmentacao_app.py

import io
import time
import zipfile
from typing import Dict

import numpy as np
import cv2
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Validação de Segmentação de Contaminantes", layout="wide")

# ======================================================================
# Funções de Métricas de Segmentação
# ======================================================================
def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict:
    """Calcula Dice, IoU, Precisão e Recall para máscaras binárias."""
    if gt_mask.shape != pred_mask.shape:
        st.error(f"Erro: As máscaras têm tamanhos diferentes! GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
        return {}

    gt_mask = gt_mask / 255.0
    pred_mask = pred_mask / 255.0

    intersection = np.sum(gt_mask * pred_mask)
    gt_sum = np.sum(gt_mask)
    pred_sum = np.sum(pred_mask)

    dice = (2.0 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0
    iou = intersection / (gt_sum + pred_sum - intersection) if (gt_sum + pred_sum - intersection) > 0 else 0
    precision = intersection / pred_sum if pred_sum > 0 else 0
    recall = intersection / gt_sum if gt_sum > 0 else 0
    
    return {
        "Dice": round(dice, 4),
        "IoU": round(iou, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }

def create_overlay(bgr_original: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Cria uma sobreposição visual com TP, FP, FN."""
    overlay = bgr_original.copy()
    
    # True Positives (Verde)
    tp_mask = cv2.bitwise_and(gt_mask, pred_mask)
    overlay[tp_mask > 0] = cv2.addWeighted(overlay[tp_mask > 0], 0.5, np.array([0, 255, 0], dtype=np.uint8), 0.5, 0)
    
    # False Positives (Vermelho)
    fp_mask = cv2.subtract(pred_mask, gt_mask)
    overlay[fp_mask > 0] = cv2.addWeighted(overlay[fp_mask > 0], 0.5, np.array([0, 0, 255], dtype=np.uint8), 0.5, 0)

    # False Negatives (Azul)
    fn_mask = cv2.subtract(gt_mask, pred_mask)
    overlay[fn_mask > 0] = cv2.addWeighted(overlay[fn_mask > 0], 0.5, np.array([255, 0, 0], dtype=np.uint8), 0.5, 0)
    
    return overlay

# ======================================================================
# Utils
# ======================================================================
def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def maybe_downscale(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if max_side and m > max_side:
        s = max_side / m
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return bgr

# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.title("⚙️ Configurações de Validação")

st.sidebar.subheader("Tamanho (pré-processamento)")
max_side = st.sidebar.slider("Redimensionar máx. lado (px)", 0, 3000, 1024, 50,
                             help="0 = não redimensiona. Útil para acelerar em imagens muito grandes.")

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros da Predição")
threshold = st.sidebar.slider("Limiar de binarização (threshold)", 0, 255, 127, 1,
                              help="Converte máscaras de predição em escala de cinza para preto e branco.")

# ======================================================================
# Abas: Benchmark e Resultados
# ======================================================================
tab_bench, tab_results = st.tabs(["📊 Validação", "🖼️ Resultados Visuais"])

if "bench_rows" not in st.session_state:
    st.session_state.bench_rows = []
if "vis_images" not in st.session_state:
    st.session_state.vis_images = []

# ----------------------------------------------------------------------
# ABA 1 — VALIDAÇÃO
# ----------------------------------------------------------------------
with tab_bench:
    st.header("Validação de Modelos de Segmentação")
    
    st.info("Faça o upload das imagens originais, das máscaras de ground-truth e das predições do seu modelo. O programa irá pareá-las pelo nome.")

    col1, col2, col3 = st.columns(3)
    with col1:
        original_files = st.file_uploader("1. Imagens Originais (RGB)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], accept_multiple_files=True)
    with col2:
        gt_files = st.file_uploader("2. Máscaras Ground-Truth (Binárias)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], accept_multiple_files=True)
    with col3:
        pred_files = st.file_uploader("3. Máscaras da Predição (Grayscale/Binárias)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], accept_multiple_files=True)

    run_btn = st.button("▶️ Executar Validação", type="primary", use_container_width=True, disabled=not (original_files and gt_files and pred_files))

    if run_btn:
        original_map = {f.name: f for f in original_files}
        gt_map = {f.name: f for f in gt_files}
        pred_map = {f.name: f for f in pred_files}
        
        common_names = set(original_map.keys()) & set(gt_map.keys()) & set(pred_map.keys())
        
        if not common_names:
            st.error("Nenhum arquivo com nome correspondente encontrado entre os três grupos de upload.")
        else:
            st.write(f"Encontrados {len(common_names)} conjuntos de imagens correspondentes.")
        
        rows = []
        vis_list = []
        progress_bar = st.progress(0)

        for i, name in enumerate(sorted(list(common_names))):
            # Carregar imagens
            pil_orig = Image.open(io.BytesIO(original_map[name].read())).convert("RGB")
            pil_gt = Image.open(io.BytesIO(gt_map[name].read())).convert("L")
            pil_pred = Image.open(io.BytesIO(pred_map[name].read())).convert("L")
            
            bgr = pil_to_cv2(pil_orig)
            gt_mask = np.array(pil_gt)
            pred_mask_gray = np.array(pil_pred)
            
            # Pré-processamento
            bgr = maybe_downscale(bgr, max_side)
            h, w = bgr.shape[:2]
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_mask_gray = cv2.resize(pred_mask_gray, (w, h), interpolation=cv2.INTER_LINEAR)

            # Binarizar predição
            _, pred_mask = cv2.threshold(pred_mask_gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Calcular métricas
            metrics = calculate_metrics(gt_mask, pred_mask)
            if metrics:
                metrics["image"] = name
                rows.append(metrics)
            
            # Criar visualizações
            overlay_img = create_overlay(bgr, gt_mask, pred_mask)
            vis_entry = {
                "name": name,
                "original": cv2_to_rgb(bgr),
                "ground_truth": gt_mask,
                "prediction": pred_mask,
                "overlay": cv2_to_rgb(overlay_img)
            }
            vis_list.append(vis_entry)
            progress_bar.progress((i + 1) / len(common_names))

        st.session_state.bench_rows = rows
        st.session_state.vis_images = vis_list
        st.success("✅ Validação executada! Vá para a aba **Resultados Visuais** para ver e baixar.")

# ----------------------------------------------------------------------
# ABA 2 — RESULTADOS
# ----------------------------------------------------------------------
with tab_results:
    st.header("Resultados da Validação")
    if not st.session_state.bench_rows:
        st.info("Rode a validação na primeira aba para ver os resultados aqui.")
    else:
        df = pd.DataFrame(st.session_state.bench_rows)
        st.subheader("Tabela de Métricas")
        st.dataframe(df, use_container_width=True)

        st.subheader("Resultados Visuais")
        for item in st.session_state.vis_images:
            st.markdown(f"#### {item['name']}")
            cols = st.columns(4)
            cols[0].image(item["original"], caption="Original", use_container_width=True)
            cols[1].image(item["ground_truth"], caption="Ground-Truth", use_container_width=True)
            cols[2].image(item["prediction"], caption="Predição Binarizada", use_container_width=True)
            cols[3].image(item["overlay"], caption="Sobreposição (TP: Verde, FP: Vermelho, FN: Azul)", use_container_width=True)

        st.subheader("Exportar Resultados")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar CSV de métricas", data=csv_bytes,
                            file_name="segmentation_metrics.csv", mime="text/csv")