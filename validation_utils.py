# validation_utils.py
import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Dict

def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict:
    """Calcula Dice, IoU, Precisão e Recall para máscaras binárias."""
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
    
    tp_mask = cv2.bitwise_and(gt_mask, pred_mask)
    overlay[tp_mask > 0] = cv2.addWeighted(overlay[tp_mask > 0], 0.5, np.array([0, 255, 0], dtype=np.uint8), 0.5, 0)
    
    fp_mask = cv2.subtract(pred_mask, gt_mask)
    overlay[fp_mask > 0] = cv2.addWeighted(overlay[fp_mask > 0], 0.5, np.array([0, 0, 255], dtype=np.uint8), 0.5, 0)

    fn_mask = cv2.subtract(gt_mask, pred_mask)
    overlay[fn_mask > 0] = cv2.addWeighted(overlay[fn_mask > 0], 0.5, np.array([255, 0, 0], dtype=np.uint8), 0.5, 0)
    
    return overlay

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

def pil_to_data_url(pil_img: Image.Image) -> str:
    """Converte uma imagem PIL para uma URL de dados Base64 para a web."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"