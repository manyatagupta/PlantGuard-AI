"""
PlantGuard AI - Computer Vision Leaf Damage & Lesion Surface Area Analyzer
Calculates real biological disease severity by segmenting leaf canopy from background
and measuring percentage of infected / necrotic leaf tissue (% Damaged Area).
"""

import cv2
import numpy as np
from PIL import Image

def analyze_leaf_damage(image_input):
    """
    Analyzes an image to segment leaf tissue, detect necrotic lesions,
    and compute quantitative phytopathological metrics.
    
    Args:
        image_input: PIL Image or RGB numpy array.
        
    Returns:
        dict: Containing damage percentage, severity grade, lesion count,
              overlay image (PIL Image), and diagnostic recommendations.
    """
    if isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    else:
        img_rgb = np.array(image_input).copy()
        
    h, w = img_rgb.shape[:2]
    # Resize temporarily for consistent processing if extremely large
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        proc_img = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        proc_img = img_rgb.copy()
        
    # Convert color spaces
    hsv = cv2.cvtColor(proc_img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(proc_img, cv2.COLOR_RGB2LAB)
    
    # 1. Segment the leaf from the background
    # Most leaves have moderate-to-high saturation or distinct hue, not grey/white background
    r = proc_img[:, :, 0].astype(np.float32)
    g = proc_img[:, :, 1].astype(np.float32)
    b = proc_img[:, :, 2].astype(np.float32)
    
    # Vegetation Index: Excess Green (ExG) = 2*G - R - B
    exg = 2.0 * g - r - b
    
    # Background removal: combine ExG, saturation, and lightness
    s_channel = hsv[:, :, 1]
    l_channel = lab[:, :, 0]
    
    # Leaf mask: not near pure white (background paper/surface) and not near pure black
    not_white_bg = ~((r > 220) & (g > 220) & (b > 220))
    not_black_bg = ~((r < 25) & (g < 25) & (b < 25))
    has_color = (s_channel > 30) | (exg > 10)
    
    leaf_mask_raw = (not_white_bg & not_black_bg & has_color).astype(np.uint8) * 255
    
    # Morphological clean up to get solid leaf canopy
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask_raw, cv2.MORPH_CLOSE, kernel_close)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Find largest contour(s) belonging to the leaf
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Keep contours that are at least 3% of the image area
        min_area = (proc_img.shape[0] * proc_img.shape[1]) * 0.03
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if valid_contours:
            leaf_mask = np.zeros_like(leaf_mask)
            cv2.drawContours(leaf_mask, valid_contours, -1, 255, -1)
            # Fill holes inside the leaf
            cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_close, dst=leaf_mask)
        else:
            # If all are small, take the single largest
            largest = max(contours, key=cv2.contourArea)
            leaf_mask = np.zeros_like(leaf_mask)
            cv2.drawContours(leaf_mask, [largest], -1, 255, -1)
    
    total_leaf_pixels = int(np.count_nonzero(leaf_mask))
    if total_leaf_pixels == 0:
        total_leaf_pixels = proc_img.shape[0] * proc_img.shape[1]
        leaf_mask = np.ones((proc_img.shape[0], proc_img.shape[1]), dtype=np.uint8) * 255

    # 2. Detect Lesions / Necrotic / Chlorotic / Diseased Patches
    # Healthy tissue: High ExG, hue between 35 and 85 (greenish), healthy green-yellow
    # Diseased tissue:
    #   a. Brown/Black necrotic spots (low lightness, low greenness, reddish-brown hues)
    #   b. Chlorotic halo / yellowing (hue between 15 and 35 with high redness or yellow)
    #   c. Powdery / white mildew or grey mold
    h_channel = hsv[:, :, 0]
    
    # Brown / Dark necrotic lesion criteria:
    is_necrotic = (l_channel < 110) & ((exg < 0) | (r > g * 0.95)) & (leaf_mask > 0)
    
    # Rust / Tan / Orange-brown lesion criteria:
    is_rust = ((h_channel < 25) | (h_channel > 165)) & (s_channel > 40) & (leaf_mask > 0)
    
    # Severe Chlorosis (yellow leaf halo or blight fringe):
    is_chlorosis = ((h_channel >= 20) & (h_channel <= 35)) & (s_channel > 60) & (exg < 15) & (leaf_mask > 0)
    
    # White / Gray mildew:
    is_mildew = (s_channel < 40) & (l_channel > 140) & (exg < 10) & (leaf_mask > 0)
    
    # Combined lesion mask
    lesion_mask_raw = (is_necrotic | is_rust | is_chlorosis | is_mildew).astype(np.uint8) * 255
    
    # Morphological noise filter on lesions
    lesion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_mask = cv2.morphologyEx(lesion_mask_raw, cv2.MORPH_OPEN, lesion_kernel)
    lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)
    
    infected_pixels = int(np.count_nonzero(lesion_mask))
    damage_pct = round((infected_pixels / float(total_leaf_pixels)) * 100.0, 1)
    
    # Bound within realistic biological leaf damage (0.0 to 95.0%)
    damage_pct = min(95.0, max(0.0, damage_pct))
    
    # 3. Categorize according to Phytopathological Standard Severity Scales
    if damage_pct < 2.0:
        stage_label = "Grade 0: Negligible / Healthy"
        severity_color = "#10b981"
        severity_badge = "Healthy Vigor"
        field_advice = "No systemic necrosis detected. Specimen shows intact cuticle and photosynthetic chloroplast density."
    elif damage_pct < 10.0:
        stage_label = "Grade 1: Mild / Incipient"
        severity_color = "#34d399"
        severity_badge = "Early / Localized"
        field_advice = "Early localized infection spots. Pruning infected leaves and preventative bio-fungicide can achieve 100% containment."
    elif damage_pct < 25.0:
        stage_label = "Grade 2: Moderate Infestation"
        severity_color = "#f59e0b"
        severity_badge = "Moderate Active"
        field_advice = "Active pathogen proliferation across foliar lamina. Apply targeted curative spray within 24-48 hours to protect uninfected foliage."
    elif damage_pct < 50.0:
        stage_label = "Grade 3: Severe / Critical"
        severity_color = "#ef4444"
        severity_badge = "High / Spreading"
        field_advice = "Extensive necrotic lesions causing significant photosynthetic impairment. Immediate systemic fungicide and canopy isolation required."
    else:
        stage_label = "Grade 4: Devastating / Terminal"
        severity_color = "#991b1b"
        severity_badge = "Devastating"
        field_advice = "Extreme tissue necrosis. Defoliation imminent. Consider roguing (removing) heavily diseased plants to protect adjacent rows."

    # 4. Generate Visual Contour Overlay with Annotations
    overlay = proc_img.copy()
    
    # Draw subtle green outline around entire leaf canopy
    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (16, 185, 129), 2)
    
    # Find lesion contours and draw glowing hot-spots
    lesion_contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter tiny speckle contours for cleaner visual report
    min_lesion_area = 10
    significant_lesions = [c for c in lesion_contours if cv2.contourArea(c) > min_lesion_area]
    lesion_count = len(significant_lesions)
    
    # Draw filled semi-transparent red highlights on lesions
    lesion_fill = np.zeros_like(proc_img)
    cv2.drawContours(lesion_fill, significant_lesions, -1, (244, 63, 94), -1) # Coral Red
    cv2.drawContours(overlay, significant_lesions, -1, (255, 230, 0), 2) # Yellow border around lesions
    
    # Alpha blend lesion fill with original
    overlay = cv2.addWeighted(overlay, 0.85, lesion_fill, 0.35, 0)
    
    # If scaled down, scale overlay back up
    if scale != 1.0:
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)
        lesion_mask = cv2.resize(lesion_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    overlay_pil = Image.fromarray(overlay)
    mask_pil = Image.fromarray(lesion_mask)
    
    return {
        "damage_percentage": damage_pct,
        "stage_label": stage_label,
        "severity_color": severity_color,
        "severity_badge": severity_badge,
        "field_advice": field_advice,
        "lesion_count": lesion_count,
        "total_leaf_pixels": total_leaf_pixels,
        "infected_pixels": infected_pixels,
        "overlay_image": overlay_pil,
        "mask_image": mask_pil
    }
