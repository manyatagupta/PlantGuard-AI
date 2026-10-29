"""
PlantGuard AI - Treatment Recovery & Before-vs-After Lesion Tracker
Compares dual scans taken before and after agricultural treatment application
to calculate quantitative recovery rate, lesion reduction, and treatment efficacy.
"""

import cv2
import numpy as np
from PIL import Image
from utils.leaf_cv import analyze_leaf_damage, calculate_canopy_spectral_indices

def evaluate_treatment_recovery(image_before, image_after) -> dict:
    """
    Analyzes before and after specimen scans, calculating healing progress
    including foliar necrosis reduction and chlorophyll re-greening (SPAD delta).
    """
    analysis_before = analyze_leaf_damage(image_before)
    analysis_after = analyze_leaf_damage(image_after)
    
    spectral_before = calculate_canopy_spectral_indices(image_before)
    spectral_after = calculate_canopy_spectral_indices(image_after)
    
    dmg_before = analysis_before["damage_percentage"]
    dmg_after = analysis_after["damage_percentage"]
    
    lesions_before = analysis_before["lesion_count"]
    lesions_after = analysis_after["lesion_count"]
    
    spad_before = spectral_before["spad_chlorophyll_proxy"]
    spad_after = spectral_after["spad_chlorophyll_proxy"]
    delta_spad = round(spad_after - spad_before, 1)
    
    delta_damage = round(dmg_before - dmg_after, 1)
    
    if dmg_before > 0:
        recovery_pct = round(((dmg_before - dmg_after) / dmg_before) * 100.0, 1)
    else:
        recovery_pct = 0.0
        
    # Healing classification
    if delta_damage > 0:
        if recovery_pct >= 70.0:
            status = "🟢 Exceptional Healing & Lesion Desiccation"
            status_color = "#10b981"
            guidance = "Outstanding treatment efficacy. Pathogen hyphae have collapsed and active lesions are calloused. Transition from curative spraying to routine preventive monitoring."
        elif recovery_pct >= 30.0:
            status = "🟡 Moderate Contained Recovery"
            status_color = "#f59e0b"
            guidance = "Foliar infection is successfully arrested, but dormant spores may remain. Complete the second scheduled spray cycle to prevent secondary re-infection."
        else:
            status = "🟠 Slow / Sub-optimal Response"
            status_color = "#fb923c"
            guidance = "Infection has plateaued with minor reduction. Verify spray coverage (especially leaf undersides) and water pH. Consider adding an approved organic surfactant."
    else:
        status = "🔴 Pathogen Progression Alert (Resistance / Ineffective MOA)"
        status_color = "#ef4444"
        guidance = "Disease symptoms have expanded despite treatment. Immediately rotate to an alternate FRAC fungicide group with a different mode of action to defeat suspected pathogen resistance."

    # Create composite side-by-side image
    img_b = np.array(analysis_before["overlay_image"].convert("RGB"))
    img_a = np.array(analysis_after["overlay_image"].convert("RGB"))
    
    # Standardize height
    target_h = 400
    w_b = int(img_b.shape[1] * (target_h / float(img_b.shape[0])))
    w_a = int(img_a.shape[1] * (target_h / float(img_a.shape[0])))
    
    res_b = cv2.resize(img_b, (w_b, target_h), interpolation=cv2.INTER_AREA)
    res_a = cv2.resize(img_a, (w_a, target_h), interpolation=cv2.INTER_AREA)
    
    # Divider line
    divider = np.ones((target_h, 8, 3), dtype=np.uint8) * 30
    
    composite = np.hstack([res_b, divider, res_a])
    composite_pil = Image.fromarray(composite)

    return {
        "dmg_before": dmg_before,
        "dmg_after": dmg_after,
        "stage_before": analysis_before["stage_label"],
        "stage_after": analysis_after["stage_label"],
        "lesions_before": lesions_before,
        "lesions_after": lesions_after,
        "spad_before": spad_before,
        "spad_after": spad_after,
        "delta_spad": delta_spad,
        "delta_damage": delta_damage,
        "recovery_pct": recovery_pct,
        "status": status,
        "status_color": status_color,
        "guidance": guidance,
        "composite_image": composite_pil,
        "analysis_before": analysis_before,
        "analysis_after": analysis_after,
        "spectral_before": spectral_before,
        "spectral_after": spectral_after
    }
