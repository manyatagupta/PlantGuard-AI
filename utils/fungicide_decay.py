"""
PlantGuard AI - Fungicide Residue Degradation, Rainfastness & Re-Application Forecaster
Models first-order chemical dissipation kinetics (C(t) = C0 * exp(-k*t)), photothermal
accelerated breakdown, precipitation wash-off dynamics, foliar growth expansion dilution,
and spray tank water quality / alkaline hydrolysis optimization.
"""

import math
import datetime
from typing import Dict, List, Any, Optional, Tuple

# Comprehensive Agricultural Fungicide & Bio-Pesticide Kinetic Database
FUNGICIDE_KINETICS_DATABASE = {
    "Azoxystrobin 22.9% (Quadris / Heritage)": {
        "active_ingredient": "Azoxystrobin",
        "trade_names": ["Quadris", "Heritage 50 WDG", "Abound"],
        "frac_code": "Group 11 (QoI Strobilurin)",
        "chemical_family": "Methoxyacrylate",
        "action_type": "Systemic Translaminar & Xylem Mobile",
        "base_half_life_days": 10.5,
        "rainfast_hours": 2.0,
        "washoff_coefficient": 0.014,  # Low wash-off once absorbed into cuticle
        "uv_sensitivity": 0.85,
        "temp_sensitivity_q10": 1.9,
        "optimal_spray_ph_min": 5.5,
        "optimal_spray_ph_max": 6.8,
        "alkaline_hydrolysis_hazard": "LOW",
        "hydrolysis_half_life_hours_ph8": 140.0,
        "critical_threshold_pct": 20.0,
        "recommended_adjuvant": "Non-Ionic Surfactant (NIS) or Crop Oil Concentrate (COC)",
        "adjuvant_purpose": "Enhances cuticular penetration and droplet spreading."
    },
    "Chlorothalonil 75 WP (Daconil Broad-Spectrum)": {
        "active_ingredient": "Chlorothalonil",
        "trade_names": ["Daconil 2787", "Bravo Weather Stik", "Echo 720"],
        "frac_code": "Group M05 (Multi-Site Chloronitrile)",
        "chemical_family": "Chloronitrile",
        "action_type": "Surface Protectant Contact Barrier",
        "base_half_life_days": 7.5,
        "rainfast_hours": 2.0,
        "washoff_coefficient": 0.024,  # Weather-stik formulations resist wash-off
        "uv_sensitivity": 1.1,
        "temp_sensitivity_q10": 2.0,
        "optimal_spray_ph_min": 5.0,
        "optimal_spray_ph_max": 7.0,
        "alkaline_hydrolysis_hazard": "MODERATE",
        "hydrolysis_half_life_hours_ph8": 36.0,
        "critical_threshold_pct": 25.0,
        "recommended_adjuvant": "Spreader-Sticker or Latex Deposition Aid",
        "adjuvant_purpose": "Maximizes foliar adhesion and resists torrential rain erosion."
    },
    "Mancozeb 75 WP (Dithane M-45 Multi-Site)": {
        "active_ingredient": "Mancozeb (Manganese + Zinc Coordination)",
        "trade_names": ["Dithane M-45", "Manzate Pro-Stick", "Penncozeb"],
        "frac_code": "Group M03 (Multi-Site Dithiocarbamate)",
        "chemical_family": "Ethylenebisdithiocarbamate (EBDC)",
        "action_type": "Surface Protectant Contact Barrier",
        "base_half_life_days": 5.5,
        "rainfast_hours": 4.0,
        "washoff_coefficient": 0.042,  # Vulnerable to heavy rainfall
        "uv_sensitivity": 1.3,
        "temp_sensitivity_q10": 2.1,
        "optimal_spray_ph_min": 5.5,
        "optimal_spray_ph_max": 6.8,
        "alkaline_hydrolysis_hazard": "MODERATE",
        "hydrolysis_half_life_hours_ph8": 28.0,
        "critical_threshold_pct": 20.0,
        "recommended_adjuvant": "Resin / Synthetic Spreader-Sticker",
        "adjuvant_purpose": "Binds multi-site crystalline deposit to waxy cuticle."
    },
    "Captan 50 WP": {
        "active_ingredient": "Captan",
        "trade_names": ["Captan 50 WP", "Captan 80 WDG", "Captec 4L"],
        "frac_code": "Group M04 (Multi-Site Phthalimide)",
        "chemical_family": "Phthalimide",
        "action_type": "Surface Protectant Contact Barrier",
        "base_half_life_days": 4.5,
        "rainfast_hours": 2.0,
        "washoff_coefficient": 0.045,
        "uv_sensitivity": 1.25,
        "temp_sensitivity_q10": 2.2,
        "optimal_spray_ph_min": 5.0,
        "optimal_spray_ph_max": 6.0,
        "alkaline_hydrolysis_hazard": "EXTREME",
        "hydrolysis_half_life_hours_ph8": 0.25,  # Decomposes in 15 minutes at pH 8.0!
        "critical_threshold_pct": 20.0,
        "recommended_adjuvant": "Acidifying Buffer (Citric Acid / AMS)",
        "adjuvant_purpose": "CRITICAL: Acidifies spray tank water to pH 5.5 to prevent instantaneous decomposition."
    },
    "Copper Octanoate (Liquid Copper Soap)": {
        "active_ingredient": "Copper Octanoate (Fixed Copper)",
        "trade_names": ["Cueva", "Bonide Liquid Copper", "Badge SC"],
        "frac_code": "Group M01 (Multi-Site Inorganic Copper)",
        "chemical_family": "Inorganic / Fatty Acid Copper Soap",
        "action_type": "Surface Bactericide & Protectant Fungicide",
        "base_half_life_days": 8.0,
        "rainfast_hours": 3.0,
        "washoff_coefficient": 0.038,
        "uv_sensitivity": 0.50,  # Highly resistant to UV breakdown
        "temp_sensitivity_q10": 1.5,
        "optimal_spray_ph_min": 6.0,
        "optimal_spray_ph_max": 7.5,
        "alkaline_hydrolysis_hazard": "LOW",
        "hydrolysis_half_life_hours_ph8": 300.0,
        "critical_threshold_pct": 20.0,
        "recommended_adjuvant": "Non-Ionic Surfactant (Low rate)",
        "adjuvant_purpose": "Ensure pH stays > 6.0; excessive acidification liberates free cupric ions causing foliar scorch."
    },
    "Myclobutanil (Immunox Systemic Curative)": {
        "active_ingredient": "Myclobutanil",
        "trade_names": ["Immunox", "Rally 40 WSP", "Nova 40W"],
        "frac_code": "Group 3 (DMI Triazole)",
        "chemical_family": "Triazole (Sterol Demethylation Inhibitor)",
        "action_type": "Systemic Curative & Translaminar Kickback",
        "base_half_life_days": 6.5,
        "rainfast_hours": 2.0,
        "washoff_coefficient": 0.016,  # Translaminar absorption protects against wash-off
        "uv_sensitivity": 1.0,
        "temp_sensitivity_q10": 1.9,
        "optimal_spray_ph_min": 5.5,
        "optimal_spray_ph_max": 7.0,
        "alkaline_hydrolysis_hazard": "LOW",
        "hydrolysis_half_life_hours_ph8": 120.0,
        "critical_threshold_pct": 18.0,
        "recommended_adjuvant": "Methylated Seed Oil (MSO) or Penetrant",
        "adjuvant_purpose": "Accelerates kickback penetration into active mycelial infections."
    },
    "Sulfur 80% Wettable Powder": {
        "active_ingredient": "Elemental Sulfur",
        "trade_names": ["Microthiol Disperss", "Kumulus DF", "Thiolux"],
        "frac_code": "Group M02 (Multi-Site Inorganic Sulfur)",
        "chemical_family": "Inorganic Mineral",
        "action_type": "Vapor Action & Surface Protectant",
        "base_half_life_days": 4.0,
        "rainfast_hours": 4.0,
        "washoff_coefficient": 0.048,
        "uv_sensitivity": 0.70,
        "temp_sensitivity_q10": 2.5,  # Sublimes rapidly above 28°C
        "optimal_spray_ph_min": 5.5,
        "optimal_spray_ph_max": 7.0,
        "alkaline_hydrolysis_hazard": "LOW",
        "hydrolysis_half_life_hours_ph8": 200.0,
        "critical_threshold_pct": 20.0,
        "recommended_adjuvant": "Wetting Agent / Spreader (Do NOT mix with oils)",
        "adjuvant_purpose": "Aids uniform foliar dispersion without inducing sulfur burn."
    },
    "Bacillus subtilis (Biofungicide - Serenade)": {
        "active_ingredient": "Bacillus subtilis strain QST 713",
        "trade_names": ["Serenade ASO", "Serenade MAX", "Taegro"],
        "frac_code": "Group BM02 (Biological Microbial)",
        "chemical_family": "Endophytic Microbial Spores & Lipopeptides",
        "action_type": "Biofilm Barrier & Pathogen Antagonism",
        "base_half_life_days": 3.0,
        "rainfast_hours": 4.0,
        "washoff_coefficient": 0.045,
        "uv_sensitivity": 1.80,  # High UV vulnerability; spores sensitive to solar radiation
        "temp_sensitivity_q10": 2.0,
        "optimal_spray_ph_min": 6.0,
        "optimal_spray_ph_max": 7.2,
        "alkaline_hydrolysis_hazard": "LOW",
        "hydrolysis_half_life_hours_ph8": 80.0,
        "critical_threshold_pct": 25.0,
        "recommended_adjuvant": "Organic Yucca Extract or Microbial Spreader",
        "adjuvant_purpose": "Shields beneficial rhizobacteria from rapid desiccation and UV."
    },
    "Neem Oil (Cold-Pressed 70%)": {
        "active_ingredient": "Clarified Hydrophobic Extract of Neem Oil (Azadirachtin)",
        "trade_names": ["Trilogy", "AzaMax", "Pure Neem 70%"],
        "frac_code": "Group UNE (Botanical Bio-Control)",
        "chemical_family": "Tetranortriterpenoid Botanical",
        "action_type": "Foliar Suffocant, Fungistatic & Insect Growth Regulator",
        "base_half_life_days": 2.2,
        "rainfast_hours": 2.0,
        "washoff_coefficient": 0.050,
        "uv_sensitivity": 1.95,  # Severe photodegradation under intense sunlight
        "temp_sensitivity_q10": 2.3,
        "optimal_spray_ph_min": 5.5,
        "optimal_spray_ph_max": 6.5,
        "alkaline_hydrolysis_hazard": "HIGH",
        "hydrolysis_half_life_hours_ph8": 12.0,
        "critical_threshold_pct": 20.0,
        "recommended_adjuvant": "Potassium Cocoate or Mild Emulsifier",
        "adjuvant_purpose": "Maintains oil-in-water micro-emulsion to avoid leaf droplet spotting."
    },
    "Potassium Bicarbonate (MilStop / GreenCure)": {
        "active_ingredient": "Potassium Bicarbonate (KHCO3)",
        "trade_names": ["MilStop", "GreenCure", "Kaligreen"],
        "frac_code": "Group NC (Not Classified - Contact Osmotic Salt)",
        "chemical_family": "Inorganic Carbonate Salt",
        "action_type": "Curative Osmotic Shock & Mycelial Cell Wall Collapse",
        "base_half_life_days": 1.5,
        "rainfast_hours": 1.0,
        "washoff_coefficient": 0.075,  # Highly water-soluble salt; easily washed off
        "uv_sensitivity": 0.60,
        "temp_sensitivity_q10": 1.4,
        "optimal_spray_ph_min": 7.0,
        "optimal_spray_ph_max": 8.0,
        "alkaline_hydrolysis_hazard": "NONE",
        "hydrolysis_half_life_hours_ph8": 500.0,
        "critical_threshold_pct": 15.0,
        "recommended_adjuvant": "Horticultural Spreader or Insecticidal Soap",
        "adjuvant_purpose": "Improves foliar film contact across powdery mildew colonies."
    }
}


def get_all_decay_products() -> List[str]:
    """Returns a list of all registered fungicide/pesticide product names."""
    return list(FUNGICIDE_KINETICS_DATABASE.keys())


def get_product_kinetics(product_name: str) -> Optional[Dict[str, Any]]:
    """Retrieves kinetics specifications for a product by exact or partial key match."""
    if product_name in FUNGICIDE_KINETICS_DATABASE:
        return FUNGICIDE_KINETICS_DATABASE[product_name]
    
    # Partial search
    p_lower = product_name.lower()
    for key, val in FUNGICIDE_KINETICS_DATABASE.items():
        if p_lower in key.lower() or val["active_ingredient"].lower() in p_lower:
            return val
            
    return None


def calculate_foliar_chemical_decay(
    product_name: str,
    initial_concentration_pct: float = 100.0,
    simulation_days: int = 14,
    avg_temperature_c: float = 22.0,
    avg_uv_index: float = 6.0,
    daily_rainfall_mm: Optional[List[float]] = None,
    canopy_growth_rate_pct: float = 3.0,
    start_date: Optional[datetime.date] = None
) -> Dict[str, Any]:
    """
    Computes daily multi-factor foliar fungicide residue degradation:
    1. First-order chemical dissipation kinetics: C(t) = C0 * exp(-k * t)
    2. Photothermal adjustment (Arrhenius Q10 thermal factor & UV photolysis index)
    3. Precipitation wash-off dynamics: W = 1 - exp(-beta * R)
    4. Foliar canopy vegetative growth dilution: C_diluted = C * (1 / (1 + growth_rate * day))
    
    Args:
        product_name: Key in FUNGICIDE_KINETICS_DATABASE or partial match.
        initial_concentration_pct: Starting effective barrier (default 100.0%).
        simulation_days: Duration of the forecast in days (default 14).
        avg_temperature_c: Ambient canopy temperature (°C).
        avg_uv_index: Solar ultraviolet radiation index (1 - 11).
        daily_rainfall_mm: Optional list of rainfall depths in mm for each day.
        canopy_growth_rate_pct: Daily canopy expansion rate (%) diluting deposits.
        start_date: Optional starting calendar date (defaults to today).
        
    Returns:
        Dict containing daily trajectory points, effective half-life, critical depletion day,
        and re-application schedule metrics.
    """
    kinetics = get_product_kinetics(product_name)
    if not kinetics:
        # Fallback generic protectant
        kinetics = {
            "active_ingredient": product_name,
            "trade_names": [product_name],
            "frac_code": "Standard Protectant",
            "chemical_family": "Agricultural Fungicide",
            "action_type": "Foliar Protectant",
            "base_half_life_days": 6.0,
            "rainfast_hours": 2.5,
            "washoff_coefficient": 0.035,
            "uv_sensitivity": 1.0,
            "temp_sensitivity_q10": 2.0,
            "optimal_spray_ph_min": 5.5,
            "optimal_spray_ph_max": 6.8,
            "alkaline_hydrolysis_hazard": "MODERATE",
            "hydrolysis_half_life_hours_ph8": 40.0,
            "critical_threshold_pct": 20.0,
            "recommended_adjuvant": "Non-Ionic Surfactant (NIS)",
            "adjuvant_purpose": "Improves spray coverage."
        }
        
    if start_date is None:
        start_date = datetime.date.today()
        
    if daily_rainfall_mm is None or len(daily_rainfall_mm) < simulation_days:
        # Default dry conditions with zero rain
        daily_rainfall_mm = [0.0] * simulation_days
        
    base_t_half = float(kinetics["base_half_life_days"])
    q10 = float(kinetics["temp_sensitivity_q10"])
    uv_sens = float(kinetics["uv_sensitivity"])
    wash_beta = float(kinetics["washoff_coefficient"])
    crit_threshold = float(kinetics["critical_threshold_pct"])
    
    # 1. Temperature Acceleration: Q10 factor relative to 20°C
    temp_factor = math.pow(q10, (avg_temperature_c - 20.0) / 10.0)
    temp_factor = max(0.4, min(3.5, temp_factor))
    
    # 2. UV Photolysis Factor (baseline UV = 5.0)
    uv_delta = avg_uv_index - 5.0
    uv_factor = 1.0 + (uv_delta * 0.08 * uv_sens)
    uv_factor = max(0.6, min(2.5, uv_factor))
    
    # Realized rate constant k (day^-1)
    k_base = math.log(2.0) / base_t_half
    k_effective = k_base * temp_factor * uv_factor
    effective_half_life_days = round(math.log(2.0) / k_effective, 2)
    
    current_conc = initial_concentration_pct
    trajectory = []
    
    depletion_day = None
    respray_window_start = None
    
    for day in range(simulation_days + 1):
        cal_date = start_date + datetime.timedelta(days=day)
        date_str = cal_date.strftime("%Y-%m-%d")
        
        if day == 0:
            rain_mm = 0.0
            washoff_loss_pct = 0.0
            dilution_factor = 1.0
            daily_loss_chem = 0.0
        else:
            rain_mm = float(daily_rainfall_mm[day - 1]) if (day - 1) < len(daily_rainfall_mm) else 0.0
            
            # Step A: Chemical first-order daily decay
            prev_conc = current_conc
            conc_after_decay = prev_conc * math.exp(-k_effective)
            daily_loss_chem = prev_conc - conc_after_decay
            
            # Step B: Precipitation wash-off
            if rain_mm > 0.0:
                washoff_fraction = 1.0 - math.exp(-wash_beta * rain_mm)
                # Cap maximum single-day rain strip to 90%
                washoff_fraction = min(0.90, washoff_fraction)
                washoff_loss_pct = conc_after_decay * washoff_fraction
                conc_after_rain = conc_after_decay - washoff_loss_pct
            else:
                washoff_loss_pct = 0.0
                conc_after_rain = conc_after_decay
                
            # Step C: Foliar vegetative growth dilution
            daily_growth_frac = max(0.0, canopy_growth_rate_pct / 100.0)
            dilution_multiplier = 1.0 / (1.0 + daily_growth_frac)
            current_conc = max(0.0, conc_after_rain * dilution_multiplier)
            dilution_factor = round(dilution_multiplier, 3)
            
        current_conc = round(current_conc, 2)
        
        # Categorize protection status
        if current_conc >= 60.0:
            status_label = "Optimal Barrier"
            status_color = "#10b981"  # Emerald
            status_badge = "🟢 Robust Protection"
        elif current_conc >= 35.0:
            status_label = "Adequate Residual"
            status_color = "#3b82f6"  # Blue
            status_badge = "🔵 Protective"
        elif current_conc >= crit_threshold:
            status_label = "Depleting Barrier"
            status_color = "#f59e0b"  # Amber
            status_badge = "🟡 Re-Spray Window Open"
            if respray_window_start is None:
                respray_window_start = day
        else:
            status_label = "Critical Exposure"
            status_color = "#ef4444"  # Red
            status_badge = "🔴 Crop Vulnerable"
            if depletion_day is None:
                depletion_day = day
                
        trajectory.append({
            "day": day,
            "date": date_str,
            "concentration_pct": current_conc,
            "rainfall_mm": rain_mm,
            "washoff_loss_pct": round(washoff_loss_pct, 2),
            "status_label": status_label,
            "status_color": status_color,
            "status_badge": status_badge
        })
        
    # Projected days until threshold
    if depletion_day is None:
        # Linear/exponential projection beyond simulation window
        if current_conc > 0.0 and current_conc > crit_threshold:
            remaining_k = k_effective + (canopy_growth_rate_pct / 100.0)
            extra_days = math.log(current_conc / crit_threshold) / remaining_k
            projected_depletion_day = round(simulation_days + extra_days, 1)
        else:
            projected_depletion_day = float(simulation_days)
    else:
        projected_depletion_day = float(depletion_day)
        
    return {
        "product_name": product_name,
        "active_ingredient": kinetics["active_ingredient"],
        "frac_code": kinetics["frac_code"],
        "action_type": kinetics["action_type"],
        "base_half_life_days": base_t_half,
        "effective_half_life_days": effective_half_life_days,
        "rainfast_hours": kinetics["rainfast_hours"],
        "critical_threshold_pct": crit_threshold,
        "simulation_days": simulation_days,
        "temperature_c": avg_temperature_c,
        "uv_index": avg_uv_index,
        "canopy_growth_rate_pct": canopy_growth_rate_pct,
        "days_to_threshold": projected_depletion_day,
        "respray_window_start_day": respray_window_start if respray_window_start is not None else max(0, int(projected_depletion_day - 2)),
        "trajectory": trajectory,
        "kinetics": kinetics
    }


def predict_reapplication_schedule(
    product_name: str,
    application_date: datetime.date,
    current_day_offset: int = 0,
    avg_temp_c: float = 22.0,
    avg_uv: float = 6.0,
    daily_rainfall_mm: Optional[List[float]] = None,
    forecasted_rain_day: Optional[int] = None
) -> Dict[str, Any]:
    """
    Determines actionable calendar re-application windows, factoring impending rain events
    and weather forecasts to advise whether sprays should precede storm fronts.
    """
    decay = calculate_foliar_chemical_decay(
        product_name=product_name,
        simulation_days=14,
        avg_temperature_c=avg_temp_c,
        avg_uv_index=avg_uv,
        daily_rainfall_mm=daily_rainfall_mm,
        start_date=application_date
    )
    
    traj = decay["trajectory"]
    current_pt = traj[min(current_day_offset, len(traj) - 1)]
    current_conc = current_pt["concentration_pct"]
    crit_thresh = decay["critical_threshold_pct"]
    days_to_crit = decay["days_to_threshold"]
    
    respray_cal_date = application_date + datetime.timedelta(days=int(days_to_crit))
    
    # Conflict analysis with upcoming forecast rain
    weather_alert = None
    urgency_level = "ROUTINE"
    action_headline = "Maintain Routine Scouting"
    
    if forecasted_rain_day is not None and forecasted_rain_day > current_day_offset:
        rain_cal_date = application_date + datetime.timedelta(days=forecasted_rain_day)
        # If chemical will be below 40% when rain arrives, advise preventive re-spray BEFORE rain
        rain_conc = traj[min(forecasted_rain_day, len(traj) - 1)]["concentration_pct"]
        if rain_conc < 40.0:
            urgency_level = "ADVANCED_BEFORE_RAIN"
            action_headline = f"Advance Application Before Rain on {rain_cal_date.strftime('%b %d')}"
            weather_alert = (
                f"Rain event arriving on Day {forecasted_rain_day} ({rain_cal_date.strftime('%Y-%m-%d')}). "
                f"Active residue will be at {rain_conc:.1f}%, leaving foliage vulnerable to rain-triggered fungal zoospores. "
                f"Apply protective fungicide 24-48 hours BEFORE rainfall to establish a waterproof barrier."
            )
            respray_cal_date = application_date + datetime.timedelta(days=max(current_day_offset, forecasted_rain_day - 1))
            
    if current_conc < crit_thresh:
        urgency_level = "CRITICAL_IMMEDIATE"
        action_headline = "🚨 Re-Application Due Immediately"
        status_color = "#ef4444"
        action_summary = f"Chemical barrier depleted to {current_conc:.1f}% (below critical {crit_thresh}% threshold). Crop is completely unprotected."
    elif current_conc < 35.0:
        urgency_level = "RESPRAY_WINDOW_ACTIVE"
        action_headline = "🟡 Re-Spray Window Currently Open"
        status_color = "#f59e0b"
        action_summary = f"Residual barrier at {current_conc:.1f}%. Prepare spray equipment and apply within next 24-48 hours."
    else:
        status_color = "#10b981"
        action_summary = f"Crop active barrier is robust at {current_conc:.1f}%. Protection projected through {respray_cal_date.strftime('%B %d')}."
        
    return {
        "product_name": product_name,
        "application_date": application_date.strftime("%Y-%m-%d"),
        "current_day": current_day_offset,
        "current_concentration_pct": current_conc,
        "critical_threshold_pct": crit_thresh,
        "days_remaining_protection": max(0.0, round(days_to_crit - current_day_offset, 1)),
        "recommended_respray_date": respray_cal_date.strftime("%Y-%m-%d"),
        "urgency_level": urgency_level,
        "action_headline": action_headline,
        "action_summary": action_summary,
        "status_color": status_color,
        "weather_alert": weather_alert,
        "decay_model": decay
    }


def evaluate_rainfastness(
    product_name: str,
    hours_since_spray: float,
    rainfall_intensity_mm: float = 12.0
) -> Dict[str, Any]:
    """
    Evaluates whether a foliar application has reached rainfast curing,
    or if an imminent/recent rainfall will cause wash-off failure.
    """
    kinetics = get_product_kinetics(product_name)
    req_rainfast = float(kinetics["rainfast_hours"]) if kinetics else 2.5
    wash_beta = float(kinetics["washoff_coefficient"]) if kinetics else 0.035
    
    curing_pct = min(100.0, round((hours_since_spray / req_rainfast) * 100.0, 1))
    
    if hours_since_spray >= req_rainfast:
        state = "SECURE_RAINFAST"
        badge = "✅ Rainfast & Cured"
        badge_color = "#10b981"
        washoff_multiplier = 1.0
        summary = f"Chemical deposit has completely dried and bonded to foliar cuticle ({hours_since_spray:.1f}h elapsed >= {req_rainfast:.1f}h required). Standard rain wash-off resistance applies."
    elif hours_since_spray >= (req_rainfast * 0.5):
        state = "PARTIAL_CURE"
        badge = "⚠️ Partially Cured"
        badge_color = "#f59e0b"
        washoff_multiplier = 2.2  # Partial drying doubles wash-off vulnerability
        summary = f"Deposit is partially set ({hours_since_spray:.1f}h / {req_rainfast:.1f}h). Moderate wash-off loss expected if precipitation occurs now."
    else:
        state = "UNCURED_VULNERABLE"
        badge = "🚨 Uncured / High Wash-Off Risk"
        badge_color = "#ef4444"
        washoff_multiplier = 5.0  # Fresh wet droplets wash off foliar surface prior to cuticular binding
        summary = f"Foliar deposit is still wet ({hours_since_spray:.1f}h < {req_rainfast:.1f}h). Rainfall will strip up to 85% of active chemical from foliage."
        
    estimated_washoff_pct = round(min(95.0, (1.0 - math.exp(-wash_beta * rainfall_intensity_mm * washoff_multiplier)) * 100.0), 1)
    
    return {
        "product_name": product_name,
        "hours_since_spray": hours_since_spray,
        "required_rainfast_hours": req_rainfast,
        "curing_pct": curing_pct,
        "state": state,
        "badge": badge,
        "badge_color": badge_color,
        "summary": summary,
        "estimated_washoff_loss_pct": estimated_washoff_pct
    }


def optimize_spray_water_and_adjuvant(
    product_name: str,
    water_ph: float = 7.8,
    water_hardness_ppm: float = 240.0,
    tank_volume_gallons: float = 100.0
) -> Dict[str, Any]:
    """
    Evaluates spray tank water quality, alkaline hydrolysis breakdown risks,
    calculates buffering/acidifier adjuvant dosages, and prescribes performance adjuvants.
    
    Args:
        product_name: Name of pesticide/fungicide.
        water_ph: Measured pH of carrier water (5.0 to 9.5).
        water_hardness_ppm: Total hardness as CaCO3 (mg/L).
        tank_volume_gallons: Total spray volume in gallons.
    """
    kinetics = get_product_kinetics(product_name)
    opt_min = float(kinetics["optimal_spray_ph_min"]) if kinetics else 5.5
    opt_max = float(kinetics["optimal_spray_ph_max"]) if kinetics else 6.8
    hazard = kinetics.get("alkaline_hydrolysis_hazard", "MODERATE") if kinetics else "MODERATE"
    half_life_ph8 = float(kinetics.get("hydrolysis_half_life_hours_ph8", 40.0)) if kinetics else 40.0
    rec_adj = kinetics.get("recommended_adjuvant", "Non-Ionic Surfactant") if kinetics else "Non-Ionic Surfactant"
    adj_purpose = kinetics.get("adjuvant_purpose", "Improves foliar coverage.") if kinetics else "Improves foliar coverage."
    
    target_ph = (opt_min + opt_max) / 2.0
    ph_delta = water_ph - opt_max
    
    # 1. Alkaline Hydrolysis Evaluation
    hydrolysis_alert = None
    if water_ph > 7.0 and hazard in ["EXTREME", "HIGH"]:
        est_tank_half_life = round(half_life_ph8 * math.pow(10, -(water_ph - 8.0)), 2)
        est_tank_half_life = max(0.1, est_tank_half_life)
        hydrolysis_alert = (
            f"ALKALINE HYDROLYSIS DANGER: Current carrier water (pH {water_ph:.1f}) rapidly decomposes {kinetics.get('active_ingredient', product_name)}. "
            f"Estimated tank half-life is only {est_tank_half_life * 60:.0f} minutes! "
            f"Active chemical degrades inside the spray tank before reaching the field."
        )
    elif water_ph > opt_max:
        hydrolysis_alert = (
            f"Water pH ({water_ph:.1f}) exceeds optimal stability window ({opt_min:.1f} - {opt_max:.1f}). "
            f"Gradual chemical breakdown will reduce residual persistence."
        )
        
    # 2. Buffering / Acidifier Dosage Calculation
    # Rule of thumb: ~2 oz Citric Acid 50% or agricultural acidifier per 100 gal per 1.0 pH unit drop,
    # adjusted for bicarbonate buffer capacity (hardness)
    hardness_factor = 1.0 + (max(0.0, water_hardness_ppm - 100.0) / 300.0)
    if water_ph > opt_max:
        required_ph_drop = water_ph - target_ph
        citric_acid_oz_per_100gal = round(required_ph_drop * 2.5 * hardness_factor, 1)
        citric_acid_grams_per_100L = round(citric_acid_oz_per_100gal * 7.49, 1)
        total_citric_oz = round(citric_acid_oz_per_100gal * (tank_volume_gallons / 100.0), 1)
        buffer_status = "ACIDIFIER_REQUIRED"
        buffer_badge = "🧪 Acidifier Required"
        buffer_color = "#ef4444"
        buffer_prescription = f"Add {total_citric_oz} oz ({round(total_citric_oz * 28.35, 1)} g) Citric Acid Buffer to bring {tank_volume_gallons:.0f} gal tank water down to pH {target_ph:.1f}."
    elif water_ph < opt_min:
        buffer_status = "TOO_ACIDIC"
        buffer_badge = "⚠️ Water Too Acidic"
        buffer_color = "#f59e0b"
        total_citric_oz = 0.0
        buffer_prescription = f"Carrier water is overly acidic (pH {water_ph:.1f}). Free metal ions may cause foliar burn. Add agricultural neutralizer or Potassium Bicarbonate to raise pH to {opt_min:.1f}."
    else:
        buffer_status = "OPTIMAL_PH"
        buffer_badge = "✅ Optimal Spray Water pH"
        buffer_color = "#10b981"
        total_citric_oz = 0.0
        buffer_prescription = f"Water pH ({water_ph:.1f}) is within the optimal zone ({opt_min:.1f} - {opt_max:.1f}). No acidifying buffer necessary."
        
    # 3. Water Hardness Assessment
    if water_hardness_ppm >= 200.0:
        hardness_rating = "Very Hard Water (>200 ppm)"
        ams_recommendation = f"Add 8.5 to 17 lbs Ammonium Sulfate (AMS) per 100 gal (1-2% w/v) to sequester calcium (Ca2+) and magnesium (Mg2+) cations."
    elif water_hardness_ppm >= 120.0:
        hardness_rating = "Moderately Hard (120-200 ppm)"
        ams_recommendation = "Consider light water conditioner (AMS or citric chelator) if using weak-acid chemistries."
    else:
        hardness_rating = "Soft / Ideal Water (<120 ppm)"
        ams_recommendation = "Cation binding risk is minimal. No water softener required."
        
    return {
        "product_name": product_name,
        "carrier_water_ph": water_ph,
        "target_ph": round(target_ph, 1),
        "optimal_ph_range": f"{opt_min:.1f} - {opt_max:.1f}",
        "water_hardness_ppm": water_hardness_ppm,
        "hardness_rating": hardness_rating,
        "ams_recommendation": ams_recommendation,
        "buffer_status": buffer_status,
        "buffer_badge": buffer_badge,
        "buffer_color": buffer_color,
        "buffer_prescription": buffer_prescription,
        "citric_acid_oz_total": total_citric_oz,
        "hydrolysis_hazard": hazard,
        "hydrolysis_alert": hydrolysis_alert,
        "recommended_adjuvant": rec_adj,
        "adjuvant_purpose": adj_purpose
    }
