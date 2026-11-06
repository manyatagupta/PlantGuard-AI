"""
FAO-56 Penman-Monteith Evapotranspiration, Crop Water Stress Index (CWSI)
and Disease-Irrigation Co-Management Engine
========================================================================
This module implements international benchmark agronomic algorithms for calculating:
1. Reference Evapotranspiration (ET0) using the Food and Agriculture Organization (FAO-56)
   Penman-Monteith physical equation.
2. Crop-specific actual evapotranspiration (ETc) with phenological single and dual
   crop coefficients (Kc_ini, Kc_mid, Kc_end).
3. Empirical Crop Water Stress Index (CWSI) based on infrared canopy thermography (Tc - Ta)
   and atmospheric vapor pressure deficit (VPD) using the Jackson-Idso baseline model.
4. Disease-Irrigation Co-Management: Optimization of irrigation delivery (drip vs overhead)
   and timing windows (dawn vs dusk) to prevent extended leaf wetness duration (LWD)
   and suppress foliar fungal and oomycete sporulation.
5. Volumetric irrigation prescriptions (m3/ha, mm/day, gallons/acre, system runtime hours).

References:
- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration:
  Guidelines for computing crop water requirements. FAO Irrigation and drainage paper 56.
- Idso, S. B., Jackson, R. D., Pinter, P. J., Reginato, R. J., & Hatfield, J. L. (1981).
  Normalizing the stress-degree-day parameter for environmental variability.
  Agricultural Meteorology, 24, 45-55.
- Jackson, R. D., Idso, S. B., Reginato, R. J., & Pinter, P. J. (1981). Canopy temperature
  as a crop water stress indicator. Water Resources Research, 17(4), 1133-1138.
"""

import math
from typing import Dict, Any, List, Optional, Tuple

# Comprehensive FAO-56 Crop Coefficients and Jackson-Idso CWSI Baselines
CROP_COEFFICIENT_DATABASE: Dict[str, Dict[str, Any]] = {
    "Tomato": {
        "common_name": "Tomato (Processing & Fresh Market)",
        "botanical_name": "Solanum lycopersicum",
        "kc_ini": 0.60,
        "kc_mid": 1.15,
        "kc_end": 0.80,
        "max_rooting_depth_m": 1.0,
        "depletion_fraction_p": 0.40,  # Readily available water factor
        "yield_response_factor_ky": 1.05,
        # Jackson-Idso Non-Water-Stressed Baseline (NWSB): (Tc - Ta)_lower = a - b * VPD
        "idso_nwsb_intercept_a": 1.45,
        "idso_nwsb_slope_b": 1.95,
        "idso_upper_limit_deg_c": 4.80,  # Max canopy heating under complete stomatal closure
        "foliar_disease_vulnerability": "High (Early Blight, Late Blight, Septoria, Powdery Mildew)",
        "overhead_irrigation_penalty": 3.20  # Multiplier on foliar spore germination hazard
    },
    "Potato": {
        "common_name": "Potato (Tubers)",
        "botanical_name": "Solanum tuberosum",
        "kc_ini": 0.50,
        "kc_mid": 1.15,
        "kc_end": 0.75,
        "max_rooting_depth_m": 0.60,
        "depletion_fraction_p": 0.35,  # Very sensitive to moisture stress during tuber bulking
        "yield_response_factor_ky": 1.10,
        "idso_nwsb_intercept_a": 1.30,
        "idso_nwsb_slope_b": 1.80,
        "idso_upper_limit_deg_c": 4.20,
        "foliar_disease_vulnerability": "Very High (Late Blight, Early Blight, Blackleg)",
        "overhead_irrigation_penalty": 3.50
    },
    "Pepper": {
        "common_name": "Bell & Hot Pepper",
        "botanical_name": "Capsicum annuum",
        "kc_ini": 0.60,
        "kc_mid": 1.05,
        "kc_end": 0.85,
        "max_rooting_depth_m": 0.80,
        "depletion_fraction_p": 0.30,
        "yield_response_factor_ky": 1.00,
        "idso_nwsb_intercept_a": 1.50,
        "idso_nwsb_slope_b": 2.05,
        "idso_upper_limit_deg_c": 4.60,
        "foliar_disease_vulnerability": "High (Bacterial Spot, Phytophthora Crown Blight, Anthracnose)",
        "overhead_irrigation_penalty": 3.00
    },
    "Apple": {
        "common_name": "Apple (Orchard with Cover Crop)",
        "botanical_name": "Malus domestica",
        "kc_ini": 0.45,
        "kc_mid": 0.95,
        "kc_end": 0.70,
        "max_rooting_depth_m": 1.40,
        "depletion_fraction_p": 0.50,
        "yield_response_factor_ky": 1.00,
        "idso_nwsb_intercept_a": 2.10,
        "idso_nwsb_slope_b": 2.40,
        "idso_upper_limit_deg_c": 5.50,
        "foliar_disease_vulnerability": "High (Apple Scab, Powdery Mildew, Fire Blight)",
        "overhead_irrigation_penalty": 2.80
    },
    "Corn": {
        "common_name": "Maize / Field Corn",
        "botanical_name": "Zea mays",
        "kc_ini": 0.30,
        "kc_mid": 1.20,
        "kc_end": 0.60,
        "max_rooting_depth_m": 1.50,
        "depletion_fraction_p": 0.55,
        "yield_response_factor_ky": 1.25,  # Very high yield loss if stressed at silking
        "idso_nwsb_intercept_a": 1.85,
        "idso_nwsb_slope_b": 2.30,
        "idso_upper_limit_deg_c": 5.20,
        "foliar_disease_vulnerability": "Moderate (Northern Corn Leaf Blight, Gray Leaf Spot, Rust)",
        "overhead_irrigation_penalty": 2.20
    },
    "Grape": {
        "common_name": "Grape / Wine Vineyard",
        "botanical_name": "Vitis vinifera",
        "kc_ini": 0.30,
        "kc_mid": 0.85,
        "kc_end": 0.45,
        "max_rooting_depth_m": 1.80,
        "depletion_fraction_p": 0.50,
        "yield_response_factor_ky": 0.85,  # Moderate deficit irrigation improves berry quality
        "idso_nwsb_intercept_a": 1.60,
        "idso_nwsb_slope_b": 2.10,
        "idso_upper_limit_deg_c": 4.90,
        "foliar_disease_vulnerability": "Very High (Powdery Mildew, Downy Mildew, Black Rot, Botrytis)",
        "overhead_irrigation_penalty": 3.80
    },
    "Strawberry": {
        "common_name": "Strawberry",
        "botanical_name": "Fragaria x ananassa",
        "kc_ini": 0.40,
        "kc_mid": 0.85,
        "kc_end": 0.75,
        "max_rooting_depth_m": 0.35,  # Very shallow root system
        "depletion_fraction_p": 0.25,
        "yield_response_factor_ky": 1.00,
        "idso_nwsb_intercept_a": 1.20,
        "idso_nwsb_slope_b": 1.75,
        "idso_upper_limit_deg_c": 4.00,
        "foliar_disease_vulnerability": "Very High (Botrytis Gray Mold, Anthracnose, Angular Leaf Spot)",
        "overhead_irrigation_penalty": 3.60
    }
}

# Irrigation delivery systems efficiency & foliar wetting duration
IRRIGATION_DELIVERY_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "Subsurface Drip": {
        "application_efficiency_pct": 92.0,
        "added_leaf_wetness_hours": 0.0,
        "spore_splash_hazard": "None",
        "evaporative_loss_pct": 5.0,
        "disease_risk_multiplier": 1.00,
        "recommended_for": "All crops, particularly disease-sensitive solanaceous and cucurbits"
    },
    "Surface Drip / T-Tape": {
        "application_efficiency_pct": 88.0,
        "added_leaf_wetness_hours": 0.0,
        "spore_splash_hazard": "Negligible",
        "evaporative_loss_pct": 10.0,
        "disease_risk_multiplier": 1.05,
        "recommended_for": "Row crops, strawberries, vegetables with plastic mulch"
    },
    "Micro-Sprinkler / Under-Canopy": {
        "application_efficiency_pct": 80.0,
        "added_leaf_wetness_hours": 1.5,
        "spore_splash_hazard": "Low to Moderate",
        "evaporative_loss_pct": 18.0,
        "disease_risk_multiplier": 1.45,
        "recommended_for": "Orchards, vineyards where trunk irrigation avoids foliage"
    },
    "Center Pivot / Overhead Sprinkler": {
        "application_efficiency_pct": 72.0,
        "added_leaf_wetness_hours": 6.5,
        "spore_splash_hazard": "Severe (Forces zoospore and conidia dispersal)",
        "evaporative_loss_pct": 28.0,
        "disease_risk_multiplier": 2.95,
        "recommended_for": "Extensive field corn, wheat, pasture; high risk for vegetables"
    },
    "Furrow / Flood Irrigation": {
        "application_efficiency_pct": 58.0,
        "added_leaf_wetness_hours": 0.5,
        "spore_splash_hazard": "Low foliar, but extreme Phytophthora crown saturation hazard",
        "evaporative_loss_pct": 35.0,
        "disease_risk_multiplier": 1.80,
        "recommended_for": "Leveled basin soils; requires strict drainage to prevent water molds"
    }
}


def calculate_fao56_penman_monteith(
    temp_mean_c: float,
    relative_humidity_pct: float,
    solar_radiation_mj_m2_day: float = 19.5,
    wind_speed_2m_m_s: float = 2.0,
    elevation_m: float = 120.0
) -> Dict[str, Any]:
    """
    Computes daily reference evapotranspiration (ET0 in mm/day) using the
    FAO-56 Penman-Monteith physical equation.
    
    Args:
        temp_mean_c: Mean daily air temperature in °C.
        relative_humidity_pct: Mean daily relative humidity in % (0 - 100).
        solar_radiation_mj_m2_day: Net solar radiation in MJ/m²/day (default 19.5 MJ/m²/day).
        wind_speed_2m_m_s: Wind speed measured at 2m height in m/s (default 2.0 m/s).
        elevation_m: Station elevation above sea level in meters (default 120m).
        
    Returns:
        Dictionary containing ET0, vapor pressures, slope delta, and psychrometric constant.
    """
    # 1. Atmospheric pressure P (kPa) at elevation z
    p_kpa = 101.3 * math.pow((293.0 - 0.0065 * elevation_m) / 293.0, 5.26)
    
    # 2. Psychrometric constant gamma (kPa / °C)
    # gamma = 0.665e-3 * P
    gamma = 0.665e-3 * p_kpa
    
    # 3. Saturation vapor pressure e_sat (kPa) at temperature T
    e_sat = 0.6108 * math.exp((17.27 * temp_mean_c) / (temp_mean_c + 237.3))
    
    # 4. Actual vapor pressure e_act (kPa)
    bounded_rh = max(5.0, min(100.0, relative_humidity_pct))
    e_act = e_sat * (bounded_rh / 100.0)
    
    # 5. Vapor Pressure Deficit (VPD in kPa)
    vpd_kpa = max(0.01, e_sat - e_act)
    
    # 6. Slope of saturation vapor pressure curve delta (kPa / °C)
    delta = (4098.0 * (0.6108 * math.exp((17.27 * temp_mean_c) / (temp_mean_c + 237.3)))) / math.pow(temp_mean_c + 237.3, 2)
    
    # 7. Net radiation Rn (MJ / m² / day)
    # For daily calculations, soil heat flux G is negligible (G = 0)
    rn = solar_radiation_mj_m2_day
    g_flux = 0.0
    
    # 8. FAO-56 Penman-Monteith equation:
    # ET0 = [0.408 * delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)] / [delta + gamma * (1 + 0.34 * u2)]
    u2 = max(0.1, wind_speed_2m_m_s)
    
    num_rad = 0.408 * delta * (rn - g_flux)
    num_aero = gamma * (900.0 / (temp_mean_c + 273.0)) * u2 * vpd_kpa
    denom = delta + gamma * (1.0 + 0.34 * u2)
    
    et0_mm_day = (num_rad + num_aero) / denom
    et0_mm_day = max(0.1, min(15.0, et0_mm_day))  # Realistic meteorological bounds
    
    return {
        "et0_mm_day": round(et0_mm_day, 2),
        "atmospheric_pressure_kpa": round(p_kpa, 2),
        "psychrometric_constant_gamma": round(gamma, 4),
        "saturation_vapor_pressure_kpa": round(e_sat, 3),
        "actual_vapor_pressure_kpa": round(e_act, 3),
        "vapor_pressure_deficit_kpa": round(vpd_kpa, 3),
        "slope_vapor_pressure_delta": round(delta, 4),
        "radiation_component_mm": round(num_rad / denom, 2),
        "aerodynamic_component_mm": round(num_aero / denom, 2)
    }


def calculate_crop_water_stress_index(
    canopy_temp_c: float,
    air_temp_c: float,
    vpd_kpa: float,
    crop: str = "Tomato"
) -> Dict[str, Any]:
    """
    Computes the empirical Crop Water Stress Index (CWSI in range [0.0, 1.0])
    using the Jackson-Idso baseline thermography model.
    
    Formula:
        Tc - Ta = Canopy temperature depression
        (Tc - Ta)_lower = a - b * VPD (Non-Water-Stressed Baseline)
        (Tc - Ta)_upper = Constant upper limit under complete stomatal shutdown
        CWSI = [(Tc - Ta) - (Tc - Ta)_lower] / [(Tc - Ta)_upper - (Tc - Ta)_lower]
    """
    crop_info = CROP_COEFFICIENT_DATABASE.get(crop, CROP_COEFFICIENT_DATABASE["Tomato"])
    
    a = crop_info["idso_nwsb_intercept_a"]
    b = crop_info["idso_nwsb_slope_b"]
    upper_limit = crop_info["idso_upper_limit_deg_c"]
    
    # Canopy temperature depression
    delta_t = canopy_temp_c - air_temp_c
    
    # Non-water-stressed baseline lower limit (fully transpiring)
    lower_limit = a - (b * vpd_kpa)
    
    # Ensure upper limit is always higher than lower limit
    effective_upper = max(lower_limit + 1.5, upper_limit)
    
    # CWSI calculation
    denom = effective_upper - lower_limit
    if denom <= 0:
        cwsi = 0.5
    else:
        cwsi = (delta_t - lower_limit) / denom
        
    cwsi = round(float(max(0.0, min(1.0, cwsi))), 3)
    
    # Classification of water stress and physiological state
    if cwsi <= 0.20:
        status = "NO STRESS / LUXURY TRANSPIRATION"
        color = "#16a34a"
        advisory = "Canopy stomata are fully open with unrestricted transpiration cooling. Soil moisture is optimal."
    elif cwsi <= 0.40:
        status = "MILD REGULATED DEFICIT"
        color = "#2563eb"
        advisory = "Mild stomatal constriction. Beneficial for fruit sugar accumulation in tomatoes and winegrapes."
    elif cwsi <= 0.60:
        status = "MODERATE WATER STRESS"
        color = "#d97706"
        advisory = "Stomatal conductance is constrained by ~50%. Irrigation replenishment is recommended within 24 hours."
    elif cwsi <= 0.80:
        status = "SEVERE DROUGHT STRESS"
        color = "#ea580c"
        advisory = "Severe moisture deficit inducing canopy heating and leaf roll. Immediate irrigation mandatory to prevent yield loss."
    else:
        status = "CRITICAL DESICCATION HAZARD"
        color = "#dc2626"
        advisory = "Canopy stomata completely shut down; thermal runaway with risk of irreversible cavitation and blossom drop."
        
    return {
        "cwsi": cwsi,
        "cwsi_pct": round(cwsi * 100.0, 1),
        "canopy_temp_c": canopy_temp_c,
        "air_temp_c": air_temp_c,
        "temp_depression_delta_t": round(delta_t, 2),
        "nwsb_lower_limit_c": round(lower_limit, 2),
        "upper_limit_c": round(effective_upper, 2),
        "status": status,
        "color": color,
        "advisory": advisory,
        "crop": crop
    }


def optimize_irrigation_and_disease_schedule(
    crop: str = "Tomato",
    growth_stage: str = "Mid-Season / Fruit Bulking",
    et0_mm_day: float = 5.2,
    effective_rainfall_mm: float = 0.0,
    delivery_system: str = "Subsurface Drip",
    target_area_ha: float = 1.0,
    planned_irrigation_time: str = "05:00 AM (Dawn)"
) -> Dict[str, Any]:
    """
    Computes precise irrigation volume and audits foliar disease risk synergy
    based on irrigation method and diurnal timing.
    """
    crop_info = CROP_COEFFICIENT_DATABASE.get(crop, CROP_COEFFICIENT_DATABASE["Tomato"])
    delivery_info = IRRIGATION_DELIVERY_SYSTEMS.get(delivery_system, IRRIGATION_DELIVERY_SYSTEMS["Subsurface Drip"])
    
    # 1. Determine phenological Crop Coefficient (Kc)
    stage_lower = growth_stage.lower()
    if "ini" in stage_lower or "seed" in stage_lower or "transplant" in stage_lower:
        kc = crop_info["kc_ini"]
        stage_label = "Initial Emergence / Early Vegetative"
    elif "mid" in stage_lower or "bulk" in stage_lower or "flower" in stage_lower or "fruit" in stage_lower:
        kc = crop_info["kc_mid"]
        stage_label = "Mid-Season / Peak Transpiration"
    else:
        kc = crop_info["kc_end"]
        stage_label = "Late Season / Maturation & Ripening"
        
    # 2. Actual Crop Evapotranspiration ETc = Kc * ET0 (mm/day)
    etc_mm_day = round(kc * et0_mm_day, 2)
    
    # 3. Net Irrigation Requirement (mm)
    net_irrigation_mm = max(0.0, etc_mm_day - effective_rainfall_mm)
    
    # 4. Gross Irrigation Requirement accounting for application efficiency
    eff_pct = delivery_info["application_efficiency_pct"]
    gross_irrigation_mm = round(net_irrigation_mm / (eff_pct / 100.0), 2) if eff_pct > 0 else net_irrigation_mm
    
    # 5. Volumetric Requirement
    # 1 mm over 1 ha = 10 m³ = 10,000 Liters
    # 1 ha = 2.47105 acres -> 1 mm over 1 acre = 1,071.3 gallons
    volume_m3 = round(gross_irrigation_mm * 10.0 * target_area_ha, 1)
    volume_liters = round(volume_m3 * 1000.0, 0)
    target_area_acres = target_area_ha * 2.47105
    volume_gallons = round(gross_irrigation_mm * 1071.3 * target_area_acres, 0)
    
    # 6. Delivery System Runtime
    # Typical delivery capacities: Drip = 2.5 mm/hr, Pivot = 6.0 mm/hr, Furrow = 15.0 mm/hr
    if "drip" in delivery_system.lower():
        sys_rate_mm_hr = 2.5
    elif "sprinkler" in delivery_system.lower() or "pivot" in delivery_system.lower():
        sys_rate_mm_hr = 6.0
    else:
        sys_rate_mm_hr = 12.0
    runtime_hours = round(gross_irrigation_mm / sys_rate_mm_hr, 1)
    
    # 7. Disease & Leaf Wetness Timing Risk Analysis
    timing_lower = planned_irrigation_time.lower()
    if "dusk" in timing_lower or "night" in timing_lower or "evening" in timing_lower or "pm" in timing_lower and any(h in timing_lower for h in ["18", "19", "20", "21", "22", "6:", "7:", "8:", "9:"]):
        timing_risk = "DANGEROUS LEAF WETNESS TRAP"
        timing_color = "#dc2626"
        timing_desc = (
            "Overnight irrigation traps free water films on leaf surfaces for 8-12 continuous hours without solar drying. "
            "Multiplies fungal zoospore germination (Phytophthora, Alternaria, Downy Mildew) by up to 3.8x."
        )
        timing_hazard_multiplier = 3.5
    elif "mid-day" in timing_lower or "noon" in timing_lower or "afternoon" in timing_lower:
        timing_risk = "SUB-OPTIMAL / HIGH EVAPORATION"
        timing_color = "#d97706"
        timing_desc = (
            "Mid-day applications suffer 25-35% evaporative loss before reaching the root zone. "
            "Can cause optical lens scorching on sensitive leaves under high UV index."
        )
        timing_hazard_multiplier = 1.3
    else:
        # Dawn / Early morning (4:00 AM - 8:00 AM)
        timing_risk = "AGRONOMICALLY OPTIMAL"
        timing_color = "#16a34a"
        timing_desc = (
            "Dawn irrigation utilizes natural morning dew period. Rising morning sun and thermal wind currents "
            "dry the foliar canopy within 30-45 minutes, neutralizing pathogen germination windows."
        )
        timing_hazard_multiplier = 1.0
        
    combined_disease_hazard = round(delivery_info["disease_risk_multiplier"] * timing_hazard_multiplier, 2)
    
    if combined_disease_hazard >= 3.0:
        overall_pathology_risk = "CRITICAL PATHOGEN MULTIPLIER"
        pathology_color = "#dc2626"
    elif combined_disease_hazard >= 1.8:
        overall_pathology_risk = "ELEVATED DISEASE HAZARD"
        pathology_color = "#ea580c"
    elif combined_disease_hazard >= 1.2:
        overall_pathology_risk = "MODERATE BIO-RISK"
        pathology_color = "#d97706"
    else:
        overall_pathology_risk = "LOW DISEASE MULTIPLIER"
        pathology_color = "#16a34a"
        
    return {
        "crop": crop,
        "growth_stage": stage_label,
        "crop_coefficient_kc": kc,
        "reference_et0_mm": et0_mm_day,
        "actual_crop_etc_mm": etc_mm_day,
        "effective_rainfall_mm": effective_rainfall_mm,
        "net_irrigation_mm": net_irrigation_mm,
        "gross_irrigation_mm": gross_irrigation_mm,
        "application_efficiency_pct": eff_pct,
        "volume_m3": volume_m3,
        "volume_liters": volume_liters,
        "volume_gallons": volume_gallons,
        "system_runtime_hours": runtime_hours,
        "delivery_system": delivery_system,
        "added_leaf_wetness_hours": delivery_info["added_leaf_wetness_hours"],
        "timing_risk_status": timing_risk,
        "timing_risk_color": timing_color,
        "timing_advisory": timing_desc,
        "combined_disease_hazard_multiplier": combined_disease_hazard,
        "overall_pathology_risk": overall_pathology_risk,
        "pathology_color": pathology_color,
        "target_area_ha": target_area_ha,
        "target_area_acres": round(target_area_acres, 2)
    }


def generate_irrigation_depletion_curve(
    daily_etc_mm: float,
    rooting_depth_m: float = 0.8,
    soil_water_capacity_mm_per_m: float = 140.0,
    depletion_fraction_p: float = 0.40,
    horizon_days: int = 7
) -> List[Dict[str, Any]]:
    """
    Simulates forward daily soil water depletion trajectory from field capacity
    to determine the exact day non-stress threshold (RAW) is breached.
    
    Total Available Water: TAW = Rooting_Depth * Soil_Capacity
    Readily Available Water: RAW = p * TAW
    """
    taw_mm = rooting_depth_m * soil_water_capacity_mm_per_m
    raw_mm = depletion_fraction_p * taw_mm
    
    current_depletion_mm = 0.0
    trajectory = []
    
    for day in range(1, horizon_days + 1):
        current_depletion_mm += daily_etc_mm
        moisture_remaining_pct = max(0.0, 100.0 * (1.0 - (current_depletion_mm / taw_mm)))
        is_stressed = current_depletion_mm > raw_mm
        
        trajectory.append({
            "day": day,
            "cumulative_depletion_mm": round(current_depletion_mm, 1),
            "moisture_remaining_pct": round(moisture_remaining_pct, 1),
            "raw_threshold_mm": round(raw_mm, 1),
            "taw_total_mm": round(taw_mm, 1),
            "is_water_stressed": is_stressed,
            "status": "WATER STRESSED" if is_stressed else "OPTIMAL WATER"
        })
        
    return trajectory


def export_irrigation_assessment_json(
    et0_data: Dict[str, Any],
    cwsi_data: Dict[str, Any],
    schedule_data: Dict[str, Any]
) -> str:
    """
    Exports a structured RFC-8259 JSON audit payload for precision irrigation controllers
    and Farm Management Information Systems (FMIS).
    """
    import json
    import datetime
    
    payload = {
        "audit_metadata": {
            "schema": "PlantGuard-FAO56-Irrigation-v1.0",
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "evaluator": "PlantGuard AI Evapotranspiration & CWSI Engine"
        },
        "reference_evapotranspiration_fao56": et0_data,
        "crop_water_stress_index_cwsi": cwsi_data,
        "precision_irrigation_prescription": schedule_data
    }
    
    return json.dumps(payload, indent=2)
