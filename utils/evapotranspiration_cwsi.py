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

