"""
Soilborne Pathogen & Seedling Damping-Off Bio-Epidemiological Risk & Soil Health Suppressiveness Engine
======================================================================================================
This module provides precision agronomic algorithms for modeling soilborne fungal and oomycete
pathogen epidemiology, seedling pre- and post-emergence damping-off risk, and quantitative soil
biological disease suppressiveness scoring.

Key Capabilities:
1. Hydro-Thermal Infection Index (HTI):
   Non-linear cardinal temperature (T_min, T_opt, T_max) and soil moisture (% Field Capacity)
   response functions with zoospore motility jumps for Oomycetes (Pythium, Phytophthora).
2. Edaphic & Soil Physical Modifiers:
   Texture drainage impedance (heavy clay vs silt loam vs sand) and compaction / bulk density
   penalties creating root asphyxiation and hypocotyl lesion vulnerability.
3. Emergence Lag & Exposure Vulnerability:
   GDD accumulation rate in cold, saturated soils extending seedling exposure to soil inoculum.
4. Quantitative Soil Biological Suppressiveness:
   Evaluation of Soil Organic Matter (SOM %), resident/inoculated beneficial antagonists
   (Trichoderma harzianum, Bacillus amyloliquefaciens, Streptomyces), crop rotation break depth,
   and tillage practices (no-till biological crust vs deep moldboard disruption).
5. Precision Seed Treatment & In-Furrow Chemigation Regimens:
   Targeted chemical (Fludioxonil, Mefenoxam, Sedaxane, Oxathiapiprolin) and biological (viable CFU/kg)
   prescriptions with FRAC resistance management and Brassica biofumigation protocols.
"""

import math
from typing import Dict, Any, List, Optional, Tuple

# Comprehensive database of destructive soilborne pathogen complexes
SOILBORNE_PATHOGENS_DATABASE: Dict[str, Dict[str, Any]] = {
    "Pythium_ultimum": {
        "common_name": "Pythium Seed Rot & Cold-Soil Damping-Off",
        "organism_class": "Oomycete",
        "target_crops": ["Tomato", "Potato", "Corn", "Soybean", "Pepper", "Cotton", "Brassicas"],
        "temp_min_c": 8.0,
        "temp_opt_c": 17.5,
        "temp_max_c": 26.0,
        "optimal_moisture_fc_min": 75.0,  # % of Field Capacity
        "optimal_moisture_fc_max": 100.0,
        "motile_zoospores": True,
        "favored_ph_range": (5.2, 7.5),
        "primary_symptoms": [
            "Water-soaked hypocotyl collapse at soil line",
            "Pre-emergence seed decay (soft brown seed mummies)",
            "Root tip browning and sloughing cortex",
            "Stunted seedling patches in low field hollows"
        ],
        "survival_structures": "Thick-walled oospores surviving 6-10 years in uncultivated soil",
        "conducive_conditions": "Cold, waterlogged soils with poor drainage, prolonged emergence (>10 days)",
        "frac_chemical_group": "FRAC 4 (PA-fungicides: Mefenoxam / Metalaxyl-M)",
        "biological_antagonists": ["Trichoderma harzianum T-22", "Bacillus amyloliquefaciens D747", "Pseudomonas fluorescens"]
    },
    "Pythium_aphanidermatum": {
        "common_name": "Pythium High-Temp Root Rot & Damping-Off",
        "organism_class": "Oomycete",
        "target_crops": ["Tomato", "Cucumber", "Pepper", "Lettuce", "Melon"],
        "temp_min_c": 22.0,
        "temp_opt_c": 32.0,
        "temp_max_c": 40.0,
        "optimal_moisture_fc_min": 80.0,
        "optimal_moisture_fc_max": 100.0,
        "motile_zoospores": True,
        "favored_ph_range": (5.5, 7.8),
        "primary_symptoms": [
            "Rapid crown collapse under high ambient heat",
            "Extensive root system rot with honey-brown discoloration",
            "Pythium leak / cottony post-emergence wilting"
        ],
        "survival_structures": "Oospores and persistent zoosporangia in wet irrigation lines and soil",
        "conducive_conditions": "Hot greenhouse substrates, saturated soils with stagnant warm water, salinity stress",
        "frac_chemical_group": "FRAC 4 (Mefenoxam), FRAC 49 (Oxathiapiprolin)",
        "biological_antagonists": ["Bacillus subtilis QST 713", "Trichoderma virens G-41"]
    },
    "Rhizoctonia_solani": {
        "common_name": "Rhizoctonia Wirestem, Canker & Black Scurf",
        "organism_class": "Basidiomycete (Imperfect)",
        "target_crops": ["Potato", "Tomato", "Soybean", "Sugarbeet", "Cotton", "Pepper"],
        "temp_min_c": 12.0,
        "temp_opt_c": 24.0,
        "temp_max_c": 33.0,
        "optimal_moisture_fc_min": 45.0,
        "optimal_moisture_fc_max": 75.0,
        "motile_zoospores": False,
        "favored_ph_range": (5.5, 7.2),
        "primary_symptoms": [
            "Sunken, reddish-brown dry cankers on lower stems (wirestem)",
            "Girdled hypocotyl causing seedling tipping without mushiness",
            "Aerial black sclerotial crusts on potato tubers (black scurf)",
            "Brown right-angle branching hyphae under stereomicroscope"
        ],
        "survival_structures": "Resistant melanized sclerotia in soil and vegetative hyphae on crop debris",
        "conducive_conditions": "Moderately moist, warm, aerated topsoil; deep seed planting; compacted crusted soil",
        "frac_chemical_group": "FRAC 12 (Fludioxonil), FRAC 7 (SDHI: Sedaxane, Fluxapyroxad)",
        "biological_antagonists": ["Trichoderma asperellum", "Bacillus amyloliquefaciens", "Rhizobacteria"]
    },
    "Fusarium_oxysporum": {
        "common_name": "Fusarium Vascular Wilt & Crown Foot Rot",
        "organism_class": "Ascomycete",
        "target_crops": ["Tomato", "Banana", "Cotton", "Melon", "Legumes", "Onion"],
        "temp_min_c": 18.0,
        "temp_opt_c": 28.0,
        "temp_max_c": 36.0,
        "optimal_moisture_fc_min": 40.0,
        "optimal_moisture_fc_max": 70.0,
        "motile_zoospores": False,
        "favored_ph_range": (4.8, 6.2),  # Distinctly favored by acidic soils
        "primary_symptoms": [
            "Unilateral yellowing of lower leaves (one side of stem first)",
            "Distinct brown discoloration of xylem vascular ring",
            "Mid-day wilting followed by irreversible flag wilting",
            "Stunted internodes with epinasty of leaves"
        ],
        "survival_structures": "Chlamydospores persisting in soil for 10-20 years even in fallow fields",
        "conducive_conditions": "Warm acidic soils, high ammonium nitrogen (NH4+) fertilizer, root knot nematode wounds",
        "frac_chemical_group": "FRAC 1 (Benzimidazoles - restricted), FRAC 3 (Prothioconazole / Difenoconazole)",
        "biological_antagonists": ["Non-pathogenic Fusarium Fo47", "Trichoderma harzianum", "Streptomyces lydicus WYEC 108"]
    },
    "Phytophthora_capsici": {
        "common_name": "Phytophthora Crown, Root & Fruit Blight",
        "organism_class": "Oomycete",
        "target_crops": ["Pepper", "Tomato", "Squash", "Pumpkin", "Cucumber", "Eggplant"],
        "temp_min_c": 15.0,
        "temp_opt_c": 27.0,
        "temp_max_c": 35.0,
        "optimal_moisture_fc_min": 85.0,
        "optimal_moisture_fc_max": 100.0,
        "motile_zoospores": True,
        "favored_ph_range": (5.5, 7.5),
        "primary_symptoms": [
            "Sudden permanent wilt without prior yellowing (green collapse)",
            "Dark brown to black water-soaked lesion girdling the crown",
            "Root system completely sloughed off, leaving taproot skeleton",
            "White velvety sporangiophores on diseased stem collars under high humidity"
        ],
        "survival_structures": "Sexual oospores persistent in soil for 5-8 years, readily spread via irrigation run-off",
        "conducive_conditions": "Standing surface water (>1-2 hours), compacted hardpans, furrow irrigation flooding",
        "frac_chemical_group": "FRAC 43 (Fluopicolide), FRAC 40 (Dimethomorph), FRAC 49 (Oxathiapiprolin)",
        "biological_antagonists": ["Bacillus subtilis", "Trichoderma harzianum Rifai", "Streptomyces griseoviridis"]
    },
    "Verticillium_dahliae": {
        "common_name": "Verticillium Vascular Wilt & Defoliation",
        "organism_class": "Ascomycete",
        "target_crops": ["Potato", "Tomato", "Cotton", "Strawberry", "Olive", "Eggplant"],
        "temp_min_c": 14.0,
        "temp_opt_c": 22.5,
        "temp_max_c": 30.0,
        "optimal_moisture_fc_min": 50.0,
        "optimal_moisture_fc_max": 80.0,
        "motile_zoospores": False,
        "favored_ph_range": (6.5, 8.2),  # Favored by neutral to alkaline soils
        "primary_symptoms": [
            "V-shaped chlorotic to necrotic foliar lesions bounded by veins",
            "Olive-brown to black vascular ring staining in stem cross-section",
            "Premature defoliation beginning from lower nodes upward",
            "Early senescence ('early dying syndrome' in potatoes)"
        ],
        "survival_structures": "Black microsclerotia viable up to 15 years in soil profile",
        "conducive_conditions": "Cool to moderate temperatures, alkaline soils, continuous solanaceous rotations",
        "frac_chemical_group": "Cultural / Biofumigation primary; FRAC 3 (Triazoles suppressive)",
        "biological_antagonists": ["Verticillium nonalfalfae (cross-protection)", "Trichoderma koningii", "Glomus intraradices (AMF)"]
    },
    "Macrophomina_phaseolina": {
        "common_name": "Macrophomina Charcoal Rot & Dry Stem Blight",
        "organism_class": "Ascomycete",
        "target_crops": ["Soybean", "Corn", "Sorghum", "Cotton", "Sunflower", "Strawberry"],
        "temp_min_c": 24.0,
        "temp_opt_c": 35.0,
        "temp_max_c": 42.0,
        "optimal_moisture_fc_min": 15.0,
        "optimal_moisture_fc_max": 45.0,  # Favored by drought / dry soils
        "motile_zoospores": False,
        "favored_ph_range": (5.5, 7.5),
        "primary_symptoms": [
            "Silver-grey discoloration of lower stem and root epidermis",
            "Pith filled with microscopic jet-black microsclerotia (charcoal appearance)",
            "Premature death with withered dry leaves clinging to petioles",
            "Severe lodging under late-season moisture stress"
        ],
        "survival_structures": "Microsclerotia persisting in dry topsoil for 3-5 years",
        "conducive_conditions": "Severe drought stress, soil moisture deficit, high temperatures (>32°C), sandy soils",
        "frac_chemical_group": "FRAC 7 (SDHI: Fluxapyroxad seed treatment), FRAC 11 (Azoxystrobin)",
        "biological_antagonists": ["Trichoderma viride", "Pseudomonas chlororaphis", "Arbuscular Mycorrhizal Fungi"]
    }
}

# Soil texture physical properties and drainage impedance
SOIL_TEXTURE_MODIFIERS: Dict[str, Dict[str, Any]] = {
    "Clay": {
        "drainage_speed": "Very Slow",
        "water_retention": "High",
        "drainage_impedance_factor": 1.45,
        "oomycete_risk_boost": 1.35,  # Flooding/water film formation
        "rhizoctonia_risk_boost": 0.85,
        "aeration_score": 30.0
    },
    "Clay Loam": {
        "drainage_speed": "Moderate Slow",
        "water_retention": "Medium-High",
        "drainage_impedance_factor": 1.25,
        "oomycete_risk_boost": 1.20,
        "rhizoctonia_risk_boost": 0.95,
        "aeration_score": 50.0
    },
    "Silt Loam": {
        "drainage_speed": "Moderate",
        "water_retention": "High",
        "drainage_impedance_factor": 1.05,
        "oomycete_risk_boost": 1.05,
        "rhizoctonia_risk_boost": 1.00,
        "aeration_score": 65.0
    },
    "Sandy Loam": {
        "drainage_speed": "Fast",
        "water_retention": "Low-Medium",
        "drainage_impedance_factor": 0.85,
        "oomycete_risk_boost": 0.75,
        "rhizoctonia_risk_boost": 1.15,
        "aeration_score": 85.0
    },
    "Sand": {
        "drainage_speed": "Very Fast",
        "water_retention": "Very Low",
        "drainage_impedance_factor": 0.65,
        "oomycete_risk_boost": 0.50,
        "rhizoctonia_risk_boost": 1.30,  # Dry topsoil canker promotion
        "aeration_score": 95.0
    }
}

# Compaction level / bulk density modifiers
SOIL_COMPACTION_FACTORS: Dict[str, Dict[str, Any]] = {
    "Low / Loose": {
        "penetrometer_psi": "< 150 PSI",
        "compaction_multiplier": 0.85,
        "emergence_delay_days": 0.0,
        "root_restriction_pct": 5.0
    },
    "Moderate": {
        "penetrometer_psi": "150 - 250 PSI",
        "compaction_multiplier": 1.05,
        "emergence_delay_days": 1.5,
        "root_restriction_pct": 20.0
    },
    "Severe / Plow-Pan": {
        "penetrometer_psi": "> 300 PSI",
        "compaction_multiplier": 1.40,
        "emergence_delay_days": 4.0,
        "root_restriction_pct": 55.0
    }
}


def calculate_thermal_suitability(temp_c: float, t_min: float, t_opt: float, t_max: float) -> float:
    """
    Calculates asymmetric non-linear thermal suitability index in range [0.0, 1.0]
    using the Yan and Hunt (1999) cardinal temperature biological response curve.
    
    Formula:
        f(T) = ((T_max - T) / (T_max - T_opt)) * ((T - T_min) / (T_opt - T_min)) ** ((T_opt - T_min) / (T_max - T_opt))
    """
    if temp_c <= t_min or temp_c >= t_max:
        return 0.0
        
    denom1 = t_max - t_opt
    denom2 = t_opt - t_min
    
    if denom1 <= 0 or denom2 <= 0:
        return 0.5
        
    exponent = denom2 / denom1
    num1 = (t_max - temp_c) / denom1
    num2 = (temp_c - t_min) / denom2
    
    try:
        val = num1 * (num2 ** exponent)
        return float(max(0.0, min(1.0, val)))
    except (ValueError, ZeroDivisionError, OverflowError):
        # Fallback to standard triangular distribution
        if temp_c < t_opt:
            return float(max(0.0, (temp_c - t_min) / denom2))
        else:
            return float(max(0.0, (t_max - temp_c) / denom1))


def calculate_moisture_suitability(
    moisture_fc_pct: float,
    fc_min: float,
    fc_max: float,
    is_oomycete: bool = False
) -> Tuple[float, float]:
    """
    Calculates soil moisture suitability index [0.0, 1.0] and Oomycete zoospore motility index [0.0, 1.0].
    
    For Oomycetes, water film continuity above 75-80% FC activates flagellated zoospore chemotaxis,
    inducing an exponential infectivity leap modeled via a sigmoidal saturation function:
        Motility = 1 / (1 + exp(-0.16 * (Moisture% - 78)))
    """
    bounded_moisture = max(0.0, min(100.0, moisture_fc_pct))
    
    # Zoospore motility calculation
    if is_oomycete and bounded_moisture >= 60.0:
        zoospore_motility = 1.0 / (1.0 + math.exp(-0.16 * (bounded_moisture - 78.0)))
    else:
        zoospore_motility = 0.0
        
    # Moisture response curve
    if bounded_moisture < (fc_min - 20.0):
        moisture_suitability = 0.05
    elif bounded_moisture < fc_min:
        # Ramp up
        moisture_suitability = 0.05 + 0.95 * ((bounded_moisture - (fc_min - 20.0)) / 20.0)
    elif fc_min <= bounded_moisture <= fc_max:
        moisture_suitability = 1.0
    else:
        # For non-oomycetes, saturation (>90% FC) leads to anaerobiosis which suppresses fungal growth
        if is_oomycete:
            moisture_suitability = 1.0
        else:
            drop = (bounded_moisture - fc_max) / max(1.0, (100.0 - fc_max))
            moisture_suitability = max(0.2, 1.0 - 0.7 * drop)
            
    # Zoospore boost for Oomycetes at near-saturation
    if is_oomycete and zoospore_motility > 0.5:
        moisture_suitability = min(1.0, moisture_suitability * (1.0 + 0.35 * zoospore_motility))
        
    return round(float(min(1.0, max(0.0, moisture_suitability))), 3), round(float(zoospore_motility), 3)


def calculate_soilborne_pathogen_risk(
    soil_temp_c: float,
    soil_moisture_fc_pct: float,
    soil_texture: str = "Clay Loam",
    compaction_level: str = "Moderate",
    crop: str = "Tomato",
    seed_depth_cm: float = 3.5
) -> Dict[str, Any]:
    """
    Computes bio-epidemiological infection hazard scores (0-100) across all soilborne pathogens
    based on real-time soil physical and hydrothermal telemetry.
    
    Args:
        soil_temp_c: Current soil temperature at seeding zone depth (5-10 cm) in °C.
        soil_moisture_fc_pct: Soil moisture expressed as % of Field Capacity (0 to 100%).
        soil_texture: Soil textural class ("Clay", "Clay Loam", "Silt Loam", "Sandy Loam", "Sand").
        compaction_level: Physical soil compaction state ("Low / Loose", "Moderate", "Severe / Plow-Pan").
        crop: Host plant or intended transplant crop.
        seed_depth_cm: Planting depth in cm (excessive depth > 4 cm exacerbates emergence lag).
        
    Returns:
        Structured evaluation dictionary containing overall threat level, primary active pathogen,
        per-pathogen risk breakdown, emergence delay metrics, and agronomic diagnostic warnings.
    """
    texture_info = SOIL_TEXTURE_MODIFIERS.get(soil_texture, SOIL_TEXTURE_MODIFIERS["Clay Loam"])
    compaction_info = SOIL_COMPACTION_FACTORS.get(compaction_level, SOIL_COMPACTION_FACTORS["Moderate"])
    
    # Emergence delay calculation:
    # Cold temperatures (<15°C) and deep sowing (>3cm) combined with compaction delay seedling emergence
    base_emergence_days = 5.0
    temp_delay = max(0.0, (18.0 - soil_temp_c) * 0.8) if soil_temp_c < 18.0 else 0.0
    depth_delay = max(0.0, (seed_depth_cm - 2.5) * 1.5)
    compaction_delay = compaction_info["emergence_delay_days"]
    total_emergence_days = base_emergence_days + temp_delay + depth_delay + compaction_delay
    
    # Pre-emergence damping-off susceptibility multiplier
    # Each extra day in soil under saturated/cold conditions exposes delicate hypocotyl to inoculum
    exposure_vulnerability_multiplier = min(3.0, 1.0 + (total_emergence_days - 5.0) * 0.15)
    
    pathogen_results: Dict[str, Any] = {}
    highest_risk_score = 0.0
    primary_threat_key = "Pythium_ultimum"
    
    for path_key, path_data in SOILBORNE_PATHOGENS_DATABASE.items():
        is_host = any(c.lower() in crop.lower() or crop.lower() in c.lower() for c in path_data["target_crops"])
        host_factor = 1.0 if is_host else 0.45  # Pathogen can still cause non-specific damping off
        
        # 1. Thermal suitability
        t_suit = calculate_thermal_suitability(
            soil_temp_c,
            path_data["temp_min_c"],
            path_data["temp_opt_c"],
            path_data["temp_max_c"]
        )
        
        # 2. Moisture suitability & zoospore motility
        m_suit, zoospore_motility = calculate_moisture_suitability(
            soil_moisture_fc_pct,
            path_data["optimal_moisture_fc_min"],
            path_data["optimal_moisture_fc_max"],
            is_oomycete=path_data["motile_zoospores"]
        )
        
        # 3. Edaphic texture & drainage modifiers
        if path_data["motile_zoospores"]:
            texture_mod = texture_info["oomycete_risk_boost"]
        elif "Rhizoctonia" in path_key or "Macrophomina" in path_key:
            texture_mod = texture_info["rhizoctonia_risk_boost"]
        else:
            texture_mod = 1.0
            
        # 4. Compaction multiplier
        compaction_mod = compaction_info["compaction_multiplier"]
        
        # Hydro-Thermal Infection Index (0.0 to 1.0)
        hti = (t_suit * 0.45) + (m_suit * 0.55)
        # Combine with edaphic and emergence lag factors
        raw_score = hti * host_factor * texture_mod * compaction_mod * exposure_vulnerability_multiplier * 100.0
        risk_score = round(float(min(100.0, max(0.0, raw_score))), 1)
        
        if risk_score > highest_risk_score:
            highest_risk_score = risk_score
            primary_threat_key = path_key
            
        # Determine pathogen tier
        if risk_score >= 75.0:
            tier = "CRITICAL"
        elif risk_score >= 50.0:
            tier = "HIGH"
        elif risk_score >= 25.0:
            tier = "MODERATE"
        else:
            tier = "LOW"
            
        pathogen_results[path_key] = {
            "common_name": path_data["common_name"],
            "organism_class": path_data["organism_class"],
            "risk_score": risk_score,
            "risk_tier": tier,
            "thermal_suitability": round(t_suit, 2),
            "moisture_suitability": round(m_suit, 2),
            "zoospore_motility_index": zoospore_motility,
            "is_preferred_host": is_host,
            "primary_symptoms": path_data["primary_symptoms"],
            "frac_chemical_group": path_data["frac_chemical_group"],
            "biological_antagonists": path_data["biological_antagonists"]
        }
        
    # Overall soilborne threat classification
    if highest_risk_score >= 75.0:
        overall_tier = "CRITICAL SEEDLING HAZARD"
        overall_color = "#dc2626"
    elif highest_risk_score >= 50.0:
        overall_tier = "HIGH DAMPING-OFF RISK"
        overall_color = "#ea580c"
    elif highest_risk_score >= 25.0:
        overall_tier = "MODERATE FIELD HAZARD"
        overall_color = "#d97706"
    else:
        overall_tier = "LOW / SUPPRESSIVE ENVIRONMENT"
        overall_color = "#16a34a"
        
    return {
        "overall_risk_score": highest_risk_score,
        "overall_tier": overall_tier,
        "overall_color": overall_color,
        "primary_pathogen_key": primary_threat_key,
        "primary_pathogen": pathogen_results[primary_threat_key],
        "projected_emergence_days": round(total_emergence_days, 1),
        "emergence_delay_penalty": round(total_emergence_days - base_emergence_days, 1),
        "exposure_vulnerability_index": round(exposure_vulnerability_multiplier, 2),
        "soil_temperature_c": soil_temp_c,
        "soil_moisture_fc_pct": soil_moisture_fc_pct,
        "soil_texture": soil_texture,
        "compaction_level": compaction_level,
        "crop": crop,
        "pathogen_breakdown": pathogen_results
    }

