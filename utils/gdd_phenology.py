"""
PlantGuard AI - Growing Degree Day (GDD) & Pathogen Phenology Biofix Forecaster
Models physiological thermal accumulation (thermal time) for crops, fungal pathogens,
and insect vectors based on university IPM models (Gadoury & MacHardy, FAST, Gubler-Thomas).
"""

import math
import datetime

# Comprehensive Pathogen & Vector Thermal Phenology Models
PATHOGEN_PHENOLOGY_MODELS = {
    "Apple Scab": {
        "scientific_name": "Venturia inaequalis",
        "crop": "Apple",
        "base_temp_c": 0.0,
        "cap_temp_c": 30.0,
        "biofix_event": "Green Tip (Bud Break)",
        "model_reference": "Gadoury & MacHardy Ascospore Maturation Model (Cornell/NEWA)",
        "milestones": [
            {
                "stage": "Dormant / Overwintering Pseudothecia",
                "gdd_min": 0.0,
                "gdd_max": 49.9,
                "ascospore_maturation_pct": "0 - 1%",
                "risk_level": "Low / Dormant",
                "risk_color": "#10b981",
                "action_advisory": "Perform orchard sanitation (flail mowing of leaf litter or 5% urea spray) to reduce primary ascospore inoculum."
            },
            {
                "stage": "Early Ascospore Ripening (First Discharge Window)",
                "gdd_min": 50.0,
                "gdd_max": 119.9,
                "ascospore_maturation_pct": "2 - 10%",
                "risk_level": "Moderate Watch",
                "risk_color": "#f59e0b",
                "action_advisory": "First mature ascospores ready for rain-triggered discharge. Apply protectant copper or mancozeb prior to predicted rain events."
            },
            {
                "stage": "Accelerated Ascospore Discharge (Peak Inoculum Window)",
                "gdd_min": 120.0,
                "gdd_max": 279.9,
                "ascospore_maturation_pct": "11 - 70%",
                "risk_level": "Critical Primary Alert",
                "risk_color": "#ef4444",
                "action_advisory": "High epidemic danger. Every rain >0.2 mm will discharge millions of spores. Maintain 7-day protective barrier or use kickback DMI/QoI fungicide within 72 hours of rain."
            },
            {
                "stage": "Late Primary Ascospore Depletion",
                "gdd_min": 280.0,
                "gdd_max": 449.9,
                "ascospore_maturation_pct": "71 - 95%",
                "risk_level": "Elevated Vigilance",
                "risk_color": "#f97316",
                "action_advisory": "Primary ascospore supply is nearly exhausted. Scout fruit spurs and terminal cluster leaves for olive-green primary lesions."
            },
            {
                "stage": "Primary Season End / Conidial Cycling",
                "gdd_min": 450.0,
                "gdd_max": 9999.0,
                "ascospore_maturation_pct": ">95% Depleted",
                "risk_level": "Secondary Conidial Phase",
                "risk_color": "#8b5cf6",
                "action_advisory": "Overwintered asci empty. If primary scab was controlled, fungicide intervals can be extended. If lesions are present, secondary conidia will cycle all summer."
            }
        ]
    },
    "Tomato Early Blight": {
        "scientific_name": "Alternaria solani",
        "crop": "Tomato",
        "base_temp_c": 7.2,
        "cap_temp_c": 32.0,
        "biofix_event": "Field Transplanting or Seedling Emergence",
        "model_reference": "FAST Model (Forecaster of Alternaria solani on Tomato, Penn State)",
        "milestones": [
            {
                "stage": "Vegetative Establishment / Low Inoculum",
                "gdd_min": 0.0,
                "gdd_max": 149.9,
                "ascospore_maturation_pct": "Conidial Spores Latent",
                "risk_level": "Low / Preventive",
                "risk_color": "#10b981",
                "action_advisory": "Maintain adequate mulch barrier to prevent rain splashing of soil-borne mycelium onto lower foliage."
            },
            {
                "stage": "Canopy Closure & Microclimate Humidity",
                "gdd_min": 150.0,
                "gdd_max": 299.9,
                "ascospore_maturation_pct": "Lower Foliar Colonization",
                "risk_level": "Moderate Watch",
                "risk_color": "#f59e0b",
                "action_advisory": "Canopy is dense and lower leaves remain wet longer. Prune bottom 12 inches of suckers to improve airflow."
            },
            {
                "stage": "FAST Critical Action Threshold (300 GDD Trigger)",
                "gdd_min": 300.0,
                "gdd_max": 449.9,
                "ascospore_maturation_pct": "Primary Inoculation Active",
                "risk_level": "High Treatment Trigger",
                "risk_color": "#ef4444",
                "action_advisory": "Thermal accumulation reaches the 300 GDD threshold. University IPM protocols mandate the FIRST protective fungicide application (Chlorothalonil, Mancozeb, or Copper)."
            },
            {
                "stage": "Secondary Polycyclic Generation",
                "gdd_min": 450.0,
                "gdd_max": 9999.0,
                "ascospore_maturation_pct": "Repeating Conidial Cycles",
                "risk_level": "Active Disease Cycling",
                "risk_color": "#dc2626",
                "action_advisory": "Alternaria cycles every 7-10 days under warm temperatures. Alternate FRAC 3, 7, and 11 chemistry to avoid target site resistance."
            }
        ]
    },
    "Potato Late Blight": {
        "scientific_name": "Phytophthora infestans",
        "crop": "Potato",
        "base_temp_c": 7.0,
        "cap_temp_c": 28.0,
        "biofix_event": "50% Crop Emergence",
        "model_reference": "SimCast Blight Unit & Thermal Accumulation Model",
        "milestones": [
            {
                "stage": "Seed Tuber Germination / Latent Sporangia",
                "gdd_min": 0.0,
                "gdd_max": 119.9,
                "ascospore_maturation_pct": "Mycelium in Seed Pieces",
                "risk_level": "Low Watch",
                "risk_color": "#10b981",
                "action_advisory": "Scout cull piles and volunteer potatoes within 1 mile of field. Destroy volunteers."
            },
            {
                "stage": "Initial Sporulation Hazard",
                "gdd_min": 120.0,
                "gdd_max": 249.9,
                "ascospore_maturation_pct": "Stem Infection Possible",
                "risk_level": "Moderate Alert",
                "risk_color": "#f59e0b",
                "action_advisory": "If relative humidity exceeds 90% for 10+ hours, sporangia will form. Initiate preventive contact spray."
            },
            {
                "stage": "Rapid Epidemic Expansion Threshold",
                "gdd_min": 250.0,
                "gdd_max": 9999.0,
                "ascospore_maturation_pct": "Explosive Zoospore Discharge",
                "risk_level": "Critical Quarantine Alert",
                "risk_color": "#ef4444",
                "action_advisory": "Zoospore swimming in foliar moisture films. Apply translaminar/systemic oomycete fungicides (Mandipropamid, Cymoxanil, or Fluopicolide)."
            }
        ]
    },
    "Grape Powdery Mildew": {
        "scientific_name": "Erysiphe necator",
        "crop": "Grape",
        "base_temp_c": 6.0,
        "cap_temp_c": 35.0,
        "biofix_event": "Bud Break (1-Inch Shoot Growth)",
        "model_reference": "UC Davis Gubler-Thomas Powdery Mildew Index (PMI)",
        "milestones": [
            {
                "stage": "Ascospore Release (Cleistothecia Dehiscence)",
                "gdd_min": 0.0,
                "gdd_max": 139.9,
                "ascospore_maturation_pct": "Spring Ascospores",
                "risk_level": "Early Season Protection",
                "risk_color": "#10b981",
                "action_advisory": "Ascospores released during spring rain (>=2.5 mm). Apply wettable sulfur at 2-3 inch shoot growth."
            },
            {
                "stage": "Primary Colony Establishment",
                "gdd_min": 140.0,
                "gdd_max": 299.9,
                "ascospore_maturation_pct": "Conidiophores Emerging",
                "risk_level": "Moderate Alert",
                "risk_color": "#f59e0b",
                "action_advisory": "First web-like patches on underside of basal leaves. Maintain tight 7-10 day spray coverage."
            },
            {
                "stage": "Explosive Conidial Cycling (Gubler Index Peak)",
                "gdd_min": 300.0,
                "gdd_max": 9999.0,
                "ascospore_maturation_pct": "Continuous Clonal Reproduction",
                "risk_level": "Severe Canopy Hazard",
                "risk_color": "#ef4444",
                "action_advisory": "Conidia reproduce every 5 days at 21-27°C. Keep berry clusters protected through veraison."
            }
        ]
    },
    "Tomato Whitefly Vector": {
        "scientific_name": "Bemisia tabaci",
        "crop": "Tomato / Bell Pepper",
        "base_temp_c": 10.0,
        "cap_temp_c": 38.0,
        "biofix_event": "First Spring Whitefly Trapping or Transplant",
        "model_reference": "Degree-Day Generational Model (UC IPM)",
        "milestones": [
            {
                "stage": "Egg Incubation & Crawler Hatch",
                "gdd_min": 0.0,
                "gdd_max": 79.9,
                "ascospore_maturation_pct": "Eggs / 1st Instar Crawlers",
                "risk_level": "Low / Monitor",
                "risk_color": "#10b981",
                "action_advisory": "Yellow sticky cards placed at canopy height. Release Encarsia formosa parasitic wasps in greenhouse."
            },
            {
                "stage": "Sessile Nymphal Feeding",
                "gdd_min": 80.0,
                "gdd_max": 189.9,
                "ascospore_maturation_pct": "2nd & 3rd Instar Scales",
                "risk_level": "Moderate Alert",
                "risk_color": "#f59e0b",
                "action_advisory": "Nymphs secrete honeydew encouraging sooty mold. Apply insecticidal soap or neem oil targeting leaf undersides."
            },
            {
                "stage": "1st Generation Adult Flight & Virus Transmission",
                "gdd_min": 190.0,
                "gdd_max": 310.0,
                "ascospore_maturation_pct": "Adult Emergence (Peak Vector Risk)",
                "risk_level": "Critical Viral Vector Alert",
                "risk_color": "#ef4444",
                "action_advisory": "Adults emerge and actively transmit Tomato Yellow Leaf Curl Virus (TYLCV). Apply systemic flupyradifurone or spiromesifen."
            },
            {
                "stage": "Multivoltine Overlapping Generations",
                "gdd_min": 310.1,
                "gdd_max": 9999.0,
                "ascospore_maturation_pct": "2nd+ Generation Overlap",
                "risk_level": "Severe Pest Pressure",
                "risk_color": "#dc2626",
                "action_advisory": "Continuous generation overlap. Rotate IRAC insecticide modes of action to prevent pyrethroid resistance."
            }
        ]
    }
}


def calculate_daily_gdd(
    t_max: float,
    t_min: float,
    base_temp: float = 10.0,
    cap_temp: float = 30.0,
    method: str = "average"
) -> float:
    """
    Computes single-day thermal heat accumulation (Growing Degree Days).
    Supports standard average method and upper thermal cutoff.
    """
    # Clamp upper temperature at physiological threshold
    adj_max = min(t_max, cap_temp) if cap_temp else t_max
    adj_min = max(t_min, base_temp) if method == "modified" else t_min
    
    if method == "average":
        mean_temp = (adj_max + t_min) / 2.0
        gdd = max(0.0, mean_temp - base_temp)
    elif method == "modified":
        mean_temp = (adj_max + adj_min) / 2.0
        gdd = max(0.0, mean_temp - base_temp)
    elif method == "sine":
        # Simplified sine wave approximation
        if t_min >= base_temp:
            mean_temp = (adj_max + t_min) / 2.0
            gdd = max(0.0, mean_temp - base_temp)
        elif adj_max <= base_temp:
            gdd = 0.0
        else:
            w = (adj_max - t_min) / 2.0
            avg = (adj_max + t_min) / 2.0
            theta = math.asin(min(1.0, max(-1.0, (base_temp - avg) / w)))
            gdd = max(0.0, ((avg - base_temp) * (math.pi / 2.0 - theta) + w * math.cos(theta)) / math.pi)
    else:
        mean_temp = (adj_max + t_min) / 2.0
        gdd = max(0.0, mean_temp - base_temp)
        
    return round(float(gdd), 2)


def calculate_cumulative_gdd_series(
    daily_records: list,
    base_temp: float = 10.0,
    cap_temp: float = 30.0,
    method: str = "average"
) -> dict:
    """
    Calculates cumulative GDD curve from a sequence of daily temperature records.
    Each record: {'date': 'YYYY-MM-DD', 't_max': float, 't_min': float}
    """
    running_gdd = 0.0
    series = []
    
    for r in daily_records:
        t_max = r.get("t_max", 20.0)
        t_min = r.get("t_min", 10.0)
        date_str = r.get("date", "N/A")
        
        daily_val = calculate_daily_gdd(t_max, t_min, base_temp, cap_temp, method)
        running_gdd += daily_val
        
        series.append({
            "date": date_str,
            "t_max": t_max,
            "t_min": t_min,
            "daily_gdd": daily_val,
            "cumulative_gdd": round(running_gdd, 2)
        })
        
    return {
        "total_cumulative_gdd": round(running_gdd, 2),
        "days_accumulated": len(daily_records),
        "average_daily_gdd": round(running_gdd / max(1, len(daily_records)), 2),
        "series": series
    }


def evaluate_pathogen_phenology(
    pathogen_or_disease: str,
    cumulative_gdd: float,
    biofix_date: str = None
) -> dict:
    """
    Evaluates current biological life cycle stage, ascospore maturation percentage,
    and timed IPM fungicide/insecticide intervention advisory based on cumulative GDD.
    """
    # Match model
    matched_key = None
    for k in PATHOGEN_PHENOLOGY_MODELS:
        if k.lower() in pathogen_or_disease.lower() or pathogen_or_disease.lower() in k.lower():
            matched_key = k
            break
            
    if not matched_key:
        # Fallback to closest crop or default Tomato Early Blight
        if "apple" in pathogen_or_disease.lower():
            matched_key = "Apple Scab"
        elif "potato" in pathogen_or_disease.lower():
            matched_key = "Potato Late Blight"
        elif "grape" in pathogen_or_disease.lower():
            matched_key = "Grape Powdery Mildew"
        elif "whitefly" in pathogen_or_disease.lower() or "virus" in pathogen_or_disease.lower():
            matched_key = "Tomato Whitefly Vector"
        else:
            matched_key = "Tomato Early Blight"
            
    model_cfg = PATHOGEN_PHENOLOGY_MODELS[matched_key]
    milestones = model_cfg["milestones"]
    
    current_milestone = milestones[-1]
    for m in milestones:
        if m["gdd_min"] <= cumulative_gdd <= m["gdd_max"]:
            current_milestone = m
            break
            
    # Calculate progress toward next stage
    next_stage_gdd = None
    gdd_to_next_stage = 0.0
    progress_pct = 100.0
    
    for i, m in enumerate(milestones):
        if m == current_milestone:
            if i + 1 < len(milestones):
                next_stage_gdd = milestones[i + 1]["gdd_min"]
                gdd_span = next_stage_gdd - m["gdd_min"]
                gdd_elapsed = cumulative_gdd - m["gdd_min"]
                progress_pct = min(100.0, max(0.0, round((gdd_elapsed / max(1.0, gdd_span)) * 100.0, 1)))
                gdd_to_next_stage = round(next_stage_gdd - cumulative_gdd, 1)
            else:
                progress_pct = 100.0
                gdd_to_next_stage = 0.0
            break
            
    return {
        "model_key": matched_key,
        "disease_name": matched_key,
        "scientific_name": model_cfg["scientific_name"],
        "crop": model_cfg["crop"],
        "base_temp_c": model_cfg["base_temp_c"],
        "cap_temp_c": model_cfg["cap_temp_c"],
        "biofix_event": model_cfg["biofix_event"],
        "biofix_date": biofix_date or "Spring Bud Break / Transplant",
        "model_reference": model_cfg["model_reference"],
        "cumulative_gdd": round(cumulative_gdd, 1),
        "stage": current_milestone["stage"],
        "ascospore_maturation_pct": current_milestone["ascospore_maturation_pct"],
        "risk_level": current_milestone["risk_level"],
        "risk_color": current_milestone["risk_color"],
        "action_advisory": current_milestone["action_advisory"],
        "stage_progress_pct": progress_pct,
        "gdd_to_next_stage": gdd_to_next_stage,
        "all_milestones": milestones
    }


def estimate_upcoming_milestones(
    pathogen_or_disease: str,
    current_gdd: float,
    projected_daily_gdd: float = 12.0,
    start_date: datetime.date = None
) -> list:
    """
    Projects target calendar dates when future pathogen developmental thresholds will be crossed.
    """
    if start_date is None:
        start_date = datetime.date.today()
        
    eval_res = evaluate_pathogen_phenology(pathogen_or_disease, current_gdd)
    milestones = eval_res["all_milestones"]
    
    projections = []
    daily_rate = max(1.0, projected_daily_gdd)
    
    for m in milestones:
        target_gdd = m["gdd_min"]
        if target_gdd > current_gdd:
            diff_gdd = target_gdd - current_gdd
            days_needed = int(math.ceil(diff_gdd / daily_rate))
            proj_date = start_date + datetime.timedelta(days=days_needed)
            projections.append({
                "stage": m["stage"],
                "target_gdd": target_gdd,
                "days_remaining": days_needed,
                "projected_date": proj_date.strftime("%Y-%m-%d"),
                "risk_level": m["risk_level"],
                "risk_color": m["risk_color"],
                "advisory": m["action_advisory"]
            })
            
    return projections


def get_supported_phenology_models() -> list:
    """
    Returns list of all supported crop and pathogen thermal models.
    """
    return [
        {
            "disease_name": k,
            "crop": v["crop"],
            "base_temp_c": v["base_temp_c"],
            "biofix_event": v["biofix_event"],
            "model_reference": v["model_reference"]
        }
        for k, v in PATHOGEN_PHENOLOGY_MODELS.items()
    ]
