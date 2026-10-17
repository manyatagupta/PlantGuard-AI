"""
PlantGuard AI - Agricultural IoT Telemetry & Vapor Pressure Deficit (VPD) Engine
Computes psychrometric Vapor Pressure Deficit (VPD) to model microclimate spore
germination risk and process greenhouse/field IoT sensor streams.
"""

import math
import datetime

def calculate_vpd(temp_c: float, relative_humidity: float) -> dict:
    """
    Computes Saturated Vapor Pressure (VPsat), Actual Vapor Pressure (VPact),
    and Vapor Pressure Deficit (VPD in kPa) using the Tetens psychrometric equation.
    """
    # Tetens formula for saturation vapor pressure over water (in kPa)
    vp_sat = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    
    # Actual vapor pressure
    rh_clamped = min(100.0, max(0.0, relative_humidity))
    vp_act = vp_sat * (rh_clamped / 100.0)
    
    # Vapor Pressure Deficit
    vpd = max(0.0, vp_sat - vp_act)
    
    return {
        "temp_c": round(temp_c, 1),
        "humidity": round(rh_clamped, 1),
        "vp_sat": round(vp_sat, 3),
        "vp_act": round(vp_act, 3),
        "vpd_kpa": round(vpd, 2)
    }

def evaluate_vpd_disease_hazard(vpd_kpa: float, hours_in_danger_zone: float = 4.0) -> dict:
    """
    Evaluates microclimate disease hazard based on continuous canopy VPD levels.
    """
    if vpd_kpa < 0.40:
        level = "CRITICAL PATHOGEN GERMINATION RISK"
        color = "#ef4444"
        badge = "🔴 Fungal Hazard"
        summary = "Air is near 100% moisture saturation. Free moisture film is condensing on leaves, providing optimal conditions for fungal zoospore motility and germination."
        target_pathogens = ["Botrytis (Grey Mold)", "Late Blight", "Downy Mildew", "Powdery Mildew"]
        action = "Activate greenhouse exhaust fans / horizontal airflow; cease overhead irrigation immediately."
    elif vpd_kpa < 0.80:
        level = "ELEVATED HUMIDITY / HIGH SPREAD ZONE"
        color = "#f59e0b"
        badge = "🟡 Moderate Spore Risk"
        summary = "Sub-optimal air movement and high relative humidity. Spores can remain viable for extended periods on lower shaded canopy leaves."
        target_pathogens = ["Early Blight", "Septoria Leaf Spot", "Anthracnose"]
        action = "Prune lower sucker leaves to increase cross-canopy ventilation."
    elif vpd_kpa <= 1.25:
        level = "OPTIMAL HORTICULTURAL EQUILIBRIUM"
        color = "#10b981"
        badge = "🟢 Disease Resilient"
        summary = "Ideal transpiration balance. Plant stomata are fully open, foliar surfaces remain dry, and cuticle defense barriers are active."
        target_pathogens = ["None (Resilient microclimate)"]
        action = "Maintain current ventilation and irrigation cycle."
    elif vpd_kpa <= 1.60:
        level = "MODERATE TRANSPIRATION STRESS"
        color = "#38bdf8"
        badge = "🔵 Dry Canopy"
        summary = "Foliage is completely dry. Fungal spore germination is inhibited, but rapid transpiration may cause slight mid-day leaf curl."
        target_pathogens = ["Spider Mites (favors hot/dry)"]
        action = "Monitor soil root moisture; ensure adequate drip line pressure."
    else:
        level = "SEVERE MOISTURE DEFICIT / PLANT STRESS"
        color = "#f97316"
        badge = "🟠 Abiotic Stress Hazard"
        summary = "Excessive atmospheric dryness forcing stomatal closure. Severe risk of blossom end rot and foliar margin desiccation."
        target_pathogens = ["Calcium deficiency tip-burn", "Heat stress scorch"]
        action = "Deploy misting / shade cloths and increase root irrigation volume."

    return {
        "vpd_kpa": vpd_kpa,
        "hazard_level": level,
        "hazard_color": color,
        "hazard_badge": badge,
        "summary": summary,
        "target_pathogens": target_pathogens,
        "field_action": action,
        "hours_in_zone": hours_in_danger_zone
    }

def generate_mock_iot_telemetry_stream(node_name: str = "Node #04 - South Greenhouse", hours: int = 8) -> list:
    """
    Generates time-series hourly IoT probe telemetry for testing microclimate sensors.
    """
    readings = []
    base_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
    
    # Simulated diurnal cycle: cooler/humid at night -> warmer/drier mid-day
    profile = [
        (16.5, 94.0, 35.0, 0),    # Night - high humidity, very low VPD
        (17.0, 92.0, 36.0, 50),   # Dawn
        (19.5, 85.0, 34.0, 250),  # Early Morning
        (22.0, 72.0, 32.0, 550),  # Mid Morning
        (24.5, 62.0, 30.0, 780),  # Noon
        (25.8, 55.0, 28.0, 850),  # Afternoon
        (23.5, 68.0, 29.0, 400),  # Late Afternoon
        (20.0, 82.0, 31.0, 100)   # Evening
    ]
    
    for i in range(min(hours, len(profile))):
        t_stamp = base_time + datetime.timedelta(hours=i)
        temp, rh, soil_moist, solar = profile[i]
        vpd_calc = calculate_vpd(temp, rh)
        hazard = evaluate_vpd_disease_hazard(vpd_calc["vpd_kpa"])
        
        readings.append({
            "timestamp": t_stamp.strftime("%H:%M"),
            "temperature_c": temp,
            "humidity_pct": rh,
            "soil_moisture_pct": soil_moist,
            "solar_radiation_wm2": solar,
            "vpd_kpa": vpd_calc["vpd_kpa"],
            "hazard_level": hazard["hazard_level"],
            "hazard_color": hazard["hazard_color"]
        })
        
    return readings
