"""
PlantGuard AI - Agricultural IoT Telemetry & Vapor Pressure Deficit (VPD) Engine
Computes psychrometric Vapor Pressure Deficit (VPD) to model microclimate spore
germination risk and process greenhouse/field IoT sensor streams.
"""

import math
import datetime

def calculate_dew_point(temp_c: float, relative_humidity: float) -> float:
    """
    Computes the thermodynamic Dew Point temperature (in °C) using the Magnus-Tetens formula.
    Dew point is the temperature at which air moisture reaches 100% saturation and condenses.
    """
    rh_clamped = min(100.0, max(1.0, relative_humidity))
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh_clamped / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return round(dew_point, 2)


def calculate_absolute_humidity(temp_c: float, relative_humidity: float) -> float:
    """
    Computes volumetric Absolute Humidity in grams of water vapor per cubic meter of air (g/m³).
    """
    rh_clamped = min(100.0, max(0.0, relative_humidity))
    vp_sat_pa = 610.78 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    vp_act_pa = vp_sat_pa * (rh_clamped / 100.0)
    abs_humidity = (2.1674 * vp_act_pa) / (temp_c + 273.15)
    return round(abs_humidity, 2)


def calculate_vpd(temp_c: float, relative_humidity: float) -> dict:
    """
    Computes Saturated Vapor Pressure (VPsat), Actual Vapor Pressure (VPact),
    Vapor Pressure Deficit (VPD in kPa), Dew Point, and Condensation Hazard.
    """
    # Tetens formula for saturation vapor pressure over water (in kPa)
    vp_sat = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    
    # Actual vapor pressure
    rh_clamped = min(100.0, max(0.0, relative_humidity))
    vp_act = vp_sat * (rh_clamped / 100.0)
    
    # Vapor Pressure Deficit
    vpd = max(0.0, vp_sat - vp_act)
    dew_point = calculate_dew_point(temp_c, rh_clamped)
    abs_humidity = calculate_absolute_humidity(temp_c, rh_clamped)
    
    # Condensation Margin (°C until foliar condensation)
    condensation_margin = round(temp_c - dew_point, 2)
    condensation_risk = condensation_margin <= 1.8
    
    return {
        "temp_c": round(temp_c, 1),
        "humidity": round(rh_clamped, 1),
        "vp_sat": round(vp_sat, 3),
        "vp_act": round(vp_act, 3),
        "vpd_kpa": round(vpd, 2),
        "dew_point_c": dew_point,
        "abs_humidity_gm3": abs_humidity,
        "condensation_margin_c": condensation_margin,
        "condensation_risk": condensation_risk
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


def simulate_multizone_farm_telemetry() -> dict:
    """
    Simulates multi-sensor spatial microclimates across distinct agricultural zones.
    Computes dew point margin, psychrometric condensation risk, and zone-level hazard.
    """
    zones = [
        {
            "zone_id": "ZONE-A",
            "zone_name": "Closed Polyhouse (Propagation)",
            "crop_type": "Tomato Seedlings",
            "temp_c": 19.0,
            "humidity_pct": 92.0,
            "soil_moisture_pct": 42.0,
            "light_lux": 15000
        },
        {
            "zone_id": "ZONE-B",
            "zone_name": "High-Tunnel Hydroponics",
            "crop_type": "Bell Pepper & Eggplant",
            "temp_c": 24.5,
            "humidity_pct": 74.0,
            "soil_moisture_pct": 34.0,
            "light_lux": 32000
        },
        {
            "zone_id": "ZONE-C",
            "zone_name": "South Orchard Canopy",
            "crop_type": "Honeycrisp Apple Trees",
            "temp_c": 21.0,
            "humidity_pct": 65.0,
            "soil_moisture_pct": 28.0,
            "light_lux": 48000
        },
        {
            "zone_id": "ZONE-D",
            "zone_name": "Open Field West",
            "crop_type": "Sweet Corn & Potatoes",
            "temp_c": 27.5,
            "humidity_pct": 46.0,
            "soil_moisture_pct": 22.0,
            "light_lux": 65000
        }
    ]
    
    zone_results = []
    high_risk_count = 0
    
    for z in zones:
        vpd_data = calculate_vpd(z["temp_c"], z["humidity_pct"])
        hazard = evaluate_vpd_disease_hazard(vpd_data["vpd_kpa"])
        
        is_critical = vpd_data["condensation_risk"] or "CRITICAL" in hazard["hazard_level"]
        if is_critical:
            high_risk_count += 1
            
        zone_results.append({
            "zone_id": z["zone_id"],
            "zone_name": z["zone_name"],
            "crop": z["crop_type"],
            "temp_c": z["temp_c"],
            "humidity_pct": z["humidity_pct"],
            "dew_point_c": vpd_data["dew_point_c"],
            "condensation_margin_c": vpd_data["condensation_margin_c"],
            "condensation_risk": vpd_data["condensation_risk"],
            "vpd_kpa": vpd_data["vpd_kpa"],
            "abs_humidity_gm3": vpd_data["abs_humidity_gm3"],
            "soil_moisture_pct": z["soil_moisture_pct"],
            "light_lux": z["light_lux"],
            "hazard_level": hazard["hazard_level"],
            "hazard_color": hazard["hazard_color"],
            "hazard_badge": hazard["hazard_badge"],
            "action": hazard["field_action"]
        })
        
    farm_status = "CRITICAL ALERT" if high_risk_count >= 2 else ("ELEVATED RISK" if high_risk_count == 1 else "STABLE / RESILIENT")
    farm_status_color = "#ef4444" if high_risk_count >= 2 else ("#f59e0b" if high_risk_count == 1 else "#10b981")
    
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_zones": len(zones),
        "high_risk_zones": high_risk_count,
        "farm_status": farm_status,
        "farm_status_color": farm_status_color,
        "zones": zone_results
    }
