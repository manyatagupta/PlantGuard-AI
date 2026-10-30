"""
PlantGuard AI - Meteorological Disease Outbreak Risk Forecaster
Models fungal, bacterial, and pest vulnerability based on real-time microclimate conditions
(Temperature, Relative Humidity, Precipitation, Leaf Wetness, and Wind).
Uses established epidemiological criteria (e.g., Mills Scab Periods, Wallin/Smith Blight Units).
"""

import requests
import datetime
from utils.gdd_phenology import evaluate_pathogen_phenology, PATHOGEN_PHENOLOGY_MODELS

# Preset Climates for quick simulation
WEATHER_PRESETS = {
    "🌧️ Cool & Wet Spring (Late Blight & Scab Alert)": {
        "temp": 17.5,
        "humidity": 92.0,
        "rainfall": 8.0,
        "leaf_wetness": 11.0,
        "wind_speed": 14.0,
        "description": "High continuous moisture with cool temperatures favors water mold germination (Phytophthora) and Apple Scab ascospores."
    },
    "⛈️ Warm & Humid Monsoon (Bacterial Spot & Blight Surge)": {
        "temp": 28.0,
        "humidity": 88.0,
        "rainfall": 15.0,
        "leaf_wetness": 9.0,
        "wind_speed": 18.0,
        "description": "High heat, saturated air, and driving rain splash accelerate bacterial pathogens (Xanthomonas) and Early Blight."
    },
    "☀️ Scorching & Arid Heatwave (Spider Mite & Vector Outbreak)": {
        "temp": 34.0,
        "humidity": 32.0,
        "rainfall": 0.0,
        "leaf_wetness": 0.5,
        "wind_speed": 16.0,
        "description": "Low humidity and extreme heat accelerate Two-Spotted Spider Mite breeding cycles and Whitefly transmission vectors."
    },
    "🌫️ Mild, Overcast & Humid (Leaf Mold & Rust Spread)": {
        "temp": 21.5,
        "humidity": 84.0,
        "rainfall": 1.5,
        "leaf_wetness": 6.5,
        "wind_speed": 7.0,
        "description": "Prolonged high ambient humidity without heavy downpours encourages greenhouse leaf mold and fungal rust pustules."
    },
    "🌤️ Temperate, Sunny & Dry (Optimal Farm Safety)": {
        "temp": 22.0,
        "humidity": 48.0,
        "rainfall": 0.0,
        "leaf_wetness": 1.0,
        "wind_speed": 11.0,
        "description": "Dry canopy and moderate temperatures suppress foliar pathogen germination. Safe condition for maintenance."
    }
}

# Real-time weather fetcher via Open-Meteo (No API key needed)
def fetch_live_weather(city_name: str) -> dict:
    """
    Fetches real-time weather data for a given city using Open-Meteo APIs.
    Returns parsed dictionary of climate parameters or error information.
    """
    if not city_name or len(city_name.strip()) < 2:
        return {"success": False, "error": "Please enter a valid city name."}
        
    try:
        # Step 1: Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name.strip()}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=5)
        if geo_resp.status_code != 200:
            return {"success": False, "error": f"Geocoding service returned status {geo_resp.status_code}."}
            
        geo_data = geo_resp.json()
        if not geo_data.get("results"):
            return {"success": False, "error": f"City '{city_name}' could not be located."}
            
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        city_label = f"{location.get('name')}, {location.get('admin1', '')} {location.get('country', '')}".strip()
        
        # Step 2: Current Weather & Hourly Forecast
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
            f"&hourly=precipitation,relative_humidity_2m&forecast_days=1"
        )
        w_resp = requests.get(weather_url, timeout=6)
        if w_resp.status_code != 200:
            return {"success": False, "error": "Unable to retrieve meteorological data."}
            
        w_data = w_resp.json()
        current = w_data.get("current", {})
        
        temp = float(current.get("temperature_2m", 22.0))
        humidity = float(current.get("relative_humidity_2m", 60.0))
        rain = float(current.get("precipitation", 0.0))
        wind = float(current.get("wind_speed_10m", 10.0))
        
        # Estimate leaf wetness duration (hours where rain > 0.1mm or humidity >= 90%)
        hourly = w_data.get("hourly", {})
        h_rains = hourly.get("precipitation", [])[:24]
        h_hums = hourly.get("relative_humidity_2m", [])[:24]
        wet_hours = 0.0
        for r, h in zip(h_rains, h_hums):
            if (r and r > 0.1) or (h and h >= 88.0):
                wet_hours += 1.0
                
        if rain > 0 and wet_hours < 3.0:
            wet_hours = max(wet_hours, 4.0)
            
        return {
            "success": True,
            "city": city_label,
            "lat": lat,
            "lon": lon,
            "temp": temp,
            "humidity": humidity,
            "rainfall": rain,
            "leaf_wetness": wet_hours,
            "wind_speed": wind,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        return {"success": False, "error": f"Weather query failed: {str(e)}"}

# Core Epidemiological Disease Risk Algorithm
def calculate_disease_outbreak_risk(
    temp: float,
    humidity: float,
    rainfall: float,
    leaf_wetness: float,
    wind_speed: float,
    crop_filter: str = "All Crops",
    cumulative_gdd: float = None
) -> dict:
    """
    Evaluates microclimate parameters against known plant pathogen criteria,
    optionally modulated by cumulative Growing Degree Day (GDD) phenological development.
    Returns risk breakdown per disease, overall threat index, and agronomic countermeasures.
    """
    diseases = [
        # --- POTATO & TOMATO LATE BLIGHT ---
        {
            "name": "Late Blight (Phytophthora infestans)",
            "crops": ["Potato", "Tomato"],
            "pathogen_type": "Oomycete / Water Mold",
            "eval": lambda t, h, r, w, s: (
                # Cool to mild (12-22°C), extremely high humidity >= 85%, long leaf wetness
                (1.0 if 12.0 <= t <= 22.0 else (0.5 if 10.0 <= t <= 25.0 else 0.1)) *
                (1.0 if h >= 90.0 else (0.7 if h >= 80.0 else (0.3 if h >= 70.0 else 0.05))) *
                (1.0 if (w >= 7.0 or r >= 3.0) else (0.6 if (w >= 4.0 or r > 0.2) else 0.15))
            ),
            "optimum": "15-21°C (60-70°F), Relative Humidity > 85%, Leaf Wetness > 6 hours",
            "advice": "Apply preventative translaminar copper/mancozeb before moisture events. Inspect lower canopy daily.",
            "emergency": "High risk of total canopy destruction within 48-72 hours if infected."
        },
        # --- APPLE SCAB ---
        {
            "name": "Apple Scab (Venturia inaequalis)",
            "crops": ["Apple"],
            "pathogen_type": "Foliar Fungus",
            "eval": lambda t, h, r, w, s: (
                # Mills period: 12-24°C + wetness hours
                (1.0 if 13.0 <= t <= 24.0 else (0.4 if 8.0 <= t <= 28.0 else 0.1)) *
                (1.0 if w >= 8.0 else (0.75 if w >= 5.0 else (0.35 if w >= 2.5 else 0.1))) *
                (1.0 if h >= 80.0 else 0.5)
            ),
            "optimum": "13-24°C (55-75°F) with 6+ hours of continuous leaf moisture",
            "advice": "Maintain protective fungicide coverage on new foliage. Avoid overhead sprinkler operation.",
            "emergency": "Primary ascospore release occurs rapidly during wet daylight hours."
        },
        # --- BACTERIAL SPOT ---
        {
            "name": "Bacterial Spot (Xanthomonas spp.)",
            "crops": ["Bell Pepper", "Tomato"],
            "pathogen_type": "Bacterial Infection",
            "eval": lambda t, h, r, w, s: (
                # Warm 24-32°C, high humidity, wind-blown rain
                (1.0 if 24.0 <= t <= 32.0 else (0.5 if 20.0 <= t <= 35.0 else 0.15)) *
                (1.0 if h >= 82.0 else (0.6 if h >= 70.0 else 0.2)) *
                (1.0 if (r >= 4.0 or s >= 15.0) else 0.5)
            ),
            "optimum": "24-30°C (75-86°F), High Humidity with driving rain splash or wind",
            "advice": "Do not enter fields or handle plants while wet. Apply fixed copper + biobactericide.",
            "emergency": "Rapid transmission across rows via wind-driven rain droplets."
        },
        # --- EARLY BLIGHT ---
        {
            "name": "Early Blight (Alternaria solani)",
            "crops": ["Potato", "Tomato"],
            "pathogen_type": "Foliar Fungus",
            "eval": lambda t, h, r, w, s: (
                # Warm 23-30°C, alternating wet/dry or high dew
                (1.0 if 22.0 <= t <= 30.0 else (0.5 if 18.0 <= t <= 33.0 else 0.2)) *
                (1.0 if h >= 75.0 else (0.6 if h >= 60.0 else 0.25)) *
                (1.0 if (w >= 4.0 or r >= 1.0) else 0.4)
            ),
            "optimum": "24-29°C (75-85°F), Warm humid days with heavy morning dews",
            "advice": "Prune lower leaves touching soil. Apply organic copper or chlorothalonil spray.",
            "emergency": "Causes progressive upward defoliation and yield loss."
        },
        # --- TWO-SPOTTED SPIDER MITES ---
        {
            "name": "Two-Spotted Spider Mites (Tetranychus urticae)",
            "crops": ["Tomato", "Bell Pepper", "Apple"],
            "pathogen_type": "Arachnid Pest",
            "eval": lambda t, h, r, w, s: (
                # Hot (>27°C) and DRY (h < 55%), no rainfall
                (1.0 if t >= 29.0 else (0.6 if t >= 25.0 else 0.15)) *
                (1.0 if h <= 45.0 else (0.6 if h <= 60.0 else 0.1)) *
                (1.0 if r == 0.0 else (0.4 if r <= 1.0 else 0.05))
            ),
            "optimum": "Hot (>28°C / 82°F) and Dry (RH < 50%), Dusty conditions with zero rain",
            "advice": "Hose down leaf undersides with high-pressure water. Apply insecticidal soap or neem oil.",
            "emergency": "Population expands exponentially in hot drought conditions; leaves bronze and shrivel."
        },
        # --- CORN BLIGHTS & RUST ---
        {
            "name": "Northern Leaf Blight & Gray Leaf Spot",
            "crops": ["Corn (Maize)"],
            "pathogen_type": "Foliar Fungus",
            "eval": lambda t, h, r, w, s: (
                (1.0 if 20.0 <= t <= 29.0 else (0.5 if 16.0 <= t <= 32.0 else 0.1)) *
                (1.0 if h >= 85.0 else (0.6 if h >= 70.0 else 0.2)) *
                (1.0 if w >= 6.0 else 0.4)
            ),
            "optimum": "20-28°C (68-82°F) with prolonged morning dew and humidity > 85%",
            "advice": "Scout ear leaves during tasseling. Ensure wide row spacing for maximum airflow.",
            "emergency": "Necrotic lesions can fuse and blight the entire photosynthetic canopy."
        },
        # --- TOMATO LEAF MOLD ---
        {
            "name": "Tomato Leaf Mold (Passalora fulva)",
            "crops": ["Tomato"],
            "pathogen_type": "Foliar Fungus",
            "eval": lambda t, h, r, w, s: (
                (1.0 if 20.0 <= t <= 26.0 else (0.4 if 16.0 <= t <= 30.0 else 0.1)) *
                (1.0 if h >= 85.0 else (0.6 if h >= 75.0 else 0.1)) *
                (1.0 if s <= 8.0 else 0.6) # favors stagnant air
            ),
            "optimum": "20-25°C (68-77°F), High relative humidity (>85%), Stagnant air",
            "advice": "Turn on ventilation fans in tunnels/greenhouses. Prune suckers to open canopy.",
            "emergency": "Olive-green velvet mold covers leaf undersides, reducing fruit sizing."
        },
        # --- APPLE BLACK ROT & RUST ---
        {
            "name": "Black Rot & Cedar Apple Rust",
            "crops": ["Apple"],
            "pathogen_type": "Foliar / Canker Fungus",
            "eval": lambda t, h, r, w, s: (
                (1.0 if 18.0 <= t <= 28.0 else (0.4 if 12.0 <= t <= 31.0 else 0.1)) *
                (1.0 if (h >= 75.0 or r >= 1.0) else 0.3) *
                (1.0 if w >= 4.0 else 0.5)
            ),
            "optimum": "20-27°C (68-80°F), Rain splash and humid conditions",
            "advice": "Remove dead wood and mummies. Apply sulfur or systemic triazole at bud break.",
            "emergency": "Causes frogeye leaf spots and trunk cankers."
        }
    ]
    
    analyzed_diseases = []
    
    for d in diseases:
        # Check crop match
        if crop_filter != "All Crops" and not any(crop_filter.lower() in c.lower() for c in d["crops"]):
            continue
            
        raw_score = d["eval"](temp, humidity, rainfall, leaf_wetness, wind_speed)
        # Scale score to 0 - 100%
        score = min(100.0, max(2.0, raw_score * 100.0))
        
        # Apply GDD phenological adjustment if cumulative thermal time is provided
        pheno_info = None
        if cumulative_gdd is not None:
            pheno_info = evaluate_pathogen_phenology(d["name"], cumulative_gdd)
            # Modulate risk by life-cycle susceptibility
            if "Critical" in pheno_info["risk_level"] or "High" in pheno_info["risk_level"]:
                score = min(100.0, score * 1.2 + 8.0)
            elif "Low" in pheno_info["risk_level"]:
                score = max(5.0, score * 0.75)
        
        if score >= 75.0:
            level = "Severe / Critical"
            badge_color = "#ef4444"
            icon = "🚨"
        elif score >= 50.0:
            level = "High"
            badge_color = "#f97316"
            icon = "🟠"
        elif score >= 28.0:
            level = "Moderate"
            badge_color = "#eab308"
            icon = "🟡"
        else:
            level = "Low"
            badge_color = "#10b981"
            icon = "🟢"
            
        disease_entry = {
            "name": d["name"],
            "crops": d["crops"],
            "pathogen_type": d["pathogen_type"],
            "score": round(score, 1),
            "level": level,
            "badge_color": badge_color,
            "icon": icon,
            "optimum": d["optimum"],
            "advice": d["advice"],
            "emergency": d["emergency"]
        }
        if pheno_info:
            disease_entry["phenology"] = pheno_info
            
        analyzed_diseases.append(disease_entry)
        
    # Sort diseases by highest risk score first
    analyzed_diseases.sort(key=lambda x: x["score"], reverse=True)
    
    # Calculate overall biosecurity farm risk
    if analyzed_diseases:
        top_scores = [d["score"] for d in analyzed_diseases[:3]]
        overall_score = round(sum(top_scores) / len(top_scores), 1)
    else:
        overall_score = 10.0
        
    if overall_score >= 70.0:
        overall_level = "Severe Threat (Action Required)"
        overall_color = "#ef4444"
        overall_icon = "🚨"
    elif overall_score >= 45.0:
        overall_level = "Elevated Outbreak Alert"
        overall_color = "#f97316"
        overall_icon = "⚠️"
    elif overall_score >= 25.0:
        overall_level = "Moderate Precautionary"
        overall_color = "#eab308"
        overall_icon = "🟡"
    else:
        overall_level = "Safe / Low Disease Activity"
        overall_color = "#10b981"
        overall_icon = "🟢"
        
    return {
        "overall_score": overall_score,
        "overall_level": overall_level,
        "overall_color": overall_color,
        "overall_icon": overall_icon,
        "diseases": analyzed_diseases,
        "top_threat": analyzed_diseases[0] if analyzed_diseases else None
    }

def get_regional_monitoring_stations(base_lat=38.58, base_lon=-121.49, location_name="Regional Belt", base_score=50.0):
    """
    Generates surrounding farm weather monitoring station nodes for pydeck geospatial radar.
    """
    stations = []
    offsets = [
        ("North Valley Station", 0.15, 0.08, base_score + 6.0),
        ("East Foothills Station", -0.05, 0.18, max(5.0, base_score - 12.0)),
        ("South Delta Farm Cluster", -0.18, -0.10, base_score + 14.0),
        ("West River Basin", 0.08, -0.20, base_score + 4.0),
        ("Central Agro-Station", 0.0, 0.0, base_score)
    ]
    for name, d_lat, d_lon, score in offsets:
        clamped_score = min(98.0, max(5.0, round(score, 1)))
        if clamped_score >= 70.0:
            color = [239, 68, 68, 200]
            level = "Severe Threat"
        elif clamped_score >= 45.0:
            color = [249, 115, 22, 200]
            level = "Elevated Alert"
        elif clamped_score >= 25.0:
            color = [234, 179, 8, 200]
            level = "Moderate"
        else:
            color = [16, 185, 129, 200]
            level = "Low Risk"
            
        stations.append({
            "name": name,
            "lat": base_lat + d_lat,
            "lon": base_lon + d_lon,
            "score": clamped_score,
            "level": level,
            "color": color,
            "radius": int(clamped_score * 35 + 800)
        })
    return stations
