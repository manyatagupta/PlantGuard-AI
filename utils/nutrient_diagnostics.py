"""
PlantGuard AI - Abiotic Plant Nutrient Deficiency & Cross-Diagnostic Matrix
Differentiates nutritional imbalances (Nitrogen, Potassium, Iron, Calcium, Boron, Zinc, etc.)
from biotic fungal, bacterial, and viral foliar pathogens.
Includes Soil pH Bioavailability Modeling and Agricultural Liming/Acidification Advisor.
"""

from typing import List, Dict, Any

NUTRIENT_DEFICIENCIES = {
    "Nitrogen (N)": {
        "nutrient": "Nitrogen (N)",
        "mobility": "Mobile (Symptoms appear on OLDEST lower leaves first)",
        "symptoms": "Uniform pale green to bright yellow chlorosis starting at leaf tips and progressing down the midrib. Stunted general growth and thin stems.",
        "confused_with_diseases": ["Fusarium Wilt", "Verticillium Wilt", "Root Rot"],
        "differential_rule": "Nitrogen chlorosis is uniform and symmetric across the lower canopy without distinct circular lesion halos or vascular browning inside the stem.",
        "soil_remedy": "Apply blood meal, ammonium sulfate, or balanced 20-20-20 fertilizer.",
        "foliar_emergency": "Foliar spray 1.0% Urea solution or liquid hydrolyzed fish emulsion."
    },
    "Potassium (K)": {
        "nutrient": "Potassium (K)",
        "mobility": "Mobile (Symptoms start on mature lower leaves)",
        "symptoms": "Marginal chlorosis quickly turning into dry, scorched brown leaf margins ('tip burn'). Leaves curl upward with necrosis between veins.",
        "confused_with_diseases": ["Tomato Early Blight", "Bacterial Leaf Scorch", "Anthracnose"],
        "differential_rule": "Potassium deficiency forms continuous crispy brown margins along the leaf perimeter, whereas fungal blights form discrete target-board spots that expand concentric rings.",
        "soil_remedy": "Apply Potassium Sulfate (SOP), Greensand, or Langbeinite.",
        "foliar_emergency": "Spray Potassium Silicate or kelp meal extract at 2-3 tbsp/gal."
    },
    "Phosphorus (P)": {
        "nutrient": "Phosphorus (P)",
        "mobility": "Mobile (Old/Lower foliage affected first)",
        "symptoms": "Dark dull green foliage turning distinct reddish-purple or bronze, especially on leaf undersides and along veins. Retarded root development.",
        "confused_with_diseases": ["Tomato Purple Leaf Disorder", "Beet Curly Top Virus"],
        "differential_rule": "Phosphorus deficiency creates purpling without leaf curling or mosaic blistering; often induced by cold wet soil locking up root uptake.",
        "soil_remedy": "Bone meal, rock phosphate, or monoammonium phosphate (MAP).",
        "foliar_emergency": "Foliar feed high-phosphorus bloom booster (e.g., 10-30-20) mixed with warm water."
    },
    "Calcium (Ca)": {
        "nutrient": "Calcium (Ca)",
        "mobility": "Immobile (Symptoms appear strictly on NEW young growth & fruits)",
        "symptoms": "Distorted, cupped, hooked young leaves with necrotic tips. Sunken leathery black lesions at the blossom end of fruits (Blossom End Rot in tomato/pepper).",
        "confused_with_diseases": ["Late Blight (Phytophthora infestans)", "Anthracnose Fruit Rot"],
        "differential_rule": "Blossom End Rot is strictly confined to the calyx/blossom end of fruit with a dry, leathery texture. Late blight attacks both foliage and fruit randomly with water-soaked, foul-smelling rot.",
        "soil_remedy": "Agricultural gypsum (calcium sulfate) or dolomitic limestone; maintain even irrigation.",
        "foliar_emergency": "Foliar spray Chelated Calcium (Calcium Chloride / Calcium Nitrate) at 15 ml/gal."
    },
    "Magnesium (Mg)": {
        "nutrient": "Magnesium (Mg)",
        "mobility": "Mobile (Lower and middle leaves)",
        "symptoms": "Dramatic interveinal chlorosis (leaf blade turns yellow while veins remain sharply green), later developing bronze/red necrotic patches between veins.",
        "confused_with_diseases": ["Tomato Yellow Leaf Curl Virus", "Septoria Leaf Spot"],
        "differential_rule": "Magnesium deficiency produces an intricate 'herringbone' yellow pattern with veins remaining emerald green. No water-soaked lesions or concentric rings are present.",
        "soil_remedy": "Epsom Salt (Magnesium Sulfate) or Dolomitic Limestone.",
        "foliar_emergency": "Dissolve 1 tablespoon Epsom Salt per gallon of water and mist foliage thoroughly."
    },
    "Iron (Fe)": {
        "nutrient": "Iron (Fe)",
        "mobility": "Immobile (Strictly upper NEW youngest shoots)",
        "symptoms": "Severe interveinal chlorosis on the youngest emerging leaves (blades turn yellow while veins remain sharply green). In acute stages, newly formed leaves emerge bright yellow to pure bleached ivory-white.",
        "confused_with_diseases": ["Powdery Mildew (due to pale color)", "Alfalfa Mosaic Virus"],
        "differential_rule": "Iron deficiency has no powdery fungal mycelium that rubs off with a finger. The leaves are structurally intact, simply devoid of green chlorophyll synthesis.",
        "soil_remedy": "Lower soil pH below 6.5; drench root zone with Iron EDDHA / EDTA chelate.",
        "foliar_emergency": "Foliar mist with Chelated Liquid Iron at 5 ml/gal during cool morning hours."
    },
    "Sulfur (S)": {
        "nutrient": "Sulfur (S)",
        "mobility": "Immobile (Upper young foliage affected first)",
        "symptoms": "Youngest emerging leaves turn pale yellowish-green with veins remaining lighter or uniform; stunted spindly stalks with delayed crop maturity.",
        "confused_with_diseases": ["Nitrogen Deficiency", "Tomato Yellow Leaf Curl Virus"],
        "differential_rule": "Sulfur deficiency mimics Nitrogen, but sulfur yellowing starts at the TOP of the plant on the newest leaves, whereas nitrogen yellowing starts at the BOTTOM on oldest leaves.",
        "soil_remedy": "Apply elemental sulfur, potassium sulfate, or agricultural gypsum.",
        "foliar_emergency": "Foliar spray Epsom salt (magnesium sulfate) or ammonium sulfate at 1 tbsp/gal."
    },
    "Zinc (Zn)": {
        "nutrient": "Zinc (Zn)",
        "mobility": "Immobile (Young and middle emerging leaves)",
        "symptoms": "Little leaf syndrome: severe foliar reduction and shortened internodes creating a rosetted cluster. Mottled interveinal chlorosis with white/yellow bleached bands.",
        "confused_with_diseases": ["Tomato Mosaic Virus", "Broad Mite Injury", "Herbicide Drift"],
        "differential_rule": "Zinc deficiency causes characteristic 'rosetting' (compressed stem nodes with tiny leaves) without mosaic leaf blistering or viral vein clearing.",
        "soil_remedy": "Zinc sulfate monohydrate broadcast at 5-10 lbs/acre.",
        "foliar_emergency": "Foliar application of Chelated Zinc (Zn-EDTA) at 5 ml/gal."
    },
    "Manganese (Mn)": {
        "nutrient": "Manganese (Mn)",
        "mobility": "Immobile (Upper and middle young foliage)",
        "symptoms": "Intricate checkered interveinal chlorosis with small tan, grey, or black speckling and necrotic pinholes across the leaf lamina.",
        "confused_with_diseases": ["Septoria Leaf Spot", "Spider Mite Stippling", "Bacterial Spot"],
        "differential_rule": "Manganese chlorosis develops an intricate net-veined checkered pattern with necrotic flecks strictly confined between veins without yellow fungal halos.",
        "soil_remedy": "Manganese sulfate; lower soil pH below 6.8.",
        "foliar_emergency": "Foliar mist with Manganese Sulfate or chelated Mn at 1 tbsp/gal."
    },
    "Boron (B)": {
        "nutrient": "Boron (B)",
        "mobility": "Immobile (Terminal growing points and reproductive organs)",
        "symptoms": "Terminal shoot dieback ('witch's broom'), thick brittle twisted leaves, poor fruit set, and corky brown lesions inside fruit (corky core in apple, hollow heart in potato).",
        "confused_with_diseases": ["Tomato Spotted Wilt Virus", "Apple Bitter Pit", "Bacterial Canker"],
        "differential_rule": "Boron deficiency arrests growing shoot tips causing brittle, snapped petioles and internal corking without viral ring spots or bacterial vascular streaking.",
        "soil_remedy": "Solubor or agricultural borax applied carefully (narrow margin between deficiency and toxicity).",
        "foliar_emergency": "Foliar feed Liquid Boron (Solubor) at 1-2 tsp/gal prior to bloom."
    }
}

def diagnose_nutrient_deficiency(leaf_position: str, primary_symptom: str) -> list:
    """
    Ranks likely nutrient deficiencies based on plant canopy location and symptom pattern.
    """
    matches = []
    pos_lower = leaf_position.lower()
    symp_lower = primary_symptom.lower()
    
    for name, data in NUTRIENT_DEFICIENCIES.items():
        score = 0
        
        # Check mobility alignment
        if "lower" in pos_lower or "old" in pos_lower or "mature" in pos_lower:
            if "Mobile" in data["mobility"] and "Immobile" not in data["mobility"]:
                score += 45
        elif "upper" in pos_lower or "new" in pos_lower or "young" in pos_lower or "shoot" in pos_lower:
            if "Immobile" in data["mobility"]:
                score += 45
        elif "fruit" in pos_lower or "blossom" in pos_lower:
            if "Calcium" in name or "Boron" in name:
                score += 55
                
        # Check symptom keyword alignment
        keywords = [
            "interveinal", "chlorosis", "yellow", "purple", "scorch", "margin", 
            "blossom end", "distorted", "cupped", "rosetted", "little leaf", 
            "checkered", "speckling", "brittle", "hollow", "corky"
        ]
        for kw in keywords:
            if kw in symp_lower and kw in data["symptoms"].lower():
                score += 20
                
        if score > 0:
            matches.append({
                "nutrient": name,
                "confidence_score": min(95, score),
                "mobility": data["mobility"],
                "symptoms": data["symptoms"],
                "confused_with": data["confused_with_diseases"],
                "differential_rule": data["differential_rule"],
                "remedy": data["soil_remedy"],
                "emergency_spray": data["foliar_emergency"]
            })
            
    matches.sort(key=lambda x: x["confidence_score"], reverse=True)
    return matches

def get_deficiency_vs_disease_comparison(disease_name: str) -> dict:
    """
    Finds nutrient deficiencies that closely mimic the specified disease.
    """
    d_clean = disease_name.lower()
    for name, data in NUTRIENT_DEFICIENCIES.items():
        for confused in data["confused_with_diseases"]:
            if any(w in d_clean for w in confused.lower().split() if len(w) > 3):
                return {
                    "matched": True,
                    "suspected_disease": disease_name,
                    "mimicking_nutrient": name,
                    "differential_rule": data["differential_rule"],
                    "foliar_remedy": data["foliar_emergency"]
                }
    return {
        "matched": False,
        "suspected_disease": disease_name,
        "mimicking_nutrient": "General N-P-K & Micronutrient Balance",
        "differential_rule": "Biotic diseases typically feature fungal fruiting bodies, bacterial exudates, or water-soaked halos unlike nutrient deficiencies.",
        "foliar_remedy": "Apply balanced micronutrient foliar spray with zinc, iron, and kelp extract."
    }

def calculate_soil_ph_bioavailability(soil_ph: float, soil_texture: str = "Loamy") -> Dict[str, Any]:
    """
    Calculates nutrient bioavailability percentages across all essential elements,
    identifies chemical lockouts, and calculates agricultural liming or acidification
    amendments based on soil buffer capacity.
    
    Args:
        soil_ph: Soil reaction pH value (typically 4.0 to 9.0).
        soil_texture: Soil texture type ('Sandy', 'Loamy', 'Clay').
        
    Returns:
        Dictionary containing bioavailability curves, locked-out elements, amendment dosages, and status.
    """
    ph = max(4.0, min(9.0, float(soil_ph)))
    texture = soil_texture.capitalize() if soil_texture else "Loamy"
    
    # Texture buffer multipliers for amendments
    texture_multipliers = {
        "Sandy": 0.8,
        "Loamy": 1.2,
        "Clay": 1.7
    }
    t_mult = texture_multipliers.get(texture, 1.2)
    
    # Calculate bioavailability for 10 elements using polynomial & sigmoid response curves
    # Nitrogen (N): High 6.0-8.0, drops under 5.5
    if ph < 5.5:
        avail_n = max(15.0, 15.0 + (ph - 4.0) * 45.0)
    elif ph <= 8.0:
        avail_n = 95.0
    else:
        avail_n = max(50.0, 95.0 - (ph - 8.0) * 35.0)
        
    # Phosphorus (P): Sharp peak 6.2 - 7.3, severe fixation <5.5 (Al/Fe) and >7.8 (Ca)
    if ph < 6.2:
        avail_p = max(15.0, 20.0 + (ph - 4.0) * 32.0)
    elif ph <= 7.3:
        avail_p = 95.0
    else:
        avail_p = max(25.0, 95.0 - (ph - 7.3) * 42.0)
        
    # Potassium (K): High 6.0-8.5, drops below 5.5
    if ph < 6.0:
        avail_k = max(25.0, 30.0 + (ph - 4.0) * 32.0)
    else:
        avail_k = 95.0
        
    # Calcium (Ca) & Magnesium (Mg): Drops sharply below 6.0
    if ph < 6.2:
        avail_ca_mg = max(10.0, 15.0 + (ph - 4.0) * 36.0)
    else:
        avail_ca_mg = 95.0
        
    # Sulfur (S): High 6.0-8.5
    if ph < 5.8:
        avail_s = max(25.0, 30.0 + (ph - 4.0) * 35.0)
    else:
        avail_s = 95.0
        
    # Iron (Fe): Highest in acidic soils, precipitates heavily >7.0
    if ph <= 6.2:
        avail_fe = 100.0
    elif ph <= 7.0:
        avail_fe = max(60.0, 100.0 - (ph - 6.2) * 50.0)
    else:
        avail_fe = max(10.0, 60.0 - (ph - 7.0) * 30.0)
        
    # Manganese (Mn): High in acid soils, drops >6.8
    if ph <= 6.5:
        avail_mn = 100.0
    elif ph <= 7.2:
        avail_mn = max(50.0, 100.0 - (ph - 6.5) * 70.0)
    else:
        avail_mn = max(10.0, 50.0 - (ph - 7.2) * 30.0)
        
    # Zinc (Zn) & Boron (B): Optimal 5.5-7.0, drops >7.4
    if ph < 5.0:
        avail_zn_b = 60.0
    elif ph <= 7.2:
        avail_zn_b = 92.0
    else:
        avail_zn_b = max(15.0, 92.0 - (ph - 7.2) * 45.0)

    bioavailability = {
        "Nitrogen (N)": round(avail_n, 1),
        "Phosphorus (P)": round(avail_p, 1),
        "Potassium (K)": round(avail_k, 1),
        "Calcium (Ca)": round(avail_ca_mg, 1),
        "Magnesium (Mg)": round(avail_ca_mg * 0.95, 1),
        "Sulfur (S)": round(avail_s, 1),
        "Iron (Fe)": round(avail_fe, 1),
        "Manganese (Mn)": round(avail_mn, 1),
        "Zinc (Zn)": round(avail_zn_b, 1),
        "Boron (B)": round(avail_zn_b * 0.92, 1)
    }
    
    # Identify severe lockouts (<50% availability)
    lockouts = [k for k, v in bioavailability.items() if v < 50.0]
    
    # Status & Soil Amendment Recommendations
    target_ph = 6.5
    lime_lbs_1000sqft = 0.0
    lime_kg_ha = 0.0
    sulfur_lbs_1000sqft = 0.0
    sulfur_kg_ha = 0.0
    
    if ph < 6.2:
        status = "Acidic Soil Lockout (Al/Fe Fixation)"
        status_color = "#ef4444"
        delta = target_ph - ph
        lime_lbs_1000sqft = round(delta * 22.0 * t_mult, 1)
        lime_kg_ha = round(lime_lbs_1000sqft * 48.8, 0)
        action_summary = (
            f"Soil is overly acidic (pH {ph:.1f}). Phosphorus is chemically fixed by aluminum and iron, "
            f"while Calcium and Magnesium uptake are severely suppressed. Broadcast Agricultural Dolomitic Limestone."
        )
    elif ph > 7.3:
        status = "Alkaline Chlorosis Hazard (Lime Lockout)"
        status_color = "#f59e0b"
        delta = ph - target_ph
        sulfur_lbs_1000sqft = round(delta * 10.0 * t_mult, 1)
        sulfur_kg_ha = round(sulfur_lbs_1000sqft * 48.8 * 0.45, 0)
        action_summary = (
            f"Soil is alkaline (pH {ph:.1f}). Micronutrients (Iron, Zinc, Manganese, Boron) form insoluble carbonates. "
            f"Apply Elemental Soil Sulfur to acidify root zone, and provide emergency foliar chelated iron (Fe-EDDHA)."
        )
    else:
        status = "Optimal Bioavailability Range"
        status_color = "#10b981"
        action_summary = (
            f"Soil pH {ph:.1f} is in the prime agronomic zone for crop nutrient uptake. "
            f"Maintain organic compost and monitor annual soil tests."
        )
        
    return {
        "ph": ph,
        "texture": texture,
        "status": status,
        "status_color": status_color,
        "action_summary": action_summary,
        "lockouts": lockouts,
        "bioavailability": bioavailability,
        "lime_lbs_1000sqft": lime_lbs_1000sqft,
        "lime_kg_ha": lime_kg_ha,
        "sulfur_lbs_1000sqft": sulfur_lbs_1000sqft,
        "sulfur_kg_ha": sulfur_kg_ha
    }
