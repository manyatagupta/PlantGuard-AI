"""
PlantGuard AI - Abiotic Plant Nutrient Deficiency & Cross-Diagnostic Matrix
Differentiates nutritional imbalances (Nitrogen, Potassium, Iron, Calcium, etc.)
from biotic fungal, bacterial, and viral foliar pathogens.
"""

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
        "symptoms": "Severe interveinal chlorosis on the youngest emerging leaves. In acute stages, newly formed leaves emerge almost pure bleached ivory-white.",
        "confused_with_diseases": ["Powdery Mildew (due to pale color)", "Alfalfa Mosaic Virus"],
        "differential_rule": "Iron deficiency has no powdery fungal mycelium that rubs off with a finger. The leaves are structurally intact, simply devoid of green chlorophyll synthesis.",
        "soil_remedy": "Lower soil pH below 6.5; drench root zone with Iron EDDHA / EDTA chelate.",
        "foliar_emergency": "Foliar mist with Chelated Liquid Iron at 5 ml/gal during cool morning hours."
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
        if "lower" in pos_lower or "old" in pos_lower:
            if "Mobile" in data["mobility"] and "Immobile" not in data["mobility"]:
                score += 45
        elif "upper" in pos_lower or "new" in pos_lower or "young" in pos_lower:
            if "Immobile" in data["mobility"]:
                score += 45
                
        # Check symptom keyword alignment
        keywords = ["interveinal", "chlorosis", "yellow", "purple", "scorch", "margin", "blossom end", "distorted", "cupped"]
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
