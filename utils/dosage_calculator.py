"""
PlantGuard AI - Precision Spray & Fertilizer Dosage Calculator
Computes exact active chemical/organic concentrate amounts, total water carrier volume,
tank mixing protocols, and safety intervals based on user plot dimensions or plant count.
"""

# Agricultural Treatment Formulations Database
TREATMENT_DATABASE = {
    # --- ORGANIC & BIO-CONTROL ---
    "Neem Oil (Cold-Pressed 70%)": {
        "type": "Organic Botanical",
        "form": "liquid",
        "rate_per_liter_ml": 5.0,  # 5 ml per Liter (approx 2 tbsp/gal)
        "recommended_for": ["Spider Mites", "Aphids", "Powdery Mildew", "Early Blight Prevention"],
        "crops": ["Tomato", "Apple", "Bell Pepper", "Potato", "Corn (Maize)"],
        "adjuvant": "Add 2 ml of mild castile or dish soap per liter as emulsifier.",
        "best_time": "Early morning or dusk (Avoid hot direct sun to prevent leaf scorch).",
        "phi_days": 0,  # Pre-harvest interval
        "rei_hours": 4,  # Re-entry interval
        "ppe": "Safety glasses and waterproof garden gloves.",
        "mixing_notes": "Mix soap with warm water first until dissolved, then slowly stir in neem oil until fully emulsified."
    },
    "Copper Octanoate (Liquid Copper Soap)": {
        "type": "Organic Fungicide / Bactericide",
        "form": "liquid",
        "rate_per_liter_ml": 10.0,
        "recommended_for": ["Bacterial Spot", "Late Blight", "Early Blight", "Apple Scab", "Leaf Mold"],
        "crops": ["Tomato", "Potato", "Bell Pepper", "Apple"],
        "adjuvant": "None required; pre-formulated soap aids spread.",
        "best_time": "Morning after dew dries, with at least 4 hours of rain-free drying time.",
        "phi_days": 1,
        "rei_hours": 12,
        "ppe": "Eye goggles, chemical-resistant nitrile gloves, and long sleeves.",
        "mixing_notes": "Fill sprayer 1/2 with water, pour copper concentrate, shake, then top off with remaining water."
    },
    "Bacillus subtilis (Biofungicide - Serenade)": {
        "type": "Biological Bio-Pesticide",
        "form": "liquid",
        "rate_per_liter_ml": 10.0,
        "recommended_for": ["Early Blight", "Gray Leaf Spot", "Septoria", "Target Spot", "Root Rot"],
        "crops": ["Tomato", "Potato", "Corn (Maize)", "Bell Pepper", "Apple"],
        "adjuvant": "Organic spreader-sticker improves biological colonization.",
        "best_time": "Overcast days or late afternoon to prevent UV breakdown of beneficial bacteria.",
        "phi_days": 0,
        "rei_hours": 4,
        "ppe": "N95 particulate dust/mist mask and gloves.",
        "mixing_notes": "Use non-chlorinated or filtered water; chlorine reduces live bacterial spore counts."
    },
    "Potassium Bicarbonate (MilStop / GreenCure)": {
        "type": "Organic Contact Fungicide",
        "form": "dry",
        "rate_per_liter_g": 3.5,  # 3.5 g per Liter (approx 1 tbsp/gal)
        "recommended_for": ["Leaf Mold", "Powdery Mildew", "Septoria Leaf Spot"],
        "crops": ["Tomato", "Apple", "Bell Pepper"],
        "adjuvant": "Add 1 ml insecticidal soap or light horticultural oil per liter.",
        "best_time": "At first visible sign of mold patches. Direct contact kills spores.",
        "phi_days": 0,
        "rei_hours": 4,
        "ppe": "Standard dust mask and gloves.",
        "mixing_notes": "Dissolve completely in warm water before transferring to spray tank."
    },
    "Sulfur 80% Wettable Powder": {
        "type": "Organic / Mineral Protectant",
        "form": "dry",
        "rate_per_liter_g": 3.0,
        "recommended_for": ["Apple Scab", "Cedar Apple Rust", "Spider Mites"],
        "crops": ["Apple", "Tomato"],
        "adjuvant": "None needed. Do NOT mix with oils.",
        "best_time": "Cool mornings. NEVER apply when temperatures exceed 29°C (85°F) or within 30 days of oil application.",
        "phi_days": 1,
        "rei_hours": 24,
        "ppe": "Chemical splash goggles, respirator mask, protective gloves.",
        "mixing_notes": "Pre-slurry the sulfur powder with a small jar of water before pouring into main sprayer."
    },

    # --- CONVENTIONAL / CHEMICAL ---
    "Chlorothalonil 75 WP (Daconil Broad-Spectrum)": {
        "type": "Chemical Contact Fungicide",
        "form": "dry",
        "rate_per_liter_g": 2.5,
        "recommended_for": ["Late Blight", "Early Blight", "Septoria", "Target Spot", "Apple Scab"],
        "crops": ["Tomato", "Potato", "Apple", "Corn (Maize)"],
        "adjuvant": "Non-ionic surfactant can improve rainfastness.",
        "best_time": "Prior to predicted rain events for multi-week preventative barrier.",
        "phi_days": 7,
        "rei_hours": 12,
        "ppe": "Full PPE: Chemical goggles, half-face respirator, rubber gloves, coveralls.",
        "mixing_notes": "Continuous agitation is recommended to keep wettable powder in suspension."
    },
    "Mancozeb 75 WP (Dithane M-45 Multi-Site)": {
        "type": "Chemical Protectant Fungicide",
        "form": "dry",
        "rate_per_liter_g": 2.5,
        "recommended_for": ["Late Blight", "Apple Scab", "Common Rust", "Bacterial Spot (in tank mix)"],
        "crops": ["Potato", "Tomato", "Apple", "Corn (Maize)", "Bell Pepper"],
        "adjuvant": "Excellent synergy when tank-mixed with Fixed Copper for bacterial spot.",
        "best_time": "Dry calm days, wind under 8 km/h.",
        "phi_days": 14,
        "rei_hours": 24,
        "ppe": "Respirator, chemical-resistant gloves, full eye protection.",
        "mixing_notes": "Add to tank after water is 75% full. Agitate vigorously."
    },
    "Myclobutanil (Immunox Systemic Curative)": {
        "type": "Chemical Systemic Triazole",
        "form": "liquid",
        "rate_per_liter_ml": 1.25,
        "recommended_for": ["Apple Scab (Curative)", "Cedar Apple Rust", "Powdery Mildew"],
        "crops": ["Apple"],
        "adjuvant": "None needed; systemic translaminar absorption into leaf tissue.",
        "best_time": "Effective up to 72 hours AFTER an infection rainfall event.",
        "phi_days": 14,
        "rei_hours": 24,
        "ppe": "Goggles, rubber gloves, and long sleeves.",
        "mixing_notes": "Do not exceed maximum 5 applications per orchard season to prevent pathogen resistance."
    },
    "Azoxystrobin 22.9% (Quadris / Heritage)": {
        "type": "Chemical Systemic Strobilurin",
        "form": "liquid",
        "rate_per_liter_ml": 0.8,
        "recommended_for": ["Northern Leaf Blight", "Gray Leaf Spot", "Early Blight", "Black Rot"],
        "crops": ["Corn (Maize)", "Potato", "Tomato"],
        "adjuvant": "Crop oil concentrate (COC) or non-ionic surfactant.",
        "best_time": "Vegetative canopy development and early bloom.",
        "phi_days": 7,
        "rei_hours": 12,
        "ppe": "Nitrile gloves, goggles, face shield.",
        "mixing_notes": "CAUTION: Highly phytotoxic to certain apple varieties. Never allow spray drift toward apple trees."
    },

    # --- FOLIAR NUTRITION & FERTILIZERS ---
    "Balanced N-P-K (19-19-19 Water Soluble)": {
        "type": "Foliar Fertilizer & Vigor Booster",
        "form": "dry",
        "rate_per_liter_g": 4.0,
        "recommended_for": ["Nutrient Deficiencies", "Post-Disease Recovery", "Vegetative Growth"],
        "crops": ["Tomato", "Potato", "Bell Pepper", "Corn (Maize)", "Apple"],
        "adjuvant": "Kelp extract enhances nutrient absorption.",
        "best_time": "Cool morning (6:00 - 9:00 AM) when leaf stomata are open.",
        "phi_days": 0,
        "rei_hours": 2,
        "ppe": "Dust mask and gloves.",
        "mixing_notes": "Dissolve completely in warm water. Avoid spraying in direct afternoon sun."
    },
    "Calcium Nitrate (Prevents Blossom End Rot)": {
        "type": "Mineral Supplement",
        "form": "dry",
        "rate_per_liter_g": 3.0,
        "recommended_for": ["Blossom End Rot", "Cell Wall Strengthening", "Heat Stress"],
        "crops": ["Tomato", "Bell Pepper", "Apple"],
        "adjuvant": "None required.",
        "best_time": "Early fruit set stage; direct spray onto young developing fruit clusters.",
        "phi_days": 0,
        "rei_hours": 2,
        "ppe": "Standard garden gloves and eyewear.",
        "mixing_notes": "Do NOT mix directly with phosphorus-heavy fertilizers in concentrated form to avoid precipitate."
    },
    "Magnesium Sulfate (Pure Epsom Salt)": {
        "type": "Micronutrient Supplement",
        "form": "dry",
        "rate_per_liter_g": 3.0,
        "recommended_for": ["Interveinal Chlorosis (Yellowing)", "Pepper Fruit Setting"],
        "crops": ["Bell Pepper", "Tomato", "Potato"],
        "adjuvant": "None needed.",
        "best_time": "At start of flowering cycle and 3 weeks later.",
        "phi_days": 0,
        "rei_hours": 0,
        "ppe": "Standard garden gloves.",
        "mixing_notes": "Dissolves rapidly in room temperature water."
    }
}

# Unit Conversion Utilities
def convert_to_household_units(grams_or_ml: float, is_liquid: bool) -> str:
    """Converts metric grams or ml into friendly kitchen measures (tsp, tbsp, cups, fl oz)."""
    if is_liquid:
        ml = grams_or_ml
        if ml < 5.0:
            return f"{round(ml / 5.0, 1)} teaspoon (tsp)"
        elif ml < 30.0:
            tbsp = round(ml / 15.0, 1)
            tsp = round((ml % 15.0) / 5.0, 1)
            return f"{tbsp} tablespoon (tbsp)" + (f" + {tsp} tsp" if tsp > 0.2 else "")
        elif ml < 240.0:
            oz = round(ml / 29.57, 1)
            return f"{oz} fluid oz ({round(ml / 15.0, 1)} tbsp)"
        else:
            cups = round(ml / 240.0, 2)
            return f"{cups} measuring cups ({round(ml, 0)} ml)"
    else:
        # Dry powder conversions (approx 1 tbsp dry chemical = 12-14g, 1 tsp = 4-5g)
        g = grams_or_ml
        if g < 4.5:
            return f"{round(g / 4.5, 1)} level teaspoon (tsp)"
        elif g < 30.0:
            tbsp = round(g / 14.0, 1)
            return f"{tbsp} level tablespoon (tbsp) (~{round(g, 1)} g)"
        elif g < 454.0:
            oz = round(g / 28.35, 1)
            return f"{oz} oz dry weight ({round(g, 0)} g)"
        else:
            lbs = round(g / 453.6, 2)
            return f"{lbs} lbs ({round(g, 0)} g)"

# Water Carrier Estimation by Crop & Area
def calculate_water_volume(area_type: str, area_val: float, plant_stage: str = "Medium Canopy") -> float:
    """
    Computes required water carrier volume in Liters based on plot configuration and canopy density.
    """
    multiplier = 1.0
    if plant_stage == "Early Seedlings / Low Canopy":
        multiplier = 0.65
    elif plant_stage == "Dense Fruiting Canopy / Orchard Trees":
        multiplier = 1.4

    if area_type == "Garden Bed (Sq. Feet)":
        # Average dense vegetable canopy uses approx 3.8 Liters (1 Gal) per 250 sq ft
        liters = (area_val / 250.0) * 3.785 * multiplier
    elif area_type == "Plant Count (Individual Plants)":
        # Average 0.35 L per mature vegetable vine (tomato/pepper)
        liters = area_val * 0.35 * multiplier
    elif area_type == "Greenhouse (Sq. Meters)":
        # Greenhouse standard: approx 10 L per 100 m²
        liters = (area_val / 100.0) * 10.0 * multiplier
    elif area_type == "Farm Acreage (Acres)":
        # Commercial tractor / backpack broadcast: approx 100 Liters per acre for full foliage
        liters = area_val * 100.0 * multiplier
    else:
        liters = 5.0
        
    return max(0.5, round(liters, 2))

# Main Calculation Engine
def compute_spray_dosage(
    product_name: str,
    area_type: str,
    area_val: float,
    plant_stage: str = "Medium Canopy"
) -> dict:
    """
    Computes exact dosage, water requirement, mixing recipe, and safety guidelines.
    """
    if product_name not in TREATMENT_DATABASE:
        raise ValueError(f"Unknown product formulation: {product_name}")
        
    product = TREATMENT_DATABASE[product_name]
    is_liquid = product["form"] == "liquid"
    
    # 1. Calculate water carrier
    water_liters = calculate_water_volume(area_type, area_val, plant_stage)
    water_gallons = round(water_liters / 3.785, 2)
    
    # 2. Calculate active concentrate
    if is_liquid:
        active_amount = round(water_liters * product["rate_per_liter_ml"], 1)
        unit = "ml"
    else:
        active_amount = round(water_liters * product["rate_per_liter_g"], 1)
        unit = "grams"
        
    household_measure = convert_to_household_units(active_amount, is_liquid)
    
    # 3. Mixing recipe steps (WALES method)
    mixing_steps = [
        f"1. Fill your spray container with **{round(water_liters * 0.6, 1)} Liters** ({round(water_gallons * 0.6, 1)} Gal) of clean water.",
        f"2. Measure out exactly **{active_amount} {unit}** (approx. **{household_measure}**) of **{product_name}**.",
        f"3. {product['mixing_notes']}",
        f"4. Add adjuvant: {product['adjuvant']}",
        f"5. Top off with remaining water to reach the full **{water_liters} Liters** ({water_gallons} Gal) mark.",
        "6. Close lid tightly and agitate / shake for 30 seconds before pressurizing."
    ]
    
    return {
        "product_name": product_name,
        "category": product["type"],
        "form": product["form"],
        "water_liters": water_liters,
        "water_gallons": water_gallons,
        "active_amount": active_amount,
        "active_unit": unit,
        "household_measure": household_measure,
        "mixing_steps": mixing_steps,
        "recommended_for": product["recommended_for"],
        "crops": product["crops"],
        "best_time": product["best_time"],
        "phi_days": product["phi_days"],
        "rei_hours": product["rei_hours"],
        "ppe": product["ppe"]
    }

def get_all_products():
    """Returns the list of all available formulations."""
    return list(TREATMENT_DATABASE.keys())

def get_product_details(name):
    """Returns full metadata for a product."""
    return TREATMENT_DATABASE.get(name, {})


# --- Tank-Mix Incompatibility Rules Engine ---
INCOMPATIBILITY_RULES = [
    {
        "pair": ("sulfur", "neem"),
        "severity": "CRITICAL",
        "badge_color": "#ef4444",
        "title": "Severe Foliar Phytotoxicity: Sulfur + Oil Incompatibility",
        "hazard": "Sulfur combines with oil droplets to penetrate deep into leaf cuticle cells, causing rapid cellular lysis, necrotic scorching, and widespread defoliation.",
        "action": "Maintain a strict 30-day withholding interval between sulfur and any oil applications. NEVER combine in the same spray tank.",
        "withholding_days": 30
    },
    {
        "pair": ("sulfur", "oil"),
        "severity": "CRITICAL",
        "badge_color": "#ef4444",
        "title": "Severe Phytotoxicity: Elemental Sulfur + Horticultural Oil",
        "hazard": "Oil acts as an aggressive penetrant for elemental sulfur, producing acute leaf scorch and fruit drop at temperatures above 26°C.",
        "action": "Separate applications by at least 30 days.",
        "withholding_days": 30
    },
    {
        "pair": ("calcium", "phosphate"),
        "severity": "HIGH",
        "badge_color": "#f59e0b",
        "title": "Chemical Precipitation: Insoluble Calcium Phosphate Sludge",
        "hazard": "Free calcium cations (Ca2+) bond with orthophosphates (PO4 3-) to precipitate out insoluble chalky calcium phosphate, clogging nozzles and blinding pump filters.",
        "action": "Apply Calcium Nitrate and Phosphate/NPK fertilizers in separate foliar passes, or switch to fully chelated Ca-EDTA.",
        "withholding_days": 0
    },
    {
        "pair": ("calcium", "sulfate"),
        "severity": "HIGH",
        "badge_color": "#f59e0b",
        "title": "Chemical Precipitation: Insoluble Gypsum Formation",
        "hazard": "Calcium Nitrate mixed with Magnesium Sulfate or Potassium Sulfate forms insoluble crystalline Calcium Sulfate (Gypsum sludge) that settles like concrete in tank bottoms.",
        "action": "Do not mix calcium salts with sulfate-bearing fertilizers or copper sulfate.",
        "withholding_days": 0
    },
    {
        "pair": ("bacillus", "copper"),
        "severity": "HIGH",
        "badge_color": "#f59e0b",
        "title": "Biological Neutralization: Copper Bactericide vs Bacillus Biocontrol",
        "hazard": "Broad-spectrum copper bactericides actively eradicate live bacterial cells and spores of Bacillus subtilis / amyloliquefaciens, neutralizing the biofungicide.",
        "action": "Stagger applications: spray biological inoculants 3 to 5 days after copper residues have weathered.",
        "withholding_days": 4
    },
    {
        "pair": ("bacillus", "mancozeb"),
        "severity": "MEDIUM",
        "badge_color": "#3b82f6",
        "title": "Biological Inactivation: Synthetic Protectant vs Biofungicide",
        "hazard": "Synthetic broad-spectrum dithiocarbamates suppress microbial colonization of beneficial fungal and bacterial biofilms on the phyllosphere.",
        "action": "Apply protectants first to establish baseline barrier, then inoculate with biologicals 7 days later.",
        "withholding_days": 7
    },
    {
        "pair": ("chlorothalonil", "oil"),
        "severity": "MEDIUM",
        "badge_color": "#3b82f6",
        "title": "Elevated Phytotoxicity Risk: Chlorothalonil + Heavy Adjuvant Oils",
        "hazard": "Excessive oil penetration can cause chlorothalonil to burn tender fruiting structures or young leaf margins under hot sun.",
        "action": "Use light non-ionic surfactants rather than concentrated crop oils when applying chlorothalonil in temperatures > 28°C.",
        "withholding_days": 0
    }
]


KEYWORD_ALIASES = {
    "sulfur": ["sulfur", "sulphur"],
    "oil": ["oil", "neem"],
    "neem": ["neem"],
    "calcium": ["calcium"],
    "phosphate": ["phosphate", "n-p-k", "phosphor"],
    "sulfate": ["sulfate", "sulphate", "epsom"],
    "bacillus": ["bacillus", "serenade"],
    "copper": ["copper"],
    "mancozeb": ["mancozeb", "dithane"],
    "chlorothalonil": ["chlorothalonil", "daconil"]
}

def _matches_chem_key(prod_text: str, key: str) -> bool:
    aliases = KEYWORD_ALIASES.get(key, [key])
    return any(a in prod_text for a in aliases)


def check_tank_mix_compatibility(selected_products: list) -> dict:
    """
    Evaluates physical, chemical, and biological compatibility between two or more
    agricultural treatment products.
    
    Args:
        selected_products: List of product formulation names from TREATMENT_DATABASE.
        
    Returns:
        dict: Overall compatibility rating, list of conflict alerts, and WALES addition sequence.
    """
    if not selected_products or len(selected_products) < 2:
        return {
            "status": "SAFE",
            "badge_color": "#10b981",
            "summary": "Single product selected. No tank-mix conflicts present.",
            "conflicts": [],
            "wales_order": selected_products,
            "overall_safety": "SAFE"
        }
        
    conflicts = []
    has_critical = False
    has_high = False
    
    # Check pairwise combinations
    for i in range(len(selected_products)):
        for j in range(i + 1, len(selected_products)):
            p1 = selected_products[i].lower()
            p2 = selected_products[j].lower()
            
            for rule in INCOMPATIBILITY_RULES:
                k1, k2 = rule["pair"]
                if (_matches_chem_key(p1, k1) and _matches_chem_key(p2, k2)) or (_matches_chem_key(p1, k2) and _matches_chem_key(p2, k1)):
                    conflicts.append({
                        "product_a": selected_products[i],
                        "product_b": selected_products[j],
                        "severity": rule["severity"],
                        "badge_color": rule["badge_color"],
                        "title": rule["title"],
                        "hazard": rule["hazard"],
                        "action": rule["action"],
                        "withholding_days": rule["withholding_days"]
                    })
                    if rule["severity"] == "CRITICAL":
                        has_critical = True
                    elif rule["severity"] == "HIGH":
                        has_high = True
                        
    # Determine overall status
    if has_critical:
        status = "CRITICAL INCOMPATIBILITY"
        badge_color = "#ef4444"
        summary = "DO NOT MIX: Severe chemical phytotoxicity or plant damage will occur if combined in the spray tank."
    elif has_high:
        status = "INCOMPATIBLE MIX HAZARD"
        badge_color = "#f59e0b"
        summary = "CAUTION: Chemical precipitation or biological neutralization detected. Products should be applied separately."
    elif conflicts:
        status = "CONDITIONAL COMPATIBILITY"
        badge_color = "#3b82f6"
        summary = "MODERATE RISK: Tank mix is possible with strict temperature control and agitation, but test in jar first."
    else:
        status = "COMPATIBLE TANK MIX"
        badge_color = "#10b981"
        summary = "No known chemical, physical, or biological antagonisms detected. Products appear compatible for co-application."
        
    # Calculate recommended WALES Sequence
    def get_wales_rank(prod_name: str) -> int:
        prod_meta = TREATMENT_DATABASE.get(prod_name, {})
        form = prod_meta.get("form", "liquid")
        name_lower = prod_name.lower()
        if form == "dry" or any(w in name_lower for w in ["wp", "wdg", "powder", "salt", "nitrate", "bicarbonate"]):
            return 1  # W: Wettable powders & drys
        if "oil" in name_lower:
            return 4  # E: Emulsifiable concentrates / oils
        if "adjuvant" in name_lower or "soap" in name_lower or "surfactant" in name_lower:
            return 5  # S: Surfactants
        return 3      # L: Liquid flowables & suspensions
        
    wales_sorted = sorted(selected_products, key=get_wales_rank)
    
    return {
        "status": status,
        "badge_color": badge_color,
        "summary": summary,
        "conflicts": conflicts,
        "wales_order": wales_sorted,
        "overall_safety": "DANGER" if has_critical else ("WARNING" if has_high else ("CAUTION" if conflicts else "SAFE"))
    }


def get_jar_test_protocol(selected_products: list = None, total_carrier_liters: float = 20.0) -> dict:
    """
    Generates a proportional 1-pint (500 ml) Mason Jar Compatibility Test protocol
    tailored to the user's specific product combination.
    """
    if selected_products is None:
        selected_products = []
    jar_volume_ml = 500.0  # 1 pint mason jar
    
    jar_dosages = []
    for prod_name in selected_products:
        meta = TREATMENT_DATABASE.get(prod_name, {})
        is_liquid = meta.get("form") == "liquid"
        if is_liquid:
            rate = meta.get("rate_per_liter_ml", 5.0)
            jar_amount = round(rate * 0.5, 2)  # for 500 ml
            unit = "ml"
            meas = f"{round(jar_amount * 20, 0):.0f} drops (~{jar_amount} ml with syringe)"
        else:
            rate = meta.get("rate_per_liter_g", 3.0)
            jar_amount = round(rate * 0.5, 2)  # for 500 ml
            unit = "grams"
            meas = f"~{jar_amount} g (approx 1/4 teaspoon)"
            
        jar_dosages.append({
            "product": prod_name,
            "jar_amount": jar_amount,
            "unit": unit,
            "household_measure": meas
        })
        
    steps = [
        "1. Fill a clean, transparent 1-pint (500 ml) glass mason jar with **400 ml of carrier water** from the intended spray source.",
        "2. Add each product in the prescribed **WALES order**, gently capping and inverting the jar 10 times after each product is added.",
        "3. Once all inputs are added, top off with water to the 500 ml mark, seal tightly, and invert 15 times.",
        "4. Allow the sealed jar to stand completely undisturbed for **15 to 30 minutes** in ambient shade."
    ]
    
    evaluation = {
        "pass_criteria": "Uniform milky or translucent suspension with no precipitate, clumps, or heat generation. If slight settling occurs, liquid redistributes smoothly with 2 gentle inversions.",
        "fail_criteria": "Curdling ('cottage cheese'), oil slick floating on top, gritty sediment at jar bottom that resists re-suspension, or the jar feeling noticeably warm to touch."
    }
    
    return {
        "jar_volume_ml": jar_volume_ml,
        "jar_dosages": jar_dosages,
        "protocol_steps": steps,
        "evaluation": evaluation
    }

