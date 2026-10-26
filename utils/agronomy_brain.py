"""
PlantGuard AI - Offline Agronomic Brain & Expert Agronomist Engine
Provides high-accuracy agricultural intelligence and answers farmers' questions
even when no external Gemini API key is configured.
"""

import re
from utils.disease_expert import DISEASE_DATABASE
from utils.dosage_calculator import TREATMENT_DATABASE

SUGGESTED_QUESTIONS = [
    "🍅 How do I treat Early Blight on Tomatoes organically?",
    "🍎 What is the best spray schedule for Apple Scab?",
    "🥔 How do I tell Late Blight apart from Early Blight?",
    "🧪 Can I mix Copper Fungicide with Neem Oil?",
    "💧 What causes blossom end rot and yellowing leaves?",
    "🌾 How does soil pH affect nutrient bioavailability and lockout?",
    "⚗️ Can I mix Calcium Nitrate with Epsom Salt or NPK?",
    "🧪 How do I perform a 1-pint mason jar compatibility test?",
    "🌽 How do I prevent Common Rust on sweet corn?",
    "⚖️ What is the dosage and dilution rate for Neem Oil?"
]

def query_agronomy_brain(user_query: str) -> str:
    """
    Analyzes user query and returns a structured, authoritative agronomic response
    from the localized plant pathology and chemical formulation databases.
    """
    q = user_query.lower().strip()
    
    # 1. Check for Mixing / Compatibility / Jar Test questions
    if any(k in q for k in ["mix", "compatible", "tank mix", "combine", "together", "jar test", "precipitation", "coagulate"]):
        if "jar test" in q or "mason jar" in q or "test compatibility" in q:
            return """### 🧪 The 1-Pint (500 ml) Mason Jar Compatibility Test Protocol
Before pouring thousands of dollars of crop protection chemicals into a 500-gallon tractor tank, always conduct a benchtop jar test:

1. **Carrier Water:** Add 400 ml of carrier water from your actual spray source to a clean 500 ml glass jar.
2. **Sequential Dosing (WALES):** Add proportional amounts of each product in sequence (Dry powders $\\to$ Agitation $\\to$ Liquid suspensions $\\to$ Emulsifiable oils $\\to$ Surfactants).
3. **Inversion & Rest:** Invert 10 times after each product. Top to 500 ml, invert 15 times, and let rest undisturbed for **15-30 minutes**.
4. **Pass Criteria:** Uniform milky suspension with no clumping or heat release. Any minor settling redisperses with 2 gentle inversions.
5. **Fail Criteria:** Curdling ("cottage cheese"), oily separation ring at top, gritty heavy sludge at jar bottom, or spontaneous exothermic heating."""

        if ("sulfur" in q and "oil" in q) or ("neem" in q and "sulfur" in q):
            return """### 🚫 Critical Phytotoxicity Alert: Sulfur + Oil Incompatibility
**Under no circumstances should Sulfur be tank-mixed or applied within 30 days of horticultural or Neem oils.**

- **Reaction:** The combination of elemental sulfur and oil penetrates plant epidermal cells, causing severe foliar burn, leaf scorch, and defoliation.
- **Safety Window:** Maintain a strict minimum **30-day interval** between oil and sulfur applications.
- **Temperature Guard:** Never apply sulfur when ambient field temperatures exceed **28°C (83°F)**."""

        if ("calcium" in q and ("sulfate" in q or "epsom" in q or "sulfur" in q)) or ("calcium" in q and "phosphate" in q) or ("calcium" in q and "npk" in q):
            return """### ⚠️ High Chemical Precipitation Hazard: Insoluble Salt Formation
**Do NOT mix Calcium Nitrate with Phosphates (NPK) or Sulfates (Epsom Salt / Potassium Sulfate) in the same concentrated tank.**

- **Reaction:** Free calcium ions ($Ca^{2+}$) bind instantaneously with sulfate ($SO_4^{2-}$) to form **Gypsum ($CaSO_4$)**, or with phosphates to form insoluble **Dicalcium Phosphate ($CaHPO_4$)**.
- **Physical Result:** Rapid precipitation of heavy, gritty white sludge that irreversibly clogs nozzle screens and line filters, with zero foliar bioavailability.
- **Remediation:** Apply Calcium Nitrate separately, or use a two-tank injection system (Tank A: Calcium/Iron; Tank B: Phosphates/Sulfates)."""

        if ("copper" in q and "bacillus" in q) or ("copper" in q and "subtilis" in q) or ("bactericide" in q and "biological" in q):
            return """### 🚫 Biological Deactivation Hazard: Copper + Bio-Inoculants
**Do NOT mix Copper Fungicides (Fixed Copper, Copper Octanoate, Bordeaux mixture) with living biological biofungicides (*Bacillus subtilis*, *Trichoderma*).**

- **Reaction:** Copper ions are non-selective broad-spectrum cellular biocides that rupture bacterial cell walls and denature proteins, immediately killing the beneficial living microorganisms upon contact.
- **Safety Interval:** Apply biologicals at least **7 to 10 days** after a copper treatment has weathered off the canopy foliage."""

        if "wales" in q or "order" in q or "sequence" in q:
            return """### 🧪 The Universal WALES Tank Mixing Protocol
When combining multiple agricultural inputs in a spray tank, always follow the **WALES** sequence to avoid coagulation:

1. **W - Wettable Powders & Dry Flowables (WP, WDG, DF):** Pre-slurry in a bucket of warm water before pouring into a half-filled tank.
2. **A - Agitation:** Turn on mechanical or bypass agitation to ensure complete suspension.
3. **L - Liquid Flowables & Suspensions (SC, F):** Pour liquid concentrates slowly into the swirling tank.
4. **E - Emulsifiable Concentrates (EC, Neem Oil):** Add oil-based solutions next.
5. **S - Surfactants, Adjuvants & Water-Soluble Concentrates (SL):** Top off with spreading agents and remaining water carrier."""

    # 2. Check for Soil pH Bioavailability & Nutrient Lockout questions
    if any(k in q for k in ["soil ph", "ph lockout", "acidic soil", "alkaline soil", "liming", "soil acidity", "truog"]):
        return """### 🧪 Soil pH Bioavailability Dynamics & Truog Lockout Curves
Soil pH is the master variable governing root-zone chemical solubility and ion exchange capacity.

#### 1. The Agronomic Sweet Spot (pH 6.2 - 6.8):
- All 10 macro- and micro-nutrients achieve simultaneous optimal solubility without phytotoxicity.

#### 2. Acidic Soil Lockout (pH < 6.0):
- **Phosphorus (P) Fixation:** Insoluble aluminum and iron phosphates ($AlPO_4$, $FePO_4$) precipitate out, turning leaves dark dull green or purple.
- **Base Cation Leaching:** Calcium ($Ca^{2+}$) and Magnesium ($Mg^{2+}$) are displaced by hydrogen and toxic aluminum ($Al^{3+}$) ions.
- **Remedy:** Broadcast Agricultural Dolomitic Limestone (providing both calcium carbonate and magnesium carbonate) at 20-50 lbs per 1,000 sq. ft. depending on soil texture.

#### 3. Alkaline Soil Lockout (pH > 7.3):
- **Micronutrient Precipitation:** Iron, Zinc, Manganese, and Boron precipitate into insoluble hydroxides, causing acute interveinal chlorosis (yellow leaves with green veins).
- **Phosphorus Binding:** Forms insoluble Tricalcium Phosphate ($Ca_3(PO_4)_2$).
- **Remedy:** Apply Elemental Agricultural Sulfur (10-30 lbs / 1,000 sq ft) or foliar chelated micronutrients (Fe-EDDHA, Zn-EDTA)."""

    # 2. Check for Specific Disease Inquiries
    matched_disease = None
    for key, info in DISEASE_DATABASE.items():
        title = info.get("title", "").lower()
        crop = info.get("crop", "").lower()
        clean_key = key.lower().replace("_", " ")
        
        # Check match with disease name or key parts
        disease_words = [w for w in clean_key.split() if w not in ["healthy", "apple", "tomato", "potato", "corn", "pepper", "bell"]]
        if any(w in q for w in disease_words if len(w) > 4) or title in q:
            # If user also specified a crop, ensure crop matches
            if crop in q or not any(c in q for c in ["apple", "tomato", "potato", "corn", "pepper"]):
                matched_disease = (key, info)
                break

    if matched_disease:
        k, d = matched_disease
        immed_bullets = "\n".join([f"- {a}" for a in d.get("immediate_actions", [])[:3]])
        org_bullets = "\n".join([f"- {o}" for o in d.get("organic_treatment", [])[:3]])
        chem_bullets = "\n".join([f"- {c}" for c in d.get("chemical_treatment", [])[:3]])
        
        return f"""### 🌿 Agronomic Diagnosis: **{d.get('title', k)}**
**Host Crop:** {d.get('crop')} | **Category:** `{d.get('category')}` | **Severity Risk:** {d.get('severity_level')}

#### 🔍 Pathology & Symptoms
{d.get('description')}
- **Visual Indicators:** {d.get('symptoms')}
- **Underlying Triggers:** {d.get('causes')}

#### ⚡ Immediate First-Aid Containment
{immed_bullets}

#### 🌿 Organic & Eco-Friendly Protocol
{org_bullets}

#### 🧪 Conventional / Chemical Countermeasures
{chem_bullets}

#### 🛡️ Long-term Cultural Prevention
{d.get('nutrition_and_care', 'Ensure balanced soil fertility and proper canopy spacing.')}

*(Need exact tank mix calculations? Switch to **⚖️ Spray & Dosage Calculator** in the sidebar.)*"""

    # 3. Check for Product / Chemical Formulation Queries
    matched_product = None
    for prod_name, details in TREATMENT_DATABASE.items():
        name_lower = prod_name.lower()
        tokens = [t for t in name_lower.split() if len(t) > 3]
        if any(t in q for t in tokens):
            matched_product = (prod_name, details)
            break

    if matched_product:
        p_name, p_info = matched_product
        rate_str = f"{p_info.get('rate_per_liter_ml', p_info.get('rate_per_liter_g'))} per Liter of water"
        return f"""### 🧪 Formulation Profile: **{p_name}**
**Classification:** `{p_info.get('type')}` | **Form:** {p_info.get('form', 'Liquid').capitalize()}

- **Standard Field Dilution:** **{rate_str}**
- **Approved Target Crops:** {', '.join(p_info.get('crops', []))}
- **Target Pathogens:** {', '.join(p_info.get('recommended_for', []))}
- **Pre-Harvest Interval (PHI):** `{p_info.get('phi_days')} Day(s)` (Wait time between spray and harvest)
- **Field Re-Entry Interval (REI):** `{p_info.get('rei_hours')} Hours`
- **Optimal Application Timing:** {p_info.get('best_time')}
- **Adjuvant Requirement:** {p_info.get('adjuvant', 'None needed')}
- **Mixing Instruction:** {p_info.get('mixing_notes', 'Stir thoroughly in clean carrier water.')}
- **Required PPE:** {p_info.get('ppe')}"""

    # 4. Check for Blossom End Rot / Calcium Deficiency
    if "blossom end rot" in q or "calcium" in q or "black bottom" in q:
        return """### 🍅 Blossom End Rot (Physiological Calcium Disorder)
**Blossom End Rot is NOT a fungal or bacterial pathogen; it is a physiological calcium (Ca) deficiency in developing fruit tissue.**

#### Primary Causes:
1. **Inconsistent Soil Moisture:** Fluctuations between severe drying and waterlogging disrupt the plant's transpirational pull of calcium from roots to fruit tips.
2. **Excessive Nitrogen (N):** High ammonium fertilizers trigger rapid foliar growth which hogs available calcium away from the fruit.

#### Corrective Action Plan:
- **Stabilize Irrigation:** Implement automated drip irrigation to maintain uniform moisture.
- **Mulch the Soil:** Apply a 2-3 inch organic mulch layer (straw, shredded bark) to minimize evaporation.
- **Foliar Calcium Spray:** Apply 5 ml/L of Calcium Chloride or chelated calcium during early fruit development.
- **Adjust Soil pH:** Calcium uptake is optimal when soil pH is maintained between **6.2 and 6.8**."""

    # 5. Check for General Watering, Spacing & Cultural Prevention
    if "water" in q or "irrigation" in q or "sprinkler" in q:
        return """### 💧 Sustainable Irrigation Guidelines for Foliar Disease Control
Overhead sprinkler watering is the **#1 catalyst** for fungal and bacterial disease outbreaks in garden and orchard crops.

1. **Adopt Ground-Level Drip Irrigation:** Keeping leaf surfaces dry prevents fungal spores (e.g. *Phytophthora*, *Septoria*, *Alternaria*) from germinating.
2. **Water Exclusively at Dawn:** If overhead watering is unavoidable, irrigate between **5:00 AM and 8:00 AM** so sunlight rapidly dries the foliage.
3. **Deep, Infrequent Soaking:** Encourage deep root architecture by watering deeply 2-3 times per week rather than shallow daily sprinkles.
4. **Soil Splash Protection:** Install clean straw or plastic mulch beneath crop rows to prevent rain droplets from splashing fungal spores from the soil onto lower leaves."""

    # 6. Fallback General Agronomic Guidance
    return f"""### 🌿 PlantGuard AI Agronomist
Thank you for your question: *"{user_query}"*

Here are key agronomic recommendations from our field pathology protocols:
- **Disease Prevention:** Maintain adequate plant spacing (18–24 inches for nightshades, 4–6 feet for orchard trees) to promote airflow and rapid canopy drying.
- **Sanitation:** Always disinfect pruning tools with a 70% alcohol solution between cuts to halt transmission of bacterial and viral pathogens.
- **Protective Sprays:** Bio-fungicides like *Bacillus subtilis* or cold-pressed 70% Neem Oil should be applied preventatively before rain events.
- **Fertility Balance:** Avoid heavy nitrogen applications late in the season; boost potassium (K) and silica to reinforce cell wall thickness against fungal hyphae penetration.

💡 *Tip: For specific chemical prescriptions, check our **📖 Care & Treatment Handbook** or enter your Google Gemini API key in the sidebar for unlimited open-ended conversations!*"""
