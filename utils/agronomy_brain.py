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
    "🌽 How do I prevent Common Rust on sweet corn?",
    "⚖️ What is the dosage and dilution rate for Neem Oil?"
]

def query_agronomy_brain(user_query: str) -> str:
    """
    Analyzes user query and returns a structured, authoritative agronomic response
    from the localized plant pathology and chemical formulation databases.
    """
    q = user_query.lower().strip()
    
    # 1. Check for Mixing / Compatibility questions
    if any(k in q for k in ["mix", "compatible", "tank mix", "combine", "together"]):
        if ("sulfur" in q and "oil" in q) or ("neem" in q and "sulfur" in q):
            return """### 🚫 Critical Phytotoxicity Alert: Sulfur + Oil Incompatibility
**Under no circumstances should Sulfur be tank-mixed or applied within 30 days of horticultural or Neem oils.**

- **Reaction:** The combination of elemental sulfur and oil penetrates plant epidermal cells, causing severe foliar burn, leaf scorch, and defoliation.
- **Safety Window:** Maintain a strict minimum **30-day interval** between oil and sulfur applications.
- **Temperature Guard:** Never apply sulfur when ambient field temperatures exceed **28°C (83°F)**."""

        if "wales" in q or "order" in q or "sequence" in q:
            return """### 🧪 The Universal WALES Tank Mixing Protocol
When combining multiple agricultural inputs in a spray tank, always follow the **WALES** sequence to avoid coagulation:

1. **W - Wettable Powders & Dry Flowables (WP, WDG, DF):** Pre-slurry in a bucket of warm water before pouring into a half-filled tank.
2. **A - Agitation:** Turn on mechanical or bypass agitation to ensure complete suspension.
3. **L - Liquid Flowables & Suspensions (SC, F):** Pour liquid concentrates slowly into the swirling tank.
4. **E - Emulsifiable Concentrates (EC, Neem Oil):** Add oil-based solutions next.
5. **S - Surfactants, Adjuvants & Water-Soluble Concentrates (SL):** Top off with spreading agents and remaining water carrier."""

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
