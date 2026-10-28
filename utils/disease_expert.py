"""
PlantGuard AI - Comprehensive Plant Disease & Care Guide Expert System
Provides detailed diagnosis insights, treatment protocols (organic & chemical),
immediate actions, and preventive agronomic regimens for all supported crops.
"""

DISEASE_DATABASE = {
    # ------------------ APPLE ------------------
    "Apple___Apple_scab": {
        "title": "Apple Scab (Venturia inaequalis)",
        "crop": "Apple",
        "category": "Fungal Infection",
        "severity_level": "High",
        "urgency": "🚨 Immediate Action Required (Within 24-48 hours)",
        "description": "Apple scab is a destructive fungal infection caused by Venturia inaequalis that targets leaves, blossoms, and developing fruit, leading to premature defoliation and severe yield loss.",
        "symptoms": "Olive-green to velvety dark brown/black lesions on the upper leaf surface, yellowing foliage, twisted or puckered leaves, and corky scab lesions on fruit.",
        "causes": "Overwintering fungal spores on fallen leaves released during cool (13-24°C / 55-75°F), wet spring weather. Prolonged leaf wetness for 6+ hours accelerates germination.",
        "immediate_actions": [
            "Prune and discard infected leaves and twigs using sterilized shears (disinfect with 70% isopropyl alcohol).",
            "Rake up and destroy or compost deeply all fallen leaves under the tree canopy to break the spore cycle.",
            "Avoid overhead sprinkler watering; keep tree foliage as dry as possible."
        ],
        "organic_treatment": [
            "Neem Oil (70% extract): Apply 2 tbsp per gallon of water in the early morning or evening every 7-10 days.",
            "Sulfur or Copper Soap Fungicide: Apply at tight cluster stage through petal fall.",
            "Biofungicide (Bacillus subtilis): Apply as a preventive protective biofilm before rainy spells."
        ],
        "chemical_treatment": [
            "Captan 50 WP: Apply 1.5 to 2.5 lbs per 100 gallons at pink bud and petal fall stages.",
            "Myclobutanil (Immunox): Systemic curative action; effective within 72-96 hours of an infection event.",
            "Mancozeb: Multi-site protectant; rotate with single-site fungicides to prevent resistance."
        ],
        "preventive_measures": [
            "Plant scab-resistant cultivars such as Liberty, Freedom, Enterprise, and GoldRush.",
            "Open the tree canopy with winter pruning to maximize air circulation and sunlight penetration.",
            "Apply a 5% urea spray to fallen autumn foliage to accelerate leaf breakdown before spring."
        ],
        "nutrition_and_care": "Avoid excessive nitrogen fertilizers in early spring as lush succulent growth is highly vulnerable to infection. Maintain balanced potassium (K) and calcium (Ca) to reinforce leaf cuticle integrity."
    },
    "Apple___Black_rot": {
        "title": "Black Rot & Frogeye Leaf Spot (Botryosphaeria obtusa)",
        "crop": "Apple",
        "category": "Fungal Infection",
        "severity_level": "High",
        "urgency": "🚨 High Priority Action Needed",
        "description": "Black rot affects apple leaves (producing 'frogeye' spots), cankers on woody branches, and rot on ripening fruit caused by the pathogen Botryosphaeria obtusa.",
        "symptoms": "Purple circular specks on leaves expanding into brown lesions with purple borders ('frogeye pattern'), dark sunken bark cankers, and firm black mummified fruit.",
        "causes": "Fungus enters through wounds, insect punctures, or fire blight strikes; spores disperse via rain splash during warm (20-27°C / 68-80°F) rainy periods.",
        "immediate_actions": [
            "Prune out dead wood, cankered limbs, and mummified fruit hanging on branches at least 6-8 inches below the lesion.",
            "Burn or bury infected wood off-site; do not leave dead brush near the orchard.",
            "Disinfect cutting tools between every cut with bleach solution (1:9) or 70% alcohol."
        ],
        "organic_treatment": [
            "Liquid Copper Fungicide: Spray thoroughly during dormant and early bud break stages.",
            "Sulfur Dust or Spray: Protective broad-spectrum application before wet weather spells."
        ],
        "chemical_treatment": [
            "Captan + Thiophanate-methyl: Synergistic tank mix offering protective and kickback suppression.",
            "Flint Extra (Trifloxystrobin) or Pristine (Pyraclostrobin + Boscalid): Effective during early cover sprays."
        ],
        "preventive_measures": [
            "Keep trees stress-free with regular mulching and proper irrigation.",
            "Avoid mechanical trunk or branch injuries during pruning and mowing.",
            "Maintain strict winter sanitation by stripping old fruit mummies."
        ],
        "nutrition_and_care": "Ensure soil pH between 6.0 and 6.8. Apply balanced compost in late winter to maintain vigor without excessive vegetative flushes."
    },
    "Apple___Cedar_apple_rust": {
        "title": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "crop": "Apple",
        "category": "Fungal Rust (Heteroecious)",
        "severity_level": "Moderate",
        "urgency": "⚠️ Action Required within 3-5 days",
        "description": "A heteroecious rust fungus that requires both apple trees and Eastern red cedar (Juniperus virginiana) to complete its complex 2-year life cycle.",
        "symptoms": "Bright yellow-orange or reddish-orange circular spots on upper leaf surfaces; later developing tiny tube-like fungal aecia structures on leaf undersides.",
        "causes": "Windblown basidiospores originating from gelatinous orange telial horns on nearby cedar galls during warm spring rains.",
        "immediate_actions": [
            "Inspect neighboring juniper or cedar trees within 1-2 miles and remove visible galls if accessible.",
            "Remove heavily spotted leaves if the tree is young to reduce stress.",
            "Monitor nearby cedar hedges during wet spring days."
        ],
        "organic_treatment": [
            "Copper Octanoate (Soap): Spray beginning at tight cluster stage through bloom.",
            "Sulfur-based fungicides: Apply every 7 days during spore flight periods."
        ],
        "chemical_treatment": [
            "Myclobutanil (Immunox): Outstanding curative and protective systemic control.",
            "Mancozeb or Propiconazole: Apply at pink bud stage and repeat 10 days later."
        ],
        "preventive_measures": [
            "Do not plant susceptible apple trees within 500 meters of cedar or juniper hedges.",
            "Select resistant cultivars such as Dayton, Freedom, Enterprise, Redfree, or William's Pride.",
            "Prune out round woody galls from nearby juniper bushes in late winter."
        ],
        "nutrition_and_care": "Maintain adequate tree hydration and foliar zinc/boron nutrition to support rapid leaf recovery and fruit set."
    },
    "Apple___healthy": {
        "title": "Healthy Apple Foliage (Malus domestica)",
        "crop": "Apple",
        "category": "Healthy Plant",
        "severity_level": "None",
        "urgency": "🟢 Optimal Condition - Routine Care",
        "description": "Your apple foliage exhibits vigorous chlorophyll density, healthy cuticle barriers, and zero signs of pathogen infection or pest distress.",
        "symptoms": "Crisp green leaves, smooth surface, uniform vein structure, no chlorosis, no spotting or fungal lesions.",
        "causes": "Proper balance of moisture, nutrient availability, adequate spacing, and good garden sanitation.",
        "immediate_actions": [
            "No disease interventions needed.",
            "Continue regular scouting every 7-10 days, inspecting leaf undersides."
        ],
        "organic_treatment": [
            "Maintain routine preventive spraying with compost tea or dilute kelp extract to strengthen cell walls."
        ],
        "chemical_treatment": [
            "No chemical fungicides or bactericides required."
        ],
        "preventive_measures": [
            "Maintain 3-4 inches of wood chip mulch around the dripline (keeping 6 inches away from the trunk).",
            "Deep soak weekly during dry dry spells (approx. 1 inch of water per week).",
            "Monitor traps for codling moth and apple maggot."
        ],
        "nutrition_and_care": "Test soil every 2 years. Topdress with slow-release organic balanced fertilizer (10-10-10 or aged compost) before bud break."
    },

    # ------------------ CORN (MAIZE) ------------------
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "title": "Gray Leaf Spot (Cercospora zeae-maydis)",
        "crop": "Corn (Maize)",
        "category": "Fungal Infection",
        "severity_level": "High",
        "urgency": "🚨 High Priority Treatment",
        "description": "Gray leaf spot is one of the most yield-limiting foliar diseases of maize worldwide, causing rapid leaf tissue necrosis and stalk lodging.",
        "symptoms": "Tan or grayish rectangular lesions restricted by leaf veins (giving sharp parallel edges), coalescing to create blighted leaf canopy.",
        "causes": "Overwintering debris on soil surface, high relative humidity (>90%), and temperatures between 25-32°C (77-90°F).",
        "immediate_actions": [
            "Evaluate crop stage: if disease reaches ear leaf prior to blister stage, treatment is critical.",
            "Remove lower infected leaves in small garden plots to improve air circulation."
        ],
        "organic_treatment": [
            "Foliar spray with Bacillus amyloliquefaciens or Trichoderma viride strains.",
            "Bio-copper formulations applied early in the vegetative phase."
        ],
        "chemical_treatment": [
            "Azoxystrobin + Difenoconazole (Quadris Top): Strong systemic preventive and curative action.",
            "Pyraclostrobin (Headline AMP): Apply at VT-R1 (tasseling to silking) stage for maximum yield protection."
        ],
        "preventive_measures": [
            "Crop rotation away from corn for at least 1-2 years.",
            "Tillage to bury infected crop residue in fields with severe history.",
            "Select maize hybrids with high Gray Leaf Spot tolerance ratings."
        ],
        "nutrition_and_care": "Avoid excessive plant density. Ensure balanced nitrogen-to-potassium ratio; potassium deficiency severely worsens stalk lodging."
    },
    "Corn_(maize)___Common_rust_": {
        "title": "Common Rust (Puccinia sorghi)",
        "crop": "Corn (Maize)",
        "category": "Fungal Rust",
        "severity_level": "Moderate",
        "urgency": "⚠️ Monitor and Treat as Needed",
        "description": "Common rust produces airborne spores that can travel long distances, infecting corn leaves during moderate temperatures and moist weather.",
        "symptoms": "Golden brown to dark reddish-cinnamon pustules (uredinia) scattered across both upper and lower leaf surfaces, erupting through leaf epidermis.",
        "causes": "Puccinia sorghi spores blown northward by southern storm fronts, favored by cool to moderate temperatures (16-25°C) and 100% relative humidity.",
        "immediate_actions": [
            "Scout weekly to calculate rust pustule density on mid-canopy leaves.",
            "Avoid overhead irrigation in late afternoon to reduce leaf wetness duration."
        ],
        "organic_treatment": [
            "Wettable sulfur spray applied in early morning.",
            "Neem-based botanical formulations to suppress spore germination."
        ],
        "chemical_treatment": [
            "Tebuconazole or Propiconazole (Tilt): Highly effective triazole fungicides when applied at first sign of pustules.",
            "Strobilurin fungicides (e.g., Headline, Quadris)."
        ],
        "preventive_measures": [
            "Plant resistant sweet corn or field corn hybrids containing the Rp1-D resistance gene.",
            "Plant early in the season to avoid peak spore migration waves."
        ],
        "nutrition_and_care": "Maintain adequate phosphorus for root vigor and avoid excessive vegetative nitrogen surges."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "title": "Northern Corn Leaf Blight (Exserohilum turcicum)",
        "crop": "Corn (Maize)",
        "category": "Fungal Blight",
        "severity_level": "High",
        "urgency": "🚨 Immediate Action Required",
        "description": "A serious fungal pathogen causing massive cigar-shaped necrotic lesions that drastically reduce photosynthesis and grain filling.",
        "symptoms": "Long, elliptical grayish-green to tan lesions (cigar-shaped, 1-6 inches long) that develop dark olivaceous spore mats in damp weather.",
        "causes": "Fungus overwinters in corn residue; favored by moderate temperatures (18-27°C / 65-80°F) and extended dew periods (6-12 hours).",
        "immediate_actions": [
            "Strip severely blighted bottom leaves if infection is confined to lower canopy.",
            "Prevent splash dispersal by switching to drip or furrow irrigation."
        ],
        "organic_treatment": [
            "Foliar sprays of potassium bicarbonate (3 tbsp/gal) combined with horticultural oil.",
            "Copper hydroxide formulations applied at first sighting."
        ],
        "chemical_treatment": [
            "Trivapro (Benzovindiflupyr + Azoxystrobin + Propiconazole): Long-lasting systemic protection.",
            "Delaro 325 SC (Prothioconazole + Trifloxystrobin): Broad spectrum control applied at tassel stage."
        ],
        "preventive_measures": [
            "Plant NCLB-resistant seed hybrids with Ht genes.",
            "Practice minimum 2-year crop rotation with non-host crops like soybeans, wheat, or alfalfa.",
            "Shred and incorporate corn stubble post-harvest to accelerate decomposition."
        ],
        "nutrition_and_care": "Maintain optimum fertility according to soil tests; avoid nutrient stress during critical silking periods."
    },
    "Corn_(maize)___healthy": {
        "title": "Healthy Corn Foliage (Zea mays)",
        "crop": "Corn (Maize)",
        "category": "Healthy Plant",
        "severity_level": "None",
        "urgency": "🟢 Optimal Condition - Routine Care",
        "description": "Corn leaves display excellent dark green turgidity, clean leaf sheaths, and robust photosynthetic capacity.",
        "symptoms": "Deep emerald green arching leaves, smooth surfaces, robust thick stalks, zero lesion patches.",
        "causes": "Adequate nitrogen, moisture, sunlight, and proper row spacing.",
        "immediate_actions": [
            "Continue standard agronomic monitoring.",
            "Check soil moisture at 4-6 inches depth."
        ],
        "organic_treatment": [
            "Apply fish hydrolysate or seaweed extract as foliar feed during knee-high (V6) stage."
        ],
        "chemical_treatment": ["None required."],
        "preventive_measures": [
            "Maintain weed-free rows to prevent pest harboring.",
            "Ensure consistent moisture supply during tasseling and silking (critical water need period)."
        ],
        "nutrition_and_care": "Side-dress nitrogen (e.g., urea, feather meal, or blood meal) when plants reach 12-18 inches tall for maximum ear development."
    },

    # ------------------ BELL PEPPER ------------------
    "Pepper__bell___Bacterial_spot": {
        "title": "Bacterial Spot (Xanthomonas euvesicatoria)",
        "crop": "Bell Pepper",
        "category": "Bacterial Infection",
        "severity_level": "High",
        "urgency": "🚨 Immediate Containment Required",
        "description": "Bacterial spot is one of the most destructive diseases of peppers, causing severe defoliation, sunscald on unprotected fruit, and unmarketable scabby lesions.",
        "symptoms": "Small, water-soaked blister-like spots on leaves turning dark brown with yellow halos; leaves turn yellow and drop prematurely; raised rough spots on peppers.",
        "causes": "Bacteria introduced on contaminated seeds/transplants; spreads rapidly via splashing water, working among wet plants, and warm humid weather (24-30°C / 75-86°F).",
        "immediate_actions": [
            "NEVER work among pepper plants while leaves are wet from dew or rain.",
            "Prune out early infected lower leaves; bag and discard them off-site.",
            "Disinfect stakes, cages, and pruning tools with 10% bleach solution."
        ],
        "organic_treatment": [
            "Fixed Copper Fungicide (Copper Hydroxide): Spray every 7-10 days, coating undersides of leaves thoroughly.",
            "Bacillus subtilis / Bacillus amyloliquefaciens: Biobactericides that colonize plant tissue and suppress bacterial multiplication."
        ],
        "chemical_treatment": [
            "Copper + Mancozeb tank mix: Synergistic standard that improves bactericidal efficacy against copper-tolerant strains.",
            "Acibenzolar-S-methyl (Actigard): Plant defense activator that triggers systemic acquired resistance."
        ],
        "preventive_measures": [
            "Always purchase certified disease-free, hot-water treated seeds or resistant pepper cultivars.",
            "Implement a 2-3 year rotation away from solanaceous crops (tomatoes, peppers, potatoes, eggplants).",
            "Use drip irrigation under black plastic mulch to eliminate soil splashing."
        ],
        "nutrition_and_care": "Ensure adequate calcium and magnesium in the soil to prevent blossom end rot while plants are coping with bacterial stress."
    },
    "Pepper__bell___healthy": {
        "title": "Healthy Bell Pepper (Capsicum annuum)",
        "crop": "Bell Pepper",
        "category": "Healthy Plant",
        "severity_level": "None",
        "urgency": "🟢 Optimal Condition - Routine Care",
        "description": "Leaves are vibrant, lush, free from bacterial blemishes, and exhibiting strong apical growth with active flower buds.",
        "symptoms": "Glossy green leaves, firm stems, upright habit, no curling or chlorosis.",
        "causes": "Proper sunlight (6-8 hours direct daily), well-drained fertile soil, and steady warm temperatures.",
        "immediate_actions": ["No disease treatment needed."],
        "organic_treatment": ["Foliar spray with Epsom salt solution (1 tbsp/gal water) during flowering to boost magnesium for pepper set."],
        "chemical_treatment": ["None required."],
        "preventive_measures": [
            "Maintain organic straw or pine needle mulch to stabilize root temperature and conserve moisture.",
            "Support plants with cages or stakes before peppers become heavy."
        ],
        "nutrition_and_care": "Use balanced fertilizer (e.g. 5-10-10) with lower nitrogen once flowering commences to stimulate abundant fruit set rather than leaf overgrowth."
    },

    # ------------------ POTATO ------------------
    "Potato___Early_blight": {
        "title": "Early Blight (Alternaria solani)",
        "crop": "Potato",
        "category": "Fungal Infection",
        "severity_level": "Moderate to High",
        "urgency": "⚠️ Urgent Intervention within 48 hours",
        "description": "Early blight is a prevalent fungal disease that attacks older foliage first, progressing upward and significantly reducing tuber size and yield.",
        "symptoms": "Dark brown to black spots with characteristic concentric rings forming a 'target board' or 'bullseye' appearance, surrounded by yellow chlorotic halos.",
        "causes": "Alternaria solani fungus survives in crop residue and volunteer potatoes; thrives in warm temperatures (24-29°C / 75-85°F) with alternating wet and dry cycles.",
        "immediate_actions": [
            "Prune off heavily spotted lower leaves that touch the soil.",
            "Do not overhead water; switch immediately to soaker hoses or drip tape.",
            "Sterilize garden shears between plants."
        ],
        "organic_treatment": [
            "Copper Sulfate / Bordeaux Mixture: Apply proactively to coat all leaf surfaces.",
            "Potassium Bicarbonate spray: Raises leaf surface pH, halting fungal spore germination.",
            "Biofungicide Serenade (Bacillus subtilis): Applied every 7 days as an eco-friendly preventive."
        ],
        "chemical_treatment": [
            "Chlorothalonil (Daconil): Broad spectrum protectant fungicide applied at 7-14 day intervals.",
            "Mancozeb (Dithane): Excellent preventive contact fungicide.",
            "Azoxystrobin or Boscalid: Systemic options for active infection outbreaks."
        ],
        "preventive_measures": [
            "Plant certified disease-free seed tubers from trusted suppliers.",
            "Maintain 3-4 year crop rotation away from solanaceous species.",
            "Hill up potatoes well to protect developing tubers from washing spores."
        ],
        "nutrition_and_care": "Keep potato vines vigorously growing with adequate potassium and nitrogen; plants under nutritional stress are exponentially more susceptible to early blight."
    },
    "Potato___Late_blight": {
        "title": "Late Blight (Phytophthora infestans)",
        "crop": "Potato",
        "category": "Oomycete / Water Mold",
        "severity_level": "Critical",
        "urgency": "🚨 EMERGENCY ACTION REQUIRED - HIGHLY CONTAGIOUS",
        "description": "Late blight is the infamous pathogen that caused the Irish Potato Famine. It can obliterate entire potato and tomato fields within days under favorable cool, wet conditions.",
        "symptoms": "Large, irregular water-soaked pale green or dark brown lesions on leaves; white velvety downy mold on the undersides of leaves during high humidity; rapid vine collapse; tubers turn brown with dry corky rot.",
        "causes": "Oomycete Phytophthora infestans spores spread over miles by wind; thrives in cool (15-21°C / 60-70°F) and saturated humid conditions (>90% RH).",
        "immediate_actions": [
            "Act within hours: If only 1-2 plants are infected, pull the entire plant, bag it immediately, and dispose of in municipal trash (NEVER compost).",
            "Cut vines at soil level if tubers are mature to prevent spores from washing into soil.",
            "Alert neighboring growers as late blight spores are airborne over long distances."
        ],
        "organic_treatment": [
            "Heavy prophylactic copper sprays (Copper Octanoate or Bordeaux mixture) every 5 days during blight risk periods.",
            "NOTE: Once late blight takes hold systemically, organic treatments have very limited curative ability."
        ],
        "chemical_treatment": [
            "Cymoxanil + Mancozeb (Curzate): Provides kickback activity up to 48 hours post-infection.",
            "Mandipropamid (Revus): Translaminar fungicide that binds to the plant cuticle.",
            "Fluopicolide (Presidio): Highly effective anti-oomycete chemistry."
        ],
        "preventive_measures": [
            "Never plant grocery store potatoes; use certified seed tubers only.",
            "Destroy all volunteer potato plants in spring.",
            "Plant resistant potato varieties (e.g. Sarpo Mira, Defender, Elba)."
        ],
        "nutrition_and_care": "Keep tubers covered with at least 4 inches of soil or thick mulch to prevent zoospores from reaching tubers during irrigation or rainfall."
    },
    "Potato___healthy": {
        "title": "Healthy Potato Vine (Solanum tuberosum)",
        "crop": "Potato",
        "category": "Healthy Plant",
        "severity_level": "None",
        "urgency": "🟢 Optimal Condition - Routine Care",
        "description": "Vigorous, dense green canopy with robust stems, producing abundant starches to feed developing underground tubers.",
        "symptoms": "Rich green composite leaves, clean stems, no lesions, yellowing, or downy mold.",
        "causes": "Full sun, cool acidic to neutral soil (pH 5.2-6.5), even watering, and certified seed stock.",
        "immediate_actions": ["No corrective actions needed."],
        "organic_treatment": ["Foliar kelp meal or compost tea spray monthly to support tuber bulking."],
        "chemical_treatment": ["None required."],
        "preventive_measures": [
            "Hill up plants by adding 3-4 inches of soil/mulch when vines reach 8 inches tall.",
            "Maintain consistent 1 to 2 inches of water per week, particularly during flowering and tuberization."
        ],
        "nutrition_and_care": "Fertilize with high potassium and phosphorus organic fertilizer (such as bone meal + sulfate of potash); avoid late-season nitrogen which promotes vine growth at the expense of tuber yield."
    },

    # ------------------ TOMATO ------------------
    "Tomato_Bacterial_spot": {
        "title": "Bacterial Spot (Xanthomonas perforans / euvesicatoria)",
        "crop": "Tomato",
        "category": "Bacterial Infection",
        "severity_level": "High",
        "urgency": "🚨 Immediate Containment Required",
        "description": "A serious bacterial disease causing leaf spotting, yellowing, blossom drop, and sunken scabs on fruit, drastically cutting production.",
        "symptoms": "Small, dark, greasy water-soaked spots (1-3 mm) on leaves that turn brown with yellow halos; leaves look ragged and shot-holed; raised scabby spots on green fruit.",
        "causes": "Warm, humid conditions, splashing rain, overhead sprinkler systems, and infected commercial seed lots.",
        "immediate_actions": [
            "Sterilize hands and tools with 70% alcohol before touching other plants.",
            "Strip infected lower foliage up to 18 inches off the ground.",
            "Discontinue all overhead watering immediately."
        ],
        "organic_treatment": [
            "Liquid Copper Soap Fungicide: Apply every 7-10 days, ensuring thorough underside coverage.",
            "Bacillus subtilis foliar spray to outcompete bacterial colonization.",
            "Garlic and clove oil botanical sprays as natural bactericides."
        ],
        "chemical_treatment": [
            "Fixed Copper combined with Mancozeb: Synergistic tank mix offering standard control.",
            "Streptomycin (where agricultural antibiotic use is permitted on seedlings)."
        ],
        "preventive_measures": [
            "Purchase certified disease-free seeds or perform hot-water seed treatments (50°C / 122°F for 25 mins).",
            "Stake and prune indeterminate tomatoes to maintain open airflow.",
            "Practice 3-year crop rotation away from solanaceous plants."
        ],
        "nutrition_and_care": "Maintain balanced calcium to prevent blossom end rot, and mulch beds with straw or landscape fabric to stop soil splatter."
    },
    "Tomato_Early_blight": {
        "title": "Early Blight (Alternaria linariae / solani)",
        "crop": "Tomato",
        "category": "Fungal Infection",
        "severity_level": "Moderate to High",
        "urgency": "⚠️ Action Required within 48 hours",
        "description": "One of the most widespread tomato diseases, starting on the lowest leaves and moving systematically upward as the season progresses.",
        "symptoms": "Characteristic dark brown target-pattern lesions with concentric rings surrounded by bright yellow margins; lower leaves brown, wither, and drop.",
        "causes": "Fungal spores splashed up from soil; favored by heavy dews, frequent rain, and warm temperatures (24-30°C / 75-86°F).",
        "immediate_actions": [
            "Prune off all lower branches below the lowest fruit cluster (bottom pruning).",
            "Apply 2-3 inches of clean organic mulch (straw, leaves, woodchips) to create a barrier over soil.",
            "Bag and discard removed foliage; never throw infected tomato leaves into home compost."
        ],
        "organic_treatment": [
            "Copper Fungicide spray applied weekly during wet periods.",
            "Potassium Bicarbonate + Horticultural Oil mix (1 tbsp bicarbonate + 1 tsp oil per gallon).",
            "Neem oil spray for early preventative control."
        ],
        "chemical_treatment": [
            "Chlorothalonil (Daconil): Apply at first symptom appearance; repeat every 7-10 days.",
            "Mancozeb (Dithane M-45): Effective preventative cover spray.",
            "Azoxystrobin (Heritage / Quadris): Systemic translaminar control."
        ],
        "preventive_measures": [
            "Stake, cage, or trellis plants to elevate foliage off ground level.",
            "Water solely at the soil level using drip irrigation or soaker hoses.",
            "Select resistant cultivars (look for 'EB' designation like Mountain Supreme, Defiant, or Iron Lady)."
        ],
        "nutrition_and_care": "Fertilize with tomato-specific fertilizer (e.g. 4-18-38 or 5-10-10) with adequate calcium and potassium to keep cell walls sturdy."
    },
    "Tomato_Late_blight": {
        "title": "Late Blight (Phytophthora infestans)",
        "crop": "Tomato",
        "category": "Oomycete / Water Mold",
        "severity_level": "Critical",
        "urgency": "🚨 EMERGENCY ACTION REQUIRED",
        "description": "The most catastrophic tomato pathogen; spreads exponentially in moist cool weather and can kill fully grown tomato plants in 48-72 hours.",
        "symptoms": "Water-soaked irregular greasy olive-brown to dark lesions on leaves and stems; white fuzzy mildew on leaf undersides in morning humidity; firm greasy brown rots on green fruit.",
        "causes": "Wind-driven airborne sporangia; high humidity (>90%) with moderate temperatures (16-24°C / 60-75°F).",
        "immediate_actions": [
            "Bag and remove entirely blighted plants immediately to protect remaining garden crops.",
            "Harvest mature green fruit from infected vines immediately; wash with dilute vinegar solution to reduce spoilage.",
            "Do not allow infected foliage to touch other garden beds."
        ],
        "organic_treatment": [
            "Prophylactic Copper Hydroxide spray applied before rainy forecasts.",
            "Note: Curative organic controls are ineffective once infection is systemic."
        ],
        "chemical_treatment": [
            "Chlorothalonil or Mancozeb: Continuous protective coverage.",
            "Cymoxanil (Curzate) or Mandipropamid (Revus): Translaminar kickback activity within 48 hours of infection.",
            "Dimethomorph (Acrobat): Potent anti-oomycete systemic agent."
        ],
        "preventive_measures": [
            "Plant resistant tomato varieties such as Defiant Ph-R, Mountain Magic, Plum Regal, or Iron Lady.",
            "Space plants at least 3 feet apart for maximum cross-ventilation.",
            "Avoid planting tomatoes near potatoes."
        ],
        "nutrition_and_care": "Maintain optimal root zone aeration and avoid over-saturating soil during periods of prolonged cloudy weather."
    },
    "Tomato_Leaf_Mold": {
        "title": "Tomato Leaf Mold (Passalora fulva / Cladosporium)",
        "crop": "Tomato",
        "category": "Fungal Infection",
        "severity_level": "Moderate",
        "urgency": "⚠️ Action Required within 3-5 days",
        "description": "Leaf mold predominantly attacks greenhouse and high-tunnel tomatoes, as well as humid outdoor gardens with dense foliage.",
        "symptoms": "Pale yellow or greenish-yellow blotches on the upper leaf surface, matching an olive-green to grayish-velvety fungal felt on the leaf underside.",
        "causes": "High relative humidity (>85%) combined with temperatures between 21-24°C (70-75°F); low air movement.",
        "immediate_actions": [
            "Increase greenhouse or garden ventilation immediately; use oscillating fans in high tunnels.",
            "Prune sucker shoots and lower dense foliage to allow air to pass directly through canopy.",
            "Heat and ventilate greenhouses in early morning to purge moisture."
        ],
        "organic_treatment": [
            "Potassium bicarbonate (MilStop): Excellent curative contact action on leaf mold.",
            "Copper soap (Copper Octanoate): Safe spray up to harvest day."
        ],
        "chemical_treatment": [
            "Chlorothalonil (Daconil): Broad spectrum preventative application.",
            "Difenoconazole + Cyprodinil (Inspire Super): Highly effective commercial standard."
        ],
        "preventive_measures": [
            "Grow leaf mold resistant varieties (look for 'Cf' genes).",
            "Avoid wetting leaves during evening waterings.",
            "Maintain greenhouse humidity below 80%."
        ],
        "nutrition_and_care": "Avoid excessive vegetative nitrogen feed that leads to ultra-dense foliage blankets."
    },
    "Tomato_Septoria_leaf_spot": {
        "title": "Septoria Leaf Spot (Septoria lycopersici)",
        "crop": "Tomato",
        "category": "Fungal Infection",
        "severity_level": "Moderate to High",
        "urgency": "⚠️ Action Required within 48-72 hours",
        "description": "A destructive fungal leaf spot that defoliates tomato plants from the bottom up, exposing developing fruit to sunscald and weakening plant vigor.",
        "symptoms": "Numerous small (1.5-3 mm) circular spots with grayish-white centers and dark brown margins, speckled with tiny black pepper-like pycnidia (fungal fruiting bodies).",
        "causes": "Spores overwintering on solanaceous weed hosts (horse nettle, nightshade) and tomato residue, splashed by raindrops.",
        "immediate_actions": [
            "Pinch off all infected bottom leaves as soon as first spots appear.",
            "Cover bare ground underneath with a 3-inch layer of straw or pine bark mulch.",
            "Disinfect stakes and cages."
        ],
        "organic_treatment": [
            "Liquid copper fungicide: Spray every 7-10 days throughout rainy periods.",
            "Biofungicides (Bacillus subtilis / Bacillus amyloliquefaciens).",
            "Neem oil extract applied as a preventative coating."
        ],
        "chemical_treatment": [
            "Chlorothalonil (Daconil): Primary standard for controlling Septoria.",
            "Mancozeb: Effective protectant spray.",
            "Pyraclostrobin (Cabrio): Commercial systemic control."
        ],
        "preventive_measures": [
            "Practice a 3-year rotation away from tomatoes, potatoes, and peppers.",
            "Keep tomato foliage dry by utilizing drip irrigation.",
            "Eradicate nearby solanaceous weeds (nightshade, groundcherry)."
        ],
        "nutrition_and_care": "Provide steady moisture and balanced nutrition; avoid excessive nitrogen which delays fruit ripening."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "title": "Two-Spotted Spider Mite Infestation (Tetranychus urticae)",
        "crop": "Tomato",
        "category": "Pest Infestation (Arachnid)",
        "severity_level": "Moderate to High",
        "urgency": "🚨 Fast Action Needed (Mites multiply rapidly in heat)",
        "description": "Microscopic sap-sucking arachnids that congregate on leaf undersides, piercing plant cells and spinning fine protective silken webbing.",
        "symptoms": "Fine pale yellow or white stippling (tiny dots) on upper leaf surfaces; dusty-looking leaves; fine spider silk webbing between leaf petioles; foliage bronzes and shrivels in heavy infestations.",
        "causes": "Hot, dry, and dusty weather (temperatures >27°C / 80°F); over-reliance on broad-spectrum insecticides that kill beneficial predatory mites.",
        "immediate_actions": [
            "Blast leaf undersides with a strong jet of water from a garden hose to knock off mites and destroy webbing.",
            "Prune and discard severely webbed branch tips in plastic bags.",
            "Keep surrounding garden paths watered to settle dust."
        ],
        "organic_treatment": [
            "Insecticidal Soap (Potassium salts of fatty acids): Spray leaf undersides thoroughly early in the morning.",
            "Horticultural Neem Oil (Cold-pressed): Smothers mite eggs and disrupts reproductive cycles.",
            "Release Beneficial Predatory Mites (Phytoseiulus persimilis or Neoseiulus californicus): Natural biological predators that devour spider mites."
        ],
        "chemical_treatment": [
            "Miticide / Acaricide: Abamectin (Agrimec) or Bifenazate (Acramite) specifically formulated for mites (avoid synthetic pyrethroids which cause mite flare-ups)."
        ],
        "preventive_measures": [
            "Avoid broad-spectrum pyrethroid insecticides that exterminate natural predators like ladybugs, lacewings, and minute pirate bugs.",
            "Mist plants lightly during extreme dry heatwaves to maintain local humidity.",
            "Remove broadleaf weeds near garden borders that serve as alternate mite hosts."
        ],
        "nutrition_and_care": "Ensure plants are well-watered; water-stressed drought plants produce higher amino acid concentrations in sap, which accelerates mite reproduction."
    },
    "Tomato__Target_Spot": {
        "title": "Target Spot (Corynespora cassiicola)",
        "crop": "Tomato",
        "category": "Fungal Infection",
        "severity_level": "Moderate",
        "urgency": "⚠️ Action Required within 3-5 days",
        "description": "Corynespora cassiicola produces dark pinpoint lesions that enlarge into target-like circles on tomato leaves, stems, and fruits.",
        "symptoms": "Brown, circular spots with concentric rings resembling an archery target; lesions may have chlorotic yellow halos; fruit lesions are sunken and dark.",
        "causes": "Favored by high temperatures (20-28°C / 68-82°F) and long periods of humidity or rain.",
        "immediate_actions": [
            "Remove and destroy infected leaves.",
            "Increase spacing between plants to enhance sunlight and airflow.",
            "Ensure leaves dry rapidly after morning dew."
        ],
        "organic_treatment": [
            "Biofungicide Serenade or Cease (Bacillus subtilis).",
            "Copper Hydroxide sprays every 7-10 days."
        ],
        "chemical_treatment": [
            "Chlorothalonil or Mancozeb: Preventative protectants.",
            "Fluxapyroxad + Pyraclostrobin (Priaxor): Outstanding dual-action control."
        ],
        "preventive_measures": [
            "Rotate crops annually with non-host crops (e.g. sweet corn, onions).",
            "Sanitize all garden stakes and trellising wire at season end.",
            "Use mulch barriers to minimize spore splashing from soil."
        ],
        "nutrition_and_care": "Maintain balanced potassium and magnesium levels to support sturdy epidermal cells."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "crop": "Tomato",
        "category": "Viral Disease (Begomovirus)",
        "severity_level": "Critical",
        "urgency": "🚨 Urgent Vector Control & Quarantine",
        "description": "A devastating plant virus transmitted by the silverleaf whitefly (Bemisia tabaci). Infected plants become severely stunted and yield little to no fruit.",
        "symptoms": "Upward curling and cupping of leaf margins, severe yellowing (chlorosis) between leaf veins, reduced leaf size, bushy stunted plant appearance, and blossom abortion.",
        "causes": "Transmitted in a persistent manner by silverleaf whiteflies; cannot be cured once the plant is infected; not spread by mechanical touch or seed.",
        "immediate_actions": [
            "Inspect the undersides of leaves for tiny fluttery whiteflies.",
            "Uproot and destroy severely infected stunted plants immediately to prevent whiteflies from acquiring and spreading the virus.",
            "Deploy bright yellow sticky traps around the garden canopy to monitor and capture whiteflies."
        ],
        "organic_treatment": [
            "Insecticidal Soap & Horticultural Oil: Spray leaf undersides weekly to smother whitefly nymphs and eggs.",
            "Neem Oil or Azadirachtin: Disrupts whitefly feeding and egg development.",
            "Reflective Aluminum Mulch: Repels incoming whiteflies by reflecting sky light."
        ],
        "chemical_treatment": [
            "Systemic Insecticides for whitefly vectors: Imidacloprid, Dinotefuran, or Acetamiprid (apply per label guidelines).",
            "Note: Fungicides and bactericides have zero efficacy against viruses."
        ],
        "preventive_measures": [
            "Plant TYLCV-resistant tomato cultivars (e.g., Tycoon, Camaro, Galahad, Chef's Choice Yellow).",
            "Use fine mesh insect netting (50-mesh) over seedlings and nursery plants.",
            "Eradicate nightshade, jimsonweed, and other alternative weed hosts."
        ],
        "nutrition_and_care": "Provide extra shade cloth protection during peak heatwaves to minimize whitefly migration stress."
    },
    "Tomato__Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus (ToMV)",
        "crop": "Tomato",
        "category": "Viral Disease (Tobamovirus)",
        "severity_level": "High",
        "urgency": "🚨 Strict Quarantine & Sanitation Needed",
        "description": "A highly stable and contagious tobamovirus that spreads through mechanical contact (hands, tools, clothing) and can persist in soil and dry debris for years.",
        "symptoms": "Mottled light and dark green mosaic patterns on leaves, leaf distortion, 'fern-leaf' or shoestring appearance, internal browning (brown wall) of fruit.",
        "causes": "Transmitted mechanically by contaminated hands, tools, or grafting; smokers can transmit related tobacco viruses; persists indefinitely in dry plant debris.",
        "immediate_actions": [
            "Wash hands thoroughly with soap and warm water (or non-fat milk solution) before handling tomato plants.",
            "Remove and incinerate/trash infected plants; do not compost.",
            "Sterilize all stakes, cages, and tools in a 20% non-fat dry milk solution or 10% trisodium phosphate (TSP)."
        ],
        "organic_treatment": [
            "No chemical or organic spray cures viral infections.",
            "Foliar milk sprays (1 part non-fat milk to 9 parts water) can help neutralize virus particles on plant surfaces during pruning."
        ],
        "chemical_treatment": [
            "There are no chemical viricides. Focus strictly on sanitation and tool disinfection."
        ],
        "preventive_measures": [
            "Plant resistant tomato varieties labeled with 'ToMV' or 'TMV'.",
            "Prohibit tobacco use anywhere near tomato gardens.",
            "Rotate away from solanaceous crops and solarize garden soil if virus history is severe."
        ],
        "nutrition_and_care": "Support healthy unaffected plants with steady watering and balanced nutrients to build natural resilience."
    },
    "Tomato_healthy": {
        "title": "Healthy Tomato Vine (Solanum lycopersicum)",
        "crop": "Tomato",
        "category": "Healthy Plant",
        "severity_level": "None",
        "urgency": "🟢 Optimal Condition - Routine Care",
        "description": "Vibrant, deep-green foliage, sturdy stems with healthy glandular hairs, vigorous blossom clusters, and zero signs of pathology.",
        "symptoms": "Deep emerald foliage, crisp erect growth, clean leaflet undersides, no spotting, curling, or chlorosis.",
        "causes": "Proper spacing, full sunlight (8+ hours daily), consistent soil moisture, and rich organic soil.",
        "immediate_actions": ["No interventions needed."],
        "organic_treatment": [
            "Apply compost tea or fish-seaweed foliar emulsion every 2-3 weeks to promote microflora."
        ],
        "chemical_treatment": ["None required."],
        "preventive_measures": [
            "Keep lower 12-18 inches of stems pruned of foliage to block soil-borne splash.",
            "Apply 2-3 inches of straw mulch across root zone.",
            "Maintain consistent soil moisture (1-2 inches weekly) to prevent blossom end rot and fruit cracking."
        ],
        "nutrition_and_care": "Feed with high-potassium, high-phosphorus organic fertilizer once fruit sets. Ensure steady calcium supply in soil."
    }
}

# Default / Fallback Disease Advice
DEFAULT_DISEASE_INFO = {
    "title": "Unidentified Plant Disease / Stress",
    "crop": "General Crop",
    "category": "Botanical Pathology",
    "severity_level": "Moderate",
    "urgency": "⚠️ Inspection Recommended within 48 hours",
    "description": "The specimen exhibits visible foliar lesions, spots, or discoloration consistent with common fungal or bacterial foliar disease.",
    "symptoms": "Leaf spots, chlorotic yellow halos, wilting, or necrotic margins.",
    "causes": "Prolonged leaf moisture, high relative humidity, poor air ventilation, or contaminated soil.",
    "immediate_actions": [
        "Isolate affected plant or prune off noticeably spotted lower leaves.",
        "Switch to ground-level drip irrigation; keep leaves completely dry.",
        "Disinfect garden shears with 70% alcohol after pruning."
    ],
    "organic_treatment": [
        "Neem Oil spray (2 tbsp/gal water) applied in early morning.",
        "Organic copper soap fungicide as broad-spectrum foliar protectant.",
        "Potassium bicarbonate spray to alter leaf pH against fungi."
    ],
    "chemical_treatment": [
        "Broad-spectrum multi-site fungicide (e.g. Chlorothalonil or Mancozeb) applied per label instructions."
    ],
    "preventive_measures": [
        "Ensure generous plant spacing for sunlight and breeze.",
        "Apply organic mulch around base to stop soil splashing.",
        "Rotate plant families annually."
    ],
    "nutrition_and_care": "Avoid excessive nitrogen fertilizers. Ensure adequate potassium and micronutrients."
}

def get_disease_expert_info(class_name: str) -> dict:
    """
    Looks up expert disease and care guide information for a given class name.
    Handles variations in naming, underscores, casing, and suffixes.
    """
    if not class_name:
        return DEFAULT_DISEASE_INFO
        
    cleaned = class_name.strip()
    
    # 1. Exact match
    if cleaned in DISEASE_DATABASE:
        return DISEASE_DATABASE[cleaned]
        
    # 2. Case-insensitive exact match
    for k, v in DISEASE_DATABASE.items():
        if k.lower() == cleaned.lower():
            return v
            
    # 3. Clean ' - Copy' or trailing underscores
    normalized = cleaned.replace(" - Copy", "").strip().rstrip("_")
    for k, v in DISEASE_DATABASE.items():
        if k.lower() == normalized.lower():
            return v
            
    # 4. Partial key match
    for k, v in DISEASE_DATABASE.items():
        if k.lower() in cleaned.lower() or cleaned.lower() in k.lower():
            return v
            
    # 5. Check if it's healthy
    if "healthy" in cleaned.lower():
        for k, v in DISEASE_DATABASE.items():
            if "healthy" in k.lower() and any(crop.lower() in cleaned.lower() for crop in ["apple", "corn", "pepper", "potato", "tomato"]):
                return v
        # Generic healthy
        return DISEASE_DATABASE.get("Tomato_healthy", DEFAULT_DISEASE_INFO)
        
    return DEFAULT_DISEASE_INFO

def get_all_disease_guides() -> dict:
    """Returns the entire encyclopedia of disease and care guides organized by crop."""
    return DISEASE_DATABASE


# =====================================================================
# Crop Pathology Vector, Incubation & Host Susceptibility Database
# =====================================================================

DISEASE_EPIDEMIOLOGY_DATABASE = {
    "Apple___Apple_scab": {
        "title": "Apple Scab (Venturia inaequalis)",
        "crop": "Apple",
        "pathogen_type": "Fungus (Ascomycete)",
        "incubation_days_min": 8,
        "incubation_days_max": 14,
        "primary_vectors": ["Wind-borne ascospores", "Rain splash", "Air currents"],
        "vector_carrier_insects": ["None (Primarily meteorological dispersal)"],
        "vector_categories": ["Wind", "Rain Splash"],
        "host_susceptibility_stage": "Green tip through petal fall (young expanding leaves & sepals)",
        "secondary_spread_rate": "Exponential (Polycyclic conidia repeat every 7-10 days under wet canopy)",
        "environmental_drivers": "Prolonged leaf wetness (> 6-9 hrs) at 13-24°C (55-75°F)",
        "quarantine_measures": [
            "Flail mow or shred fallen autumn leaves to eliminate perithecia overwintering sites.",
            "Apply 5% agricultural urea foliar spray at leaf drop to accelerate leaf decomposition.",
            "Maintain wide pruning spacing for rapid canopy wind drying."
        ]
    },
    "Apple___Black_rot": {
        "title": "Black Rot & Frogeye Leaf Spot (Botryosphaeria obtusa)",
        "crop": "Apple",
        "pathogen_type": "Fungus",
        "incubation_days_min": 7,
        "incubation_days_max": 15,
        "primary_vectors": ["Rain splash", "Insect feeding wounds", "Pruning tool contact", "Wind"],
        "vector_carrier_insects": ["Plum curculio (creates fruit entry wounds)", "Tarnished plant bug"],
        "vector_categories": ["Rain Splash", "Insects", "Pruning Tools", "Wind"],
        "host_susceptibility_stage": "Petal fall through mid-summer fruit sizing; wounded shoots",
        "secondary_spread_rate": "Moderate (Pycnidiospores wash down into fruit clusters)",
        "environmental_drivers": "Warm wet spells (20-27°C / 68-80°F) accompanied by 6+ hours of free water",
        "quarantine_measures": [
            "Prune dead brush and mummified fruit; burn or bury infected wood.",
            "Disinfect pruning shears between tree cuts with 70% isopropyl alcohol.",
            "Control puncture-feeding insect pests early in season."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "title": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "crop": "Apple",
        "pathogen_type": "Heteroecious Rust Fungus",
        "incubation_days_min": 10,
        "incubation_days_max": 21,
        "primary_vectors": ["Wind-borne basidiospores (from Eastern Red Cedar)", "Wind-borne aeciospores"],
        "vector_carrier_insects": ["None (Obligate airborne heteroecious cycle)"],
        "vector_categories": ["Wind"],
        "host_susceptibility_stage": "Pink bud through 3 weeks post-bloom (young leaf tissue < 8 days old)",
        "secondary_spread_rate": "Low (Monocyclic on apple; cannot spread apple-to-apple)",
        "environmental_drivers": "Wet spring rains causing cedar gall gel-horn eruption; 10-24°C (50-75°F)",
        "quarantine_measures": [
            "Eradicate Eastern Red Cedar (Juniperus virginiana) within 1-2 mile buffer perimeter if feasible.",
            "Prune gelatinous orange galls off adjacent landscape junipers before spring rains.",
            "Apply DMI or multi-site protective barrier during cedar spore dispersal window."
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "title": "Gray Leaf Spot (Cercospora zeae-maydis)",
        "crop": "Corn (Maize)",
        "pathogen_type": "Fungus",
        "incubation_days_min": 12,
        "incubation_days_max": 21,
        "primary_vectors": ["Wind-blown air currents", "Rain splash from crop residue", "Conservation tillage debris"],
        "vector_carrier_insects": ["None"],
        "vector_categories": ["Wind", "Rain Splash", "Soil/Debris"],
        "host_susceptibility_stage": "Tasseling (VT) through grain fill (R3 Milk stage)",
        "secondary_spread_rate": "Rapid (Polycyclic under continuous high humidity)",
        "environmental_drivers": "Relative humidity > 90% for 12+ consecutive hours; warm days (24-30°C / 75-86°F)",
        "quarantine_measures": [
            "Rotate out of corn for 1-2 seasons to deplete surface stalk residue.",
            "Till or incorporate crop debris into soil where microbial activity degrades inoculum.",
            "Select high-tolerance hybrid genetics with strong stay-green characteristics."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "title": "Common Rust (Puccinia sorghi)",
        "crop": "Corn (Maize)",
        "pathogen_type": "Obligate Rust Fungus",
        "incubation_days_min": 7,
        "incubation_days_max": 10,
        "primary_vectors": ["High-altitude wind currents", "Storm fronts originating from southern latitudes"],
        "vector_carrier_insects": ["None"],
        "vector_categories": ["Wind"],
        "host_susceptibility_stage": "Whorl through silking and pollination",
        "secondary_spread_rate": "Rapid (Prolific urediniospore production re-infects every 7 days)",
        "environmental_drivers": "Cool to moderate temperatures (16-24°C / 60-75°F) with heavy overnight dew",
        "quarantine_measures": [
            "Scout upper leaves in late vegetative stage (V8 to VT); apply fungicide if pustules exceed 5% leaf area.",
            "Plant resistant sweet corn and field corn hybrids possessing Rp single-gene resistance."
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "title": "Northern Corn Leaf Blight (Exserohilum turcicum)",
        "crop": "Corn (Maize)",
        "pathogen_type": "Fungus",
        "incubation_days_min": 7,
        "incubation_days_max": 14,
        "primary_vectors": ["Wind-blown conidia", "Rain splash from residue", "Machinery across un-sanitized fields"],
        "vector_carrier_insects": ["None"],
        "vector_categories": ["Wind", "Rain Splash", "Farm Machinery"],
        "host_susceptibility_stage": "Silking to dough stage (severe loss when infection reaches ear leaf prior to pollination)",
        "secondary_spread_rate": "Rapid (Polycyclic cigar-shaped lesions produce secondary conidia)",
        "environmental_drivers": "Moderate temperatures (18-27°C / 65-80°F) and 6-12 hours of leaf moisture",
        "quarantine_measures": [
            "Deep tillage of infected stubble post-harvest.",
            "Two-year rotation with non-grass crops (Soybean, Alfalfa).",
            "Fungicide intervention prior to tassel emergence if lower canopy is symptomatic."
        ]
    },
    "Grape___Black_rot": {
        "title": "Grape Black Rot (Guignardia bidwellii)",
        "crop": "Grape",
        "pathogen_type": "Fungus",
        "incubation_days_min": 10,
        "incubation_days_max": 18,
        "primary_vectors": ["Rain splash", "Wind-blown ascospores", "Overwintering mummified clusters in canopy"],
        "vector_carrier_insects": ["None direct; occasional berry moth wound opportunistic"],
        "vector_categories": ["Rain Splash", "Wind", "Mummified Fruit"],
        "host_susceptibility_stage": "Bud break through 4-5 weeks post-bloom (young green berries most vulnerable)",
        "secondary_spread_rate": "Exponential (Conidia erupt from pycnidia across infected shoots and berries)",
        "environmental_drivers": "Temperatures 21-27°C (70-80°F) with 7+ hours of continuous leaf wetness",
        "quarantine_measures": [
            "Drop and destroy all dried mummies from trellis wires during dormant winter pruning.",
            "Canopy management: Shoot-positioning and leaf-pulling around fruiting zone for sunlight exposure.",
            "Protective sprays starting at 1-3 inch shoot growth."
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "title": "Esca Complex & Black Measles",
        "crop": "Grape",
        "pathogen_type": "Fungal Wood Canker Complex (Phaeomoniella, Phaeoacremonium)",
        "incubation_days_min": 180,
        "incubation_days_max": 730,
        "primary_vectors": ["Pruning wound entry", "Airborne spores during winter rain", "Contaminated nursery propagation stock"],
        "vector_carrier_insects": ["Wood-boring beetles (minor mechanical entry)"],
        "vector_categories": ["Pruning Tools", "Wind", "Nursery Stock"],
        "host_susceptibility_stage": "Mature vines (> 5-7 years); fresh pruning wounds during winter dormancy",
        "secondary_spread_rate": "Low to Slow (Multi-year chronic vascular colonization)",
        "environmental_drivers": "Pruning in wet rainy winter periods; extreme summer heat triggering apoplexy stroke",
        "quarantine_measures": [
            "Delay pruning until late winter (just before bud break) when wounds heal faster.",
            "Paint all large wood pruning cuts with sealant containing fungicide or biocontrol trichoderma.",
            "Rogue and incinerate severely stunted dying vines."
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "title": "Grape Leaf Blight (Pseudocercospora vitis)",
        "crop": "Grape",
        "pathogen_type": "Fungus",
        "incubation_days_min": 10,
        "incubation_days_max": 16,
        "primary_vectors": ["Rain splash", "Air currents", "Overwintering leaf debris"],
        "vector_carrier_insects": ["None"],
        "vector_categories": ["Rain Splash", "Wind", "Soil/Debris"],
        "host_susceptibility_stage": "Mid to late summer canopy maturity (post-veraison)",
        "secondary_spread_rate": "Moderate (Foliar lesions spread across lower vine layers)",
        "environmental_drivers": "High relative humidity and temperatures above 24°C (75°F)",
        "quarantine_measures": [
            "Rake and bury fallen autumn foliage.",
            "Maintain fungicide coverage during warm rainy post-veraison spells."
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "title": "Bacterial Spot (Xanthomonas euvesicatoria)",
        "crop": "Bell Pepper",
        "pathogen_type": "Bacterium",
        "incubation_days_min": 5,
        "incubation_days_max": 10,
        "primary_vectors": ["Rain splash", "Overhead irrigation", "Contaminated seed lots", "Pruning tools and worker hands"],
        "vector_carrier_insects": ["None direct; leaf abrasions from strong winds or insects facilitate entry"],
        "vector_categories": ["Rain Splash", "Irrigation", "Seed Borne", "Pruning Tools", "Worker Hands"],
        "host_susceptibility_stage": "All vegetative and fruiting stages; especially warm wet transplant establishment",
        "secondary_spread_rate": "Exponential (Billions of bacteria multiply rapidly in guttation droplets)",
        "environmental_drivers": "Warm temperatures (24-30°C / 75-86°F) and free water on leaves for > 2 hours",
        "quarantine_measures": [
            "Hot water seed treatment (50°C for 25 minutes) or use certified disease-free seed.",
            "Strictly avoid working in pepper rows when foliage is wet from morning dew or rain.",
            "Sterilize greenhouse transplant trays and shears with 10% bleach solution."
        ]
    },
    "Potato___Early_blight": {
        "title": "Potato Early Blight (Alternaria solani)",
        "crop": "Potato",
        "pathogen_type": "Fungus",
        "incubation_days_min": 3,
        "incubation_days_max": 7,
        "primary_vectors": ["Wind-borne spores", "Rain splash", "Flea beetles (feeding injury vectors)", "Soil crop debris"],
        "vector_carrier_insects": ["Epitrix cucumeris (Potato flea beetle - feeding wounds provide entry points)"],
        "vector_categories": ["Wind", "Rain Splash", "Insects", "Soil/Debris"],
        "host_susceptibility_stage": "Tuber bulking and canopy maturity (older stressed lower leaves)",
        "secondary_spread_rate": "Rapid (Conidia form rapidly during alternating dry-wet diurnal cycles)",
        "environmental_drivers": "Alternating wet and dry periods; temperatures 24-29°C (75-85°F)",
        "quarantine_measures": [
            "Maintain optimal plant nitrogen and potassium nutrition to delay physiological senescence.",
            "Control flea beetle populations during emergence.",
            "Avoid planting adjacent to older solanaceous fields."
        ]
    },
    "Potato___Late_blight": {
        "title": "Potato Late Blight (Phytophthora infestans)",
        "crop": "Potato",
        "pathogen_type": "Oomycete (Water Mold)",
        "incubation_days_min": 3,
        "incubation_days_max": 5,
        "primary_vectors": ["Airborne sporangia (traveling up to 10+ miles on wind)", "Driving rain storms", "Infected seed tubers", "Cull piles"],
        "vector_carrier_insects": ["None (Ultra-virulent wind-borne dispersal)"],
        "vector_categories": ["Wind", "Rain Splash", "Seed Borne", "Cull Piles"],
        "host_susceptibility_stage": "All growth stages; especially dense canopy prior to row closure and tuber bulking",
        "secondary_spread_rate": "Catastrophic Exponential (Can devastate an entire field in 7-10 days)",
        "environmental_drivers": "Cool humid weather (15-21°C / 60-70°F) with relative humidity > 90% and rain",
        "quarantine_measures": [
            "Destroy all volunteer potatoes and cull piles; never leave culled tubers exposed to open air.",
            "Plant only certified disease-free seed tubers.",
            "Kill vines completely 2-3 weeks before harvest to prevent tuber rot during digging."
        ]
    },
    "Tomato___Bacterial_spot": {
        "title": "Bacterial Spot (Xanthomonas spp.)",
        "crop": "Tomato",
        "pathogen_type": "Bacterium",
        "incubation_days_min": 5,
        "incubation_days_max": 10,
        "primary_vectors": ["Wind-driven rain", "Aerosolized droplets", "Overhead irrigation", "Contaminated seed", "Pruning shears"],
        "vector_carrier_insects": ["Thrips or beetles (indirect mechanical transmission via feeding wounds)"],
        "vector_categories": ["Rain Splash", "Wind", "Irrigation", "Seed Borne", "Pruning Tools"],
        "host_susceptibility_stage": "Transplant stage through flowering and green fruit sizing",
        "secondary_spread_rate": "Exponential (Explosive multiplication in stomata and foliar hydathodes)",
        "environmental_drivers": "High temperatures (24-30°C / 75-86°F) and frequent rainfall or overhead sprinkling",
        "quarantine_measures": [
            "Use certified pathogen-tested seed or treat with hydrochloric acid or hot water.",
            "Switch immediately to subterranean drip irrigation.",
            "Disinfect stakes, trellises, and greenhouse benches between seasons."
        ]
    },
    "Tomato___Early_blight": {
        "title": "Tomato Early Blight (Alternaria linariae / solani)",
        "crop": "Tomato",
        "pathogen_type": "Fungus",
        "incubation_days_min": 4,
        "incubation_days_max": 8,
        "primary_vectors": ["Wind currents", "Rain splash from soil", "Flea beetles", "Infected tomato transplant stock"],
        "vector_carrier_insects": ["Epitrix spp. (Flea beetles)"],
        "vector_categories": ["Wind", "Rain Splash", "Insects", "Soil/Debris"],
        "host_susceptibility_stage": "Heavy fruit load stage (plant energy diverted to fruit weakens lower foliage)",
        "secondary_spread_rate": "Rapid (Secondary conidiophores form concentric bullseye rings)",
        "environmental_drivers": "Warm temperatures (24-29°C / 75-85°F) with heavy dew or frequent rains",
        "quarantine_measures": [
            "Mulch bed with organic straw or plastic barrier to prevent soil splash onto lower foliage.",
            "Prune bottom 12-18 inches of foliage (suckers and leaves) once plant reaches 3 feet height.",
            "Implement a 3-year crop rotation away from solanaceae."
        ]
    },
    "Tomato___Late_blight": {
        "title": "Tomato Late Blight (Phytophthora infestans)",
        "crop": "Tomato",
        "pathogen_type": "Oomycete",
        "incubation_days_min": 3,
        "incubation_days_max": 5,
        "primary_vectors": ["Long-distance wind currents", "Storm cloud fronts", "Infected potato volunteers nearby"],
        "vector_carrier_insects": ["None"],
        "vector_categories": ["Wind", "Rain Splash", "Nearby Culled Plants"],
        "host_susceptibility_stage": "Any stage, especially mid-to-late summer canopies with heavy foliage",
        "secondary_spread_rate": "Catastrophic Exponential (Millions of sporangia per lesion released into breeze)",
        "environmental_drivers": "Cool wet conditions (15-21°C / 60-70°F) with extended leaf wetness (> 8 hrs)",
        "quarantine_measures": [
            "Inspect garden weekly; immediately rogue out and bag severely infected plants in sealed plastic.",
            "Do not compost late-blight infected vines in home compost piles.",
            "Space plants generously (24-36 inches) on tall sturdy stakes for cross-breeze drying."
        ]
    },
    "Tomato___Leaf_Mold": {
        "title": "Tomato Leaf Mold (Passalora fulva / Cladosporium)",
        "crop": "Tomato",
        "pathogen_type": "Fungus",
        "incubation_days_min": 10,
        "incubation_days_max": 14,
        "primary_vectors": ["Air currents in high-tunnel greenhouses", "Worker clothing and hands", "Water condensation drip"],
        "vector_carrier_insects": ["None"],
        "vector_categories": ["Wind", "Greenhouse Air Currents", "Worker Hands/Clothing"],
        "host_susceptibility_stage": "Flowering through fruit harvesting in protected culture/greenhouses",
        "secondary_spread_rate": "Rapid (Velvety olive-brown spore masses on leaflet undersides)",
        "environmental_drivers": "High relative humidity (> 85%) and warm temperatures (21-24°C / 70-75°F)",
        "quarantine_measures": [
            "Increase greenhouse ventilation with exhaust fans and horizontal airflow (HAF).",
            "Heat and vent before sunset to prevent condensation on foliage overnight.",
            "Disinfect greenhouse structural posts and ground covers between crop cycles."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Septoria Leaf Spot (Septoria lycopersici)",
        "crop": "Tomato",
        "pathogen_type": "Fungus",
        "incubation_days_min": 6,
        "incubation_days_max": 10,
        "primary_vectors": ["Rain splash from soil", "Overhead irrigation", "Worker hands and cultivating tools"],
        "vector_carrier_insects": ["Colorado potato beetle (mechanical spore carrier on body)"],
        "vector_categories": ["Rain Splash", "Irrigation", "Worker Hands", "Tools", "Insects"],
        "host_susceptibility_stage": "First fruit set onward (starts at ground-level foliage and climbs upward)",
        "secondary_spread_rate": "Rapid (Pycnidiospores ooze in gelatinous tendrils during rain)",
        "environmental_drivers": "Extended wet weather (20-25°C / 68-77°F) with high humidity",
        "quarantine_measures": [
            "Strip bottom leaves showing tiny circular dark margins with gray centers.",
            "Never cultivate, weed, or stake when leaves are damp.",
            "Apply thick barrier mulch over exposed soil."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "Two-Spotted Spider Mites (Tetranychus urticae)",
        "crop": "Tomato",
        "pathogen_type": "Arachnid Pest & Foliar Damage",
        "incubation_days_min": 5,
        "incubation_days_max": 12,
        "primary_vectors": ["Two-spotted spider mite adults and nymphs", "Wind ballooning on fine silk threads", "Farm machinery and worker clothing"],
        "vector_carrier_insects": ["Tetranychus urticae (Two-Spotted Spider Mite - primary causative pest)"],
        "vector_categories": ["Insects/Mites", "Wind", "Farm Machinery", "Worker Hands/Clothing"],
        "host_susceptibility_stage": "Drought-stressed vegetative and fruiting canopy in midsummer",
        "secondary_spread_rate": "Exponential (A female lays up to 100 eggs; lifecycle completes in 5 days at 30°C)",
        "environmental_drivers": "Hot (> 28°C / 82°F), dry, dusty conditions with low relative humidity (< 40%)",
        "quarantine_measures": [
            "Release predatory mites (Phytoseiulus persimilis or Neoseiulus californicus) at first webbing sign.",
            "Hose down dusty farm roads near fields to prevent dust particle settling on leaf cuticles.",
            "Mist undersides of leaves with cold water or organic insecticidal soap."
        ]
    },
    "Tomato___Target_Spot": {
        "title": "Target Spot (Corynespora cassiicola)",
        "crop": "Tomato",
        "pathogen_type": "Fungus",
        "incubation_days_min": 7,
        "incubation_days_max": 14,
        "primary_vectors": ["Airborne conidia", "Rain splash", "Crop debris from previous solanaceous or cucurbit crops"],
        "vector_carrier_insects": ["None"],
        "vector_categories": ["Wind", "Rain Splash", "Soil/Debris"],
        "host_susceptibility_stage": "Mid to late season during dense canopy shading",
        "secondary_spread_rate": "Moderate to Rapid (Brown target-like concentric lesions spread under humid canopy)",
        "environmental_drivers": "High relative humidity (80-90%) and warm temperatures (20-28°C / 68-82°F)",
        "quarantine_measures": [
            "Improve canopy air circulation by suckering and trellising.",
            "Rotate with non-host crops (corn, sorghum, small grains).",
            "Apply protective fungicides (Chlorothalonil, Mancozeb) before canopy closure."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "crop": "Tomato",
        "pathogen_type": "Begomovirus (DNA Virus)",
        "incubation_days_min": 10,
        "incubation_days_max": 21,
        "primary_vectors": ["Silverleaf whitefly (Bemisia tabaci)", "Greenhouse whitefly (Trialeurodes vaporariorum)", "Infected nursery transplants"],
        "vector_carrier_insects": ["Bemisia tabaci (Silverleaf Whitefly / Sweetpotato Whitefly B-biotype - obligate vector)", "Trialeurodes vaporariorum"],
        "vector_categories": ["Insects/Mites", "Nursery Stock"],
        "host_susceptibility_stage": "Seedling to vegetative stages prior to flowering (early infection leads to 100% yield loss)",
        "secondary_spread_rate": "Rapid (Persistent circulative transmission by whitefly populations)",
        "environmental_drivers": "Warm, sunny, dry climates favorable to whitefly breeding explosions",
        "quarantine_measures": [
            "Install 50-mesh insect exclusion netting on all greenhouse and high-tunnel side vents.",
            "Deploy yellow sticky monitoring traps across crop perimeter.",
            "Rogue out and destroy symptomatic stunted plants immediately to remove virus reservoirs.",
            "Apply systemic neonicotinoid or organic azadirachtin/insecticidal soap to control whitefly nymphs."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Tobamovirus (ToMV)",
        "crop": "Tomato",
        "pathogen_type": "Tobamovirus (RNA Virus)",
        "incubation_days_min": 7,
        "incubation_days_max": 14,
        "primary_vectors": ["Mechanical transmission (worker hands, pruning shears, stakes, ties)", "Worker tobacco use / residue", "Seed coat contamination", "Root contact in hydroponic or close beds"],
        "vector_carrier_insects": ["Chewing grasshoppers or beetles (rare opportunistic transmission)"],
        "vector_categories": ["Pruning Tools", "Worker Hands/Clothing", "Seed Borne", "Mechanical Contact"],
        "host_susceptibility_stage": "All growth stages; severe distortion if infected as young seedling",
        "secondary_spread_rate": "Extremely High Mechanical (Tobamoviruses are ultra-stable and easily spread by simple touch)",
        "environmental_drivers": "Any environment; virus remains infectious on dried stakes and dead leaves for years",
        "quarantine_measures": [
            "Enforce strict handwashing with milk or 20% nonfat dry milk solution (milk proteins deactivate tobamovirus).",
            "Prohibit tobacco use (cigarettes, chewing tobacco) in or near greenhouse/field.",
            "Disinfect metal pruning shears with trisodium phosphate (TSP) or flaming.",
            "Bag and discard infected plants; do not compost."
        ]
    }
}

DEFAULT_EPIDEMIOLOGY_INFO = {
    "title": "General Foliar Pathology Profile",
    "crop": "General Plant Specimen",
    "pathogen_type": "Broad-Spectrum Foliar Pathogen",
    "incubation_days_min": 5,
    "incubation_days_max": 12,
    "primary_vectors": ["Wind-borne fungal spores", "Rain splash and high moisture droplet drift"],
    "vector_carrier_insects": ["Aphids, thrips, or general foliar pests"],
    "vector_categories": ["Wind", "Rain Splash"],
    "host_susceptibility_stage": "Active vegetative growth and dense canopy closure",
    "secondary_spread_rate": "Moderate (Dependent on continuous leaf moisture and temperature)",
    "environmental_drivers": "Leaf wetness duration > 6 hours and relative humidity > 80%",
    "quarantine_measures": [
        "Isolate symptomatic plant material immediately.",
        "Sanitize all pruning tools in 70% isopropyl alcohol between plants.",
        "Transition immediately to root-zone drip irrigation to prevent foliar water splashing."
    ]
}


def get_disease_epidemiology_profile(class_name: str) -> dict:
    """
    Looks up vector transmission pathways, latent incubation period, host susceptibility,
    and biosecurity quarantine measures for a given plant disease.
    """
    if not class_name:
        return DEFAULT_EPIDEMIOLOGY_INFO
        
    cleaned = class_name.strip()
    
    # 1. Exact match
    if cleaned in DISEASE_EPIDEMIOLOGY_DATABASE:
        return DISEASE_EPIDEMIOLOGY_DATABASE[cleaned]
        
    # 2. Case-insensitive exact match
    for k, v in DISEASE_EPIDEMIOLOGY_DATABASE.items():
        if k.lower() == cleaned.lower():
            return v
            
    # 3. Clean ' - Copy' or trailing underscores
    normalized = cleaned.replace(" - Copy", "").strip().rstrip("_")
    for k, v in DISEASE_EPIDEMIOLOGY_DATABASE.items():
        if k.lower() == normalized.lower():
            return v
            
    # 4. Partial key match
    for k, v in DISEASE_EPIDEMIOLOGY_DATABASE.items():
        if k.lower() in cleaned.lower() or cleaned.lower() in k.lower():
            return v
            
    # 5. Healthy check
    if "healthy" in cleaned.lower():
        return {
            "title": f"Healthy Canopy Profile ({cleaned})",
            "crop": cleaned.split("_")[0] if "_" in cleaned else "Crop",
            "pathogen_type": "Clean Physiological Canopy",
            "incubation_days_min": 0,
            "incubation_days_max": 0,
            "primary_vectors": ["None (Plant shows robust immune defense and zero systemic infection)"],
            "vector_carrier_insects": ["None detected"],
            "vector_categories": ["None"],
            "host_susceptibility_stage": "Optimal vigor across all phenological phases",
            "secondary_spread_rate": "None (No pathogen present)",
            "environmental_drivers": "Sustained by proper spacing, balanced nutrition, and appropriate irrigation",
            "quarantine_measures": [
                "Maintain routine sanitation and scouting.",
                "Continue protective cultural regimens to preserve canopy immunity."
            ]
        }
        
    return DEFAULT_EPIDEMIOLOGY_INFO


def find_diseases_by_vector(vector_keyword: str) -> list:
    """
    Searches across all disease epidemiology records for diseases matching a specific
    vector mechanism or insect carrier (e.g. 'whitefly', 'aphid', 'wind', 'rain', 'pruning', 'seed').
    """
    if not vector_keyword:
        return []
        
    query = vector_keyword.strip().lower()
    matches = []
    
    for key, data in DISEASE_EPIDEMIOLOGY_DATABASE.items():
        # Check primary vectors, vector categories, and carrier insects
        vectors_text = " ".join(data.get("primary_vectors", [])).lower()
        cats_text = " ".join(data.get("vector_categories", [])).lower()
        insects_text = " ".join(data.get("vector_carrier_insects", [])).lower()
        
        if query in vectors_text or query in cats_text or query in insects_text:
            matches.append({
                "disease_key": key,
                "title": data["title"],
                "crop": data["crop"],
                "pathogen_type": data["pathogen_type"],
                "incubation_range": f"{data['incubation_days_min']}-{data['incubation_days_max']} days",
                "primary_vectors": data["primary_vectors"],
                "secondary_spread_rate": data["secondary_spread_rate"]
            })
            
    return matches
