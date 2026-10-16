"""
PlantGuard AI - Seasonal Disease & Crop Phenology Calendar
Guides growers through critical crop growth stages, identifying key vulnerability windows
and proactive preventative spray schedules throughout the growing season.
"""

PHENOLOGY_CALENDAR = {
    "Apple": [
        {
            "stage": "1. Dormant Stage (Late Winter)",
            "timing": "January - February",
            "threats": ["Overwintering Apple Scab spores", "San Jose Scale", "Fire Blight cankers"],
            "preventive_spray": "Liquid Copper Fungicide + Horticultural Dormant Oil",
            "action": "Prune out dead wood, fire blight strikes, and mummified fruits. Disinfect shears between cuts."
        },
        {
            "stage": "2. Green Tip & Tight Cluster (Early Spring)",
            "timing": "March - April",
            "threats": ["Primary Apple Scab ascospores", "Cedar Apple Rust"],
            "preventive_spray": "Mancozeb or Captan 50WP (Protectant)",
            "action": "Monitor Mills scab infection periods whenever rain lasts longer than 6 hours."
        },
        {
            "stage": "3. Pink Bud to Bloom",
            "timing": "April - May",
            "threats": ["Apple Scab", "Powdery Mildew", "Blossom Blight"],
            "preventive_spray": "Myclobutanil (Immunox) or Biofungicide (Bacillus subtilis). DO NOT SPRAY INSECTICIDES during bloom to protect bees.",
            "action": "Ensure maximum bee pollination; avoid all pesticide applications during active foraging hours."
        },
        {
            "stage": "4. Petal Fall to Fruit Set",
            "timing": "May - June",
            "threats": ["Black Rot (Frogeye leaf spot)", "Cedar Apple Rust", "Codling Moth"],
            "preventive_spray": "Captan 50WP + Thiophanate-methyl or Neem Oil 70%",
            "action": "Thin fruits to one per cluster to minimize rot spread between touching fruit skins."
        },
        {
            "stage": "5. Cover Sprays & Maturation",
            "timing": "July - August",
            "threats": ["Summer Rots (Bitter Rot, Sooty Blotch, Flyspeck)"],
            "preventive_spray": "Captan + Potassium Bicarbonate before heavy late-summer rainstorms.",
            "action": "Mow orchard grass low to reduce ambient ground humidity around the bottom canopy."
        },
        {
            "stage": "6. Post-Harvest & Leaf Drop (Autumn)",
            "timing": "October - November",
            "threats": ["Overwintering inoculum on fallen leaf floor"],
            "preventive_spray": "5% Foliar Urea spray just prior to leaf fall to accelerate leaf decomposition.",
            "action": "Flail mow or rake all fallen leaves to eradicate Venturia inaequalis overwintering pseudothecia."
        }
    ],
    "Tomato": [
        {
            "stage": "1. Seedling & Nursery (Indoor / Greenhouse)",
            "timing": "Weeks 1 - 4",
            "threats": ["Damping-Off (Pythium)", "Bacterial Spot (Seed-borne)"],
            "preventive_spray": "Sterile potting mix; Trichoderma or Bacillus soil drench.",
            "action": "Ensure bottom heat (21-24°C), strong ventilation fans, and avoid overwatering seedling trays."
        },
        {
            "stage": "2. Transplanting & Vegetative Growth",
            "timing": "Weeks 5 - 8",
            "threats": ["Early Blight (Alternaria solani)", "Septoria Leaf Spot", "Bacterial Spot"],
            "preventive_spray": "Liquid Copper Octanoate soap or Neem Oil spray every 7-10 days.",
            "action": "Strip lower leaves touching soil; install 3-inch straw mulch layer and ground drip lines."
        },
        {
            "stage": "3. Flowering & Early Fruit Set",
            "timing": "Weeks 9 - 12",
            "threats": ["Early Blight", "Two-Spotted Spider Mites", "Target Spot", "Blossom End Rot (Calcium deficit)"],
            "preventive_spray": "Mancozeb or Chlorothalonil + Foliar Chelated Calcium spray.",
            "action": "Stake or trellis vines upright to maximize airflow. Scout undersides of leaves for spider mite webbing."
        },
        {
            "stage": "4. Mid-Season Fruit Bulking",
            "timing": "Weeks 13 - 16",
            "threats": ["Late Blight (Phytophthora infestans)", "Tomato Yellow Leaf Curl Virus (Whitefly vector)"],
            "preventive_spray": "Chlorothalonil or Copper + Bacillus subtilis bio-protective film.",
            "action": "Inspect weather radar for cool, foggy, or rainy spells (Late Blight trigger). Isolate suspicious plants."
        },
        {
            "stage": "5. Harvest & Late Season",
            "timing": "Weeks 17 - 22",
            "threats": ["Anthracnose fruit rot", "Defoliation from blights"],
            "preventive_spray": "Copper Soap (0-day Pre-Harvest Interval).",
            "action": "Harvest ripe fruit promptly; do not allow split or overripe tomatoes to rot on the vine."
        }
    ],
    "Potato": [
        {
            "stage": "1. Seed Planting & Sprouting",
            "timing": "Early Spring",
            "threats": ["Rhizoctonia Canker", "Seed Piece Decay (Fusarium)"],
            "preventive_spray": "Certified disease-free seed potatoes; sulfur dust on cut seed pieces.",
            "action": "Plant in warm (10°C+), well-drained soil; allow cut seed pieces to suberize (callus) 2-3 days before planting."
        },
        {
            "stage": "2. Tuber Emergence & Canopy Closure",
            "timing": "Late Spring",
            "threats": ["Early Blight", "Colorado Potato Beetle"],
            "preventive_spray": "Preventative Copper or Mancozeb application prior to row closure.",
            "action": "Hill soil 4-6 inches around stems to protect developing tubers from sun-greening and fungal spores."
        },
        {
            "stage": "3. Tuber Bulking (Critical Risk Period)",
            "timing": "Summer",
            "threats": ["Late Blight (Phytophthora infestans) - High Devastation Risk"],
            "preventive_spray": "Curative/Systemic fungicide (Chlorothalonil or Metalaxyl) on 5-7 day intervals if weather is wet.",
            "action": "Immediately rogue (uproot and bag) any late-blight positive plant to protect adjacent rows."
        },
        {
            "stage": "4. Vine Senescence & Harvest Prep",
            "timing": "Autumn",
            "threats": ["Tuber rot from spores washing into soil"],
            "preventive_spray": "Vine-killing (desiccation) 2 weeks before harvest.",
            "action": "Allow tuber skins to set firmly in dry soil before harvest; avoid harvesting during wet mud conditions."
        }
    ],
    "Corn (Maize)": [
        {
            "stage": "1. Emergence & V6 Vegetative Stage",
            "timing": "May - June",
            "threats": ["Seedling blights", "Stewart's Wilt (Flea beetle vector)"],
            "preventive_spray": "Fungicidal seed treatment; monitor early leaf feeding.",
            "action": "Ensure adequate nitrogen side-dressing and soil drainage."
        },
        {
            "stage": "2. V12 to Tasseling (VT)",
            "timing": "July",
            "threats": ["Northern Corn Leaf Blight (Exserohilum turcicum)", "Gray Leaf Spot (Cercospora zeae-maydis)"],
            "preventive_spray": "Azoxystrobin or Pyraclostrobin strobilurin fungicide at early tassel.",
            "action": "Scout lower canopy for rectangular lesions (Gray Leaf Spot) or cigar-shaped lesions (Northern Leaf Blight)."
        },
        {
            "stage": "3. Silking (R1) to Milk Stage (R3)",
            "timing": "August",
            "threats": ["Common Rust (Puccinia sorghi)", "Southern Rust", "Ear Rots"],
            "preventive_spray": "Triazole + Strobilurin premix if disease threshold exceeds 1-2 pustules per ear leaf.",
            "action": "Protect the 'ear leaf' and top two leaves as they drive 80% of final grain fill yield."
        },
        {
            "stage": "4. Maturity & Harvest (R6)",
            "timing": "September - October",
            "threats": ["Stalk rot lodging", "Gibberella / Fusarium ear rot"],
            "preventive_spray": "Harvest promptness; dry grain immediately to below 15% moisture.",
            "action": "Chop and deep-till crop residue post-harvest to speed fungal breakdown before winter."
        }
    ],
    "Bell Pepper": [
        {
            "stage": "1. Nursery to Transplant",
            "timing": "Spring",
            "threats": ["Bacterial Spot (Xanthomonas)", "Phytophthora root rot"],
            "preventive_spray": "Copper Hydroxide + Mancozeb dip; certified disease-free seed.",
            "action": "Avoid overhead watering; space trays for good aeration."
        },
        {
            "stage": "2. Vegetative Canopy Development",
            "timing": "Late Spring",
            "threats": ["Bacterial Spot", "Aphids / Thrips (Virus vectors)"],
            "preventive_spray": "Neem Oil 70% or Insecticidal Soap for vectors; Copper Soap for bacteria.",
            "action": "Install drip lines; apply mulch to suppress weed reservoirs of mosaic viruses."
        },
        {
            "stage": "3. Flowering & Fruit Development",
            "timing": "Summer",
            "threats": ["Sunscald followed by secondary Alternaria rot", "Bacterial Spot on fruit"],
            "preventive_spray": "Foliar calcium + Potassium Bicarbonate / Copper.",
            "action": "Maintain dense leaf canopy over peppers to prevent sunscald burns."
        }
    ]
}

def get_crop_calendar(crop_name: str) -> list:
    """Returns the phenological stages and spray guide for a given crop."""
    for k, v in PHENOLOGY_CALENDAR.items():
        if k.lower() in crop_name.lower():
            return v
    return PHENOLOGY_CALENDAR.get("Tomato", [])
