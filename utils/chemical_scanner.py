"""
PlantGuard AI - Chemical Formulation, EPA Registration & Resistance Engine
Decodes EPA registration numbers, FRAC modes of action, pollinator toxicity,
and multi-site rotation protocols to prevent pathogen resistance.
"""

import re

# Comprehensive Agricultural Chemical Registry
EPA_REGISTRATION_REGISTRY = {
    "100-1098": {
        "trade_name": "Heritage 50 WDG",
        "active_ingredient": "Azoxystrobin (50.0%)",
        "epa_reg_no": "100-1098",
        "barcode": "702941010983",
        "frac_group": "Group 11 (QoI / Strobilurins)",
        "mode_of_action": "Inhibits mitochondrial respiration at Complex III (Qo site)",
        "resistance_risk": "HIGH",
        "signal_word": "CAUTION",
        "max_annual_apps": 2,
        "rainfast_hours": 2.0,
        "pollinator_safety": "Low Toxicity to Honeybees",
        "restricted_mixes": ["Do not tank mix with silicone surfactants or crop oils on sensitive apple varieties (Gala, McIntosh)"],
        "recommended_rotation": ["Group 3 (DMI / Triazoles)", "Group 7 (SDHI)", "Group M (Multi-site Protectants)"]
    },
    "70051-107": {
        "trade_name": "Trilogy Bio-Fungicide & Miticide",
        "active_ingredient": "Clarified Hydrophobic Extract of Neem Oil (70.0%)",
        "epa_reg_no": "70051-107",
        "barcode": "735623001072",
        "frac_group": "Group NC (Not Classified / Botanical Lipid Disrupter)",
        "mode_of_action": "Suffocation and cellular membrane disruption of hyphae",
        "resistance_risk": "VERY LOW (No known physiological resistance)",
        "signal_word": "CAUTION",
        "max_annual_apps": 12,
        "rainfast_hours": 4.0,
        "pollinator_safety": "Toxic if directly sprayed onto foraging bees; safe once dried",
        "restricted_mixes": ["Strictly prohibited to mix with or apply within 30 days of Sulfur WP"],
        "recommended_rotation": ["Group 44 (Bacillus subtilis)", "Group BM02 (Tea tree oil)", "Group M (Copper)"]
    },
    "4-61": {
        "trade_name": "Bonide Liquid Copper Octanoate",
        "active_ingredient": "Copper Octanoate (Copper Soap) (10.0%)",
        "epa_reg_no": "4-61",
        "barcode": "037321004618",
        "frac_group": "Group M01 (Inorganic Copper - Multi-Site)",
        "mode_of_action": "Denatures multi-site fungal enzymes; prevents spore germination",
        "resistance_risk": "LOW (Multi-site biochemical interference)",
        "signal_word": "CAUTION",
        "max_annual_apps": 6,
        "rainfast_hours": 4.0,
        "pollinator_safety": "Virtually non-toxic to adult honeybees",
        "restricted_mixes": ["Do not tank mix with acidic buffers that drop spray pH below 6.0 (phytotoxic copper ion release)"],
        "recommended_rotation": ["Group 11 (Strobilurin)", "Group 3 (DMI)", "Group 44 (Biofungicides)"]
    },
    "264-1152": {
        "trade_name": "Luna Tranquility Fungicide",
        "active_ingredient": "Fluopyram (11.3%) + Pyrimethanil (33.8%)",
        "epa_reg_no": "264-1152",
        "barcode": "026411520014",
        "frac_group": "Group 7 (SDHI) + Group 9 (Anilinopyrimidines)",
        "mode_of_action": "Dual: Inhibits succinate dehydrogenase + blocks methionine biosynthesis",
        "resistance_risk": "MEDIUM to HIGH",
        "signal_word": "CAUTION",
        "max_annual_apps": 2,
        "rainfast_hours": 1.5,
        "pollinator_safety": "Safe for bees when applied per label directions",
        "restricted_mixes": ["Do not apply in conjunction with high salt fertilizers during bloom"],
        "recommended_rotation": ["Group 3 (Triazoles)", "Group 11 (Strobilurins)", "Group M02 (Sulfur)"]
    },
    "62719-414": {
        "trade_name": "Dithane M-45 (Mancozeb)",
        "active_ingredient": "Mancozeb (75.0% DF)",
        "epa_reg_no": "62719-414",
        "barcode": "062719414002",
        "frac_group": "Group M03 (Dithiocarbamates - Multi-Site)",
        "mode_of_action": "Multi-site contact activity disabling sulfhydryl-containing enzymes",
        "resistance_risk": "LOW",
        "signal_word": "CAUTION",
        "max_annual_apps": 4,
        "rainfast_hours": 3.0,
        "pollinator_safety": "Non-toxic to bees once foliar deposit is dry",
        "restricted_mixes": ["Incompatible with highly alkaline materials (Bordeaux mix, lime)"],
        "recommended_rotation": ["Group 11 (Strobilurin)", "Group 7 (SDHI)", "Group 21 (Cyazofamid)"]
    },
    "50534-114": {
        "trade_name": "Daconil Ultrex (Chlorothalonil)",
        "active_ingredient": "Chlorothalonil (82.5% WDG)",
        "epa_reg_no": "50534-114",
        "barcode": "050534114008",
        "frac_group": "Group M05 (Chloronitriles - Multi-Site)",
        "mode_of_action": "Inactivates cellular glutathione and essential sulfhydryl enzymes",
        "resistance_risk": "LOW",
        "signal_word": "DANGER (Eye irritation hazard)",
        "max_annual_apps": 3,
        "rainfast_hours": 2.0,
        "pollinator_safety": "Non-toxic to adult bees; toxic to aquatic organisms",
        "restricted_mixes": ["Do not combine with organosilicate stickers; do not apply to apples post-petal fall"],
        "recommended_rotation": ["Group 3 (DMI)", "Group 7 (SDHI)", "Group 40 (Carboxylic acid amides)"]
    },
    "264-1150": {
        "trade_name": "Serenade ASO (Bacillus subtilis)",
        "active_ingredient": "Bacillus amyloliquefaciens / subtilis strain QST 713 (1.34%)",
        "epa_reg_no": "264-1150",
        "barcode": "026411500021",
        "frac_group": "Group 44 (Microbial Lipopeptides)",
        "mode_of_action": "Disrupts pathogen cell membranes and triggers Systemic Acquired Resistance (SAR)",
        "resistance_risk": "VERY LOW",
        "signal_word": "CAUTION",
        "max_annual_apps": 10,
        "rainfast_hours": 3.0,
        "pollinator_safety": "Completely bee safe; OMRI listed for organic production",
        "restricted_mixes": ["Compatible with most protectants; do not mix with live mycorrhizal inoculants"],
        "recommended_rotation": ["Group M (Copper)", "Group 44 (Alternate Bio-fungicides)", "Group BM02"]
    }
}

def lookup_chemical_by_epa_or_name(query: str) -> dict:
    """
    Looks up chemical product details by EPA Reg Number, Barcode, or trade name.
    """
    clean_q = query.strip().lower()
    
    # 1. Exact match by EPA Reg No
    for epa_no, data in EPA_REGISTRATION_REGISTRY.items():
        if epa_no.lower() == clean_q or epa_no.replace("-", "") == clean_q.replace("-", ""):
            return {"match_found": True, "record": data}
            
    # 2. Exact match by Barcode (UPC)
    for epa_no, data in EPA_REGISTRATION_REGISTRY.items():
        if data.get("barcode") == clean_q:
            return {"match_found": True, "record": data}
            
    # 3. Fuzzy match by Trade Name or Active Ingredient
    for epa_no, data in EPA_REGISTRATION_REGISTRY.items():
        t_name = data["trade_name"].lower()
        active = data["active_ingredient"].lower()
        if clean_q in t_name or clean_q in active or any(w in t_name for w in clean_q.split() if len(w) > 3):
            return {"match_found": True, "record": data}
            
    return {
        "match_found": False,
        "query": query,
        "message": f"No registered EPA chemical entry matched '{query}'. Please check the EPA Registration number on the container label."
    }

def get_all_chemical_catalog() -> list:
    """Returns all registered chemical formulation records."""
    return list(EPA_REGISTRATION_REGISTRY.values())

def get_frac_rotation_guidance(current_frac: str) -> dict:
    """
    Generates recommended FRAC rotation protocols to break chemical resistance.
    """
    for _, item in EPA_REGISTRATION_REGISTRY.items():
        if current_frac.lower() in item["frac_group"].lower():
            return {
                "current_frac": item["frac_group"],
                "risk_level": item["resistance_risk"],
                "recommended_rotation": item["recommended_rotation"],
                "max_consecutive_sprays": 2 if "HIGH" in item["resistance_risk"] else 3
            }
    return {
        "current_frac": current_frac,
        "risk_level": "UNKNOWN",
        "recommended_rotation": ["Rotate with Group M Multi-site protectants"],
        "max_consecutive_sprays": 2
    }
