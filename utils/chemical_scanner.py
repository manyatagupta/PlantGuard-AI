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
        "rei_hours": 4,
        "phi_days": 0,
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
        "rei_hours": 4,
        "phi_days": 0,
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
        "rei_hours": 12,
        "phi_days": 1,
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
        "rei_hours": 12,
        "phi_days": 7,
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
        "rei_hours": 24,
        "phi_days": 14,
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
        "rei_hours": 12,
        "phi_days": 7,
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
        "rei_hours": 4,
        "phi_days": 0,
        "pollinator_safety": "Completely bee safe; OMRI listed for organic production",
        "restricted_mixes": ["Compatible with most protectants; do not mix with live mycorrhizal inoculants"],
        "recommended_rotation": ["Group M (Copper)", "Group 44 (Alternate Bio-fungicides)", "Group BM02"]
    },
    "62719-410": {
        "trade_name": "Rally 40WSP (Myclobutanil)",
        "active_ingredient": "Myclobutanil (40.0%)",
        "epa_reg_no": "62719-410",
        "barcode": "062719410011",
        "frac_group": "Group 3 (DMI / Triazoles)",
        "mode_of_action": "Inhibits C14-demethylase in sterol biosynthesis (ergosterol inhibitor)",
        "resistance_risk": "MEDIUM",
        "signal_word": "CAUTION",
        "max_annual_apps": 4,
        "rainfast_hours": 2.0,
        "rei_hours": 24,
        "phi_days": 14,
        "pollinator_safety": "Non-toxic to adult honeybees",
        "restricted_mixes": ["Do not tank mix with captan during bloom on sensitive fruit cultivars"],
        "recommended_rotation": ["Group 11 (Strobilurins)", "Group 7 (SDHI)", "Group M (Multi-site)"]
    },
    "7969-199": {
        "trade_name": "Pristine Fungicide",
        "active_ingredient": "Pyraclostrobin (12.8%) + Boscalid (25.2%)",
        "epa_reg_no": "7969-199",
        "barcode": "079691990033",
        "frac_group": "Group 11 (QoI) + Group 7 (SDHI)",
        "mode_of_action": "Dual site: Inhibits Complex II + Complex III fungal respiration",
        "resistance_risk": "HIGH",
        "signal_word": "CAUTION",
        "max_annual_apps": 2,
        "rainfast_hours": 2.0,
        "rei_hours": 12,
        "phi_days": 0,
        "pollinator_safety": "Low toxicity to adult bees; avoid direct bloom mist",
        "restricted_mixes": ["Do not combine with foliar nutrient salts at high concentration"],
        "recommended_rotation": ["Group 3 (Triazoles)", "Group M01 (Copper)", "Group M05 (Chlorothalonil)"]
    },
    "2935-502": {
        "trade_name": "Kumulus DF (Micronized Sulfur)",
        "active_ingredient": "Sulfur (80.0%)",
        "epa_reg_no": "2935-502",
        "barcode": "029355020088",
        "frac_group": "Group M02 (Inorganic Sulfur - Multi-Site)",
        "mode_of_action": "Disrupts multi-site electron transport and fungal respiration",
        "resistance_risk": "LOW (Zero known resistance after centuries of use)",
        "signal_word": "CAUTION",
        "max_annual_apps": 8,
        "rainfast_hours": 4.0,
        "rei_hours": 24,
        "phi_days": 1,
        "pollinator_safety": "Non-toxic to honeybees",
        "restricted_mixes": ["Strictly DO NOT combine with or apply within 30 days of horticultural or neem oils"],
        "recommended_rotation": ["Group 11 (Strobilurin)", "Group 3 (Triazole)", "Group 44 (Biofungicide)"]
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


def evaluate_seasonal_rotation_plan(applied_products: list) -> dict:
    """
    Evaluates a sequence of seasonal fungicide applications for FRAC rotation compliance,
    consecutive single-site exposure risks, and cumulative pathogen resistance pressure.
    """
    if not applied_products:
        return {
            "status": "EMPTY_PLAN",
            "score": 0,
            "badge_color": "#94a3b8",
            "total_applications": 0,
            "distinct_frac_groups": 0,
            "violations": [],
            "history": [],
            "recommendation": "Add applied products to evaluate seasonal rotation."
        }
        
    resolved_history = []
    frac_counts = {}
    violations = []
    
    prev_frac = None
    consecutive_count = 0
    
    for idx, p_name in enumerate(applied_products):
        lookup = lookup_chemical_by_epa_or_name(p_name)
        if lookup["match_found"]:
            rec = lookup["record"]
            trade = rec["trade_name"]
            frac = rec["frac_group"]
            risk = rec["resistance_risk"]
            max_app = rec["max_annual_apps"]
        else:
            trade = p_name
            frac = "Unknown Mode of Action"
            risk = "MEDIUM"
            max_app = 4
            
        frac_counts[frac] = frac_counts.get(frac, 0) + 1
        
        # Check consecutive exposure for high-risk single-site chemistries
        if frac == prev_frac and "Group M" not in frac and "NC" not in frac and "44" not in frac:
            consecutive_count += 1
            if consecutive_count >= 2:
                violations.append({
                    "spray_index": idx + 1,
                    "product": trade,
                    "frac_group": frac,
                    "type": "CONSECUTIVE_OVEREXPOSURE",
                    "severity": "CRITICAL",
                    "description": f"Applied {trade} ({frac}) consecutively {consecutive_count} times. Overexposes target fungi to identical single-site cellular mutation pressures."
                })
        else:
            consecutive_count = 1
            prev_frac = frac
            
        # Check max annual limit
        if frac_counts[frac] > max_app:
            violations.append({
                "spray_index": idx + 1,
                "product": trade,
                "frac_group": frac,
                "type": "ANNUAL_LABEL_EXCEEDED",
                "severity": "HIGH",
                "description": f"Exceeded legal annual maximum label applications ({max_app} allowed, {frac_counts[frac]} applied)."
            })
            
        resolved_history.append({
            "spray_order": idx + 1,
            "product": trade,
            "frac_group": frac,
            "resistance_risk": risk
        })
        
    # Calculate Resistance Score (0 to 100)
    base_score = 15
    for v in violations:
        if v["severity"] == "CRITICAL":
            base_score += 35
        elif v["severity"] == "HIGH":
            base_score += 20
            
    # Distinct FRAC bonus
    distinct_fracs = len(set(h["frac_group"] for h in resolved_history))
    if distinct_fracs >= 3:
        base_score = max(5, base_score - 15)
        
    resistance_score = min(100, max(5, base_score))
    
    if resistance_score >= 60:
        status = "CRITICAL RESISTANCE RISK"
        badge_color = "#ef4444"
        rec_text = "Foliar pathogen population is under severe selective mutation pressure. Immediately switch to multi-site protectants (FRAC Group M01 Copper or M05 Chlorothalonil) or biological biofungicides (Group 44)."
    elif resistance_score >= 35:
        status = "MODERATE ROTATION WARNING"
        badge_color = "#f59e0b"
        rec_text = "Rotate next application to an unshared biochemical mode of action. Ensure at least one multi-site buffer spray between single-site systemic applications."
    else:
        status = "ANTI-RESISTANCE COMPLIANT"
        badge_color = "#10b981"
        rec_text = "Excellent mode-of-action stewardship. Your rotation successfully breaks fungal mutation cycles and preserves chemical efficacy."
        
    return {
        "status": status,
        "score": resistance_score,
        "badge_color": badge_color,
        "total_applications": len(applied_products),
        "distinct_frac_groups": distinct_fracs,
        "violations": violations,
        "history": resolved_history,
        "recommendation": rec_text
    }


def filter_chemicals(
    frac_group: str = None,
    max_rei_hours: int = None,
    max_phi_days: int = None,
    pollinator_safe_only: bool = False,
    max_resistance_risk: str = None
) -> list:
    """
    Filters registered EPA chemical formulations by FRAC mode of action group,
    REI reentry interval, PHI harvest safety, pollinator toxicity, and resistance risk.
    """
    results = []
    
    for _, item in EPA_REGISTRATION_REGISTRY.items():
        # 1. Filter by FRAC group (substring match)
        if frac_group and frac_group.strip() and frac_group.lower() not in item["frac_group"].lower():
            continue
            
        # 2. Filter by max REI
        if max_rei_hours is not None and item.get("rei_hours", 24) > max_rei_hours:
            continue
            
        # 3. Filter by max PHI
        if max_phi_days is not None and item.get("phi_days", 14) > max_phi_days:
            continue
            
        # 4. Filter by pollinator safety
        if pollinator_safe_only:
            poll_text = item.get("pollinator_safety", "").lower()
            if not ("bee safe" in poll_text or "non-toxic" in poll_text or "low toxicity" in poll_text or "virtually non-toxic" in poll_text):
                continue
                
        # 5. Filter by resistance risk
        if max_resistance_risk:
            risk = item.get("resistance_risk", "MEDIUM").upper()
            req = max_resistance_risk.strip().upper()
            if req == "LOW" and ("HIGH" in risk or "MEDIUM" in risk):
                continue
            elif req == "MEDIUM" and "HIGH" in risk:
                continue
                
        results.append(item)
        
    return results
