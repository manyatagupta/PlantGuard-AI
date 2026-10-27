"""
PlantGuard AI - Agricultural Treatment Cost & Economic Loss Estimator
Computes commercial product purchase quantities, application labor cost,
potential yield loss prevented, and financial Return on Investment (ROI).
"""

import math

# Market benchmark prices for standard formulation packs
DEFAULT_PRODUCT_PRICING = {
    "Neem Oil (Cold-Pressed 70%)": {"pack_size": 500, "unit": "ml", "price": 24.50},
    "Copper Octanoate (Liquid Copper Soap)": {"pack_size": 1000, "unit": "ml", "price": 29.00},
    "Bacillus subtilis (Biofungicide - Serenade)": {"pack_size": 1000, "unit": "ml", "price": 38.00},
    "Potassium Bicarbonate (MilStop / GreenCure)": {"pack_size": 450, "unit": "g", "price": 22.00},
    "Sulfur 80% Wettable Powder": {"pack_size": 1000, "unit": "g", "price": 18.50},
    "Mancozeb 75% DF (Dithane M-45)": {"pack_size": 1000, "unit": "g", "price": 32.00},
    "Chlorothalonil 720 S-C (Daconil)": {"pack_size": 1000, "unit": "ml", "price": 42.00},
    "Captan 50% WP (Fungicide)": {"pack_size": 1000, "unit": "g", "price": 36.00},
    "Myclobutanil (Immunox / Rally)": {"pack_size": 480, "unit": "ml", "price": 46.00},
    "Balanced N-P-K (20-20-20) Foliar Feed": {"pack_size": 1000, "unit": "g", "price": 16.00},
    "Chelated Calcium + Boron Foliar Tonic": {"pack_size": 1000, "unit": "ml", "price": 28.00},
    "Liquid Kelp & Seaweed Bio-Stimulant": {"pack_size": 1000, "unit": "ml", "price": 22.00}
}

# Average benchmark gross crop market value per unit
BENCHMARK_CROP_VALUES = {
    "Garden Bed (Sq. Feet)": 0.85,     # ~$0.85 gross harvest per sq.ft
    "Plant Count (Individual Plants)": 12.00, # ~$12.00 gross yield per plant
    "Greenhouse (Sq. Meters)": 18.50,  # ~$18.50 per m²
    "Farm Acreage (Acres)": 3800.00    # ~$3,800 gross harvest revenue per acre
}

def estimate_treatment_economics(
    product_name: str,
    active_amount: float,
    active_unit: str,
    area_type: str,
    area_val: float,
    custom_price_per_pack: float = None,
    custom_pack_size: float = None,
    custom_labor_rate: float = 18.0,
    crop_value_estimate: float = None,
    untreated_loss_pct: float = 35.0
) -> dict:
    """
    Computes financial breakdown of treatment application vs crop loss prevention.
    """
    bench = DEFAULT_PRODUCT_PRICING.get(
        product_name, 
        {"pack_size": 1000, "unit": active_unit, "price": 25.00}
    )
    
    pack_size = custom_pack_size if custom_pack_size and custom_pack_size > 0 else bench["pack_size"]
    price_per_pack = custom_price_per_pack if custom_price_per_pack and custom_price_per_pack > 0 else bench["price"]
    
    # 1. Product quantity & cost
    packs_needed = max(1, math.ceil(active_amount / pack_size))
    total_product_cost = round(packs_needed * price_per_pack, 2)
    leftover_amount = round((packs_needed * pack_size) - active_amount, 1)
    
    # 2. Labor & Application equipment cost
    if "Acre" in area_type:
        labor_hours = round(max(0.5, area_val * 1.25), 1)
    elif "Greenhouse" in area_type:
        labor_hours = round(max(0.3, (area_val / 200.0) * 1.0), 1)
    elif "Plant" in area_type:
        labor_hours = round(max(0.2, (area_val / 50.0) * 0.5), 1)
    else: # Sq. Feet
        labor_hours = round(max(0.2, (area_val / 1000.0) * 0.75), 1)
        
    labor_cost = round(labor_hours * custom_labor_rate, 2)
    total_investment = round(total_product_cost + labor_cost, 2)
    
    # 3. Gross Crop Value & Loss Mitigation
    if crop_value_estimate is not None and crop_value_estimate > 0:
        gross_crop_value = crop_value_estimate
    else:
        benchmark_rate = BENCHMARK_CROP_VALUES.get(area_type, 1.0)
        gross_crop_value = round(area_val * benchmark_rate, 2)
        
    potential_crop_loss = round(gross_crop_value * (untreated_loss_pct / 100.0), 2)
    net_economic_savings = round(max(0.0, potential_crop_loss - total_investment), 2)
    
    roi_ratio = round((potential_crop_loss / total_investment), 1) if total_investment > 0 else 1.0

    return {
        "product_name": product_name,
        "active_amount": active_amount,
        "active_unit": active_unit,
        "packs_needed": packs_needed,
        "pack_size": pack_size,
        "pack_unit": bench["unit"],
        "price_per_pack": price_per_pack,
        "total_product_cost": total_product_cost,
        "leftover_amount": leftover_amount,
        "labor_hours": labor_hours,
        "labor_cost": labor_cost,
        "total_investment": total_investment,
        "gross_crop_value": gross_crop_value,
        "potential_crop_loss": potential_crop_loss,
        "net_economic_savings": net_economic_savings,
        "roi_ratio": roi_ratio,
        "loss_pct": untreated_loss_pct
    }


def calculate_seasonal_spray_program_budget(
    treatment_schedule: list = None,
    area_type: str = "Garden Bed (Sq. Feet)",
    area_val: float = 250.0,
    custom_labor_rate: float = 18.0,
    crop_value_estimate: float = None,
    untreated_loss_pct: float = 40.0
) -> dict:
    """
    Computes cumulative seasonal budget across multiple phenological treatment stages.
    """
    if not treatment_schedule:
        treatment_schedule = [
            {"stage": "Dormant / Green Tip", "product": "Copper Octanoate (Liquid Copper Soap)", "sprays": 1},
            {"stage": "Pink Bud / Pre-Bloom", "product": "Neem Oil (Cold-Pressed 70%)", "sprays": 2},
            {"stage": "Petal Fall / Early Fruit", "product": "Bacillus subtilis (Biofungicide - Serenade)", "sprays": 2},
            {"stage": "Mid-Summer Canopy Maintenance", "product": "Potassium Bicarbonate (MilStop / GreenCure)", "sprays": 2}
        ]
        
    stage_breakdowns = []
    total_chem_cost = 0.0
    total_labor_hours = 0.0
    
    # Estimate water volume per spray pass
    if "Acre" in area_type:
        base_liters_per_pass = area_val * 400.0
    elif "Greenhouse" in area_type:
        base_liters_per_pass = area_val * 0.15
    elif "Plant" in area_type:
        base_liters_per_pass = area_val * 0.5
    else: # Sq. Feet
        base_liters_per_pass = (area_val / 250.0) * 15.0
        
    for item in treatment_schedule:
        stg = item.get("stage", "General Application")
        prod = item.get("product", "Neem Oil (Cold-Pressed 70%)")
        num_sprays = item.get("sprays", 1)
        
        # Approximate rate: 5 ml/L or 3 g/L
        rate = 5.0
        active_per_spray = base_liters_per_pass * rate
        total_active = active_per_spray * num_sprays
        
        bench = DEFAULT_PRODUCT_PRICING.get(prod, {"pack_size": 1000, "unit": "ml", "price": 28.00})
        packs = max(1, math.ceil(total_active / bench["pack_size"]))
        chem_cost = round(packs * bench["price"], 2)
        
        # Labor per pass
        if "Acre" in area_type:
            pass_labor = round(max(0.5, area_val * 1.0), 1)
        elif "Greenhouse" in area_type:
            pass_labor = round(max(0.3, (area_val / 200.0) * 0.8), 1)
        elif "Plant" in area_type:
            pass_labor = round(max(0.2, (area_val / 50.0) * 0.4), 1)
        else:
            pass_labor = round(max(0.2, (area_val / 1000.0) * 0.5), 1)
            
        stage_labor_hours = pass_labor * num_sprays
        stage_labor_cost = round(stage_labor_hours * custom_labor_rate, 2)
        
        total_chem_cost += chem_cost
        total_labor_hours += stage_labor_hours
        
        stage_breakdowns.append({
            "stage": stg,
            "product": prod,
            "passes": num_sprays,
            "active_needed": round(total_active, 1),
            "unit": bench["unit"],
            "packs": packs,
            "chemical_cost": chem_cost,
            "labor_hours": stage_labor_hours,
            "labor_cost": stage_labor_cost,
            "stage_total": round(chem_cost + stage_labor_cost, 2)
        })
        
    total_labor_cost = round(total_labor_hours * custom_labor_rate, 2)
    grand_total_investment = round(total_chem_cost + total_labor_cost, 2)
    
    # Crop Valuation
    if crop_value_estimate is not None and crop_value_estimate > 0:
        gross_val = crop_value_estimate
    else:
        benchmark_rate = BENCHMARK_CROP_VALUES.get(area_type, 1.0)
        gross_val = round(area_val * benchmark_rate, 2)
        
    potential_loss = round(gross_val * (untreated_loss_pct / 100.0), 2)
    net_savings = round(max(0.0, potential_loss - grand_total_investment), 2)
    roi_multiplier = round(potential_loss / grand_total_investment, 1) if grand_total_investment > 0 else 1.0
    
    return {
        "area_type": area_type,
        "area_val": area_val,
        "stages": stage_breakdowns,
        "total_applications": sum(s["passes"] for s in stage_breakdowns),
        "total_chemical_cost": round(total_chem_cost, 2),
        "total_labor_hours": round(total_labor_hours, 1),
        "total_labor_cost": total_labor_cost,
        "grand_total_investment": grand_total_investment,
        "gross_crop_value": gross_val,
        "potential_loss_avoided": potential_loss,
        "net_economic_savings": net_savings,
        "roi_multiplier": roi_multiplier
    }
