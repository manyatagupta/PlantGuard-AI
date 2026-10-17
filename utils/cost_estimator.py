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
