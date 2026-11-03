"""
PlantGuard AI - Precision Spatial GIS Field Mapping & Variable-Rate Spraying Engine
Computes Inverse Distance Weighting (IDW) spatial disease severity interpolation,
classifies field polygons into Variable-Rate Application (VRA) prescription zones,
calculates agrochemical & financial cost savings, and exports RFC-7946 GeoJSON,
Google Earth KML, and machine-rate CSV controller matrices.
"""

import math
import io
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple

# Pre-configured demonstration farm blocks with real-world spatial geometries
DEMO_FARM_BLOCKS = {
    "North Valley Vineyard (Block 7)": {
        "crop": "Grape / Vineyard",
        "primary_threat": "Grape Downy Mildew / Black Rot",
        "center_lat": 38.5432,
        "center_lon": -121.7456,
        "width_meters": 240.0,
        "length_meters": 300.0,
        "area_acres": 17.8,
        "recommended_product": "Azoxystrobin 22.9% (Quadris / Heritage)",
        "base_application_rate_l_ha": 2.2,
        "scout_points": [
            {"id": "SP-01", "offset_x": 30, "offset_y": 40, "disease": "Downy Mildew", "damage_pct": 34.5, "severity_grade": "Grade 3"},
            {"id": "SP-02", "offset_x": 60, "offset_y": 70, "disease": "Downy Mildew", "damage_pct": 28.0, "severity_grade": "Grade 3"},
            {"id": "SP-03", "offset_x": 120, "offset_y": 140, "disease": "Mildew Leaf Spot", "damage_pct": 14.2, "severity_grade": "Grade 2"},
            {"id": "SP-04", "offset_x": 180, "offset_y": 200, "disease": "Healthy", "damage_pct": 2.1, "severity_grade": "Grade 0"},
            {"id": "SP-05", "offset_x": 210, "offset_y": 250, "disease": "Healthy", "damage_pct": 0.5, "severity_grade": "Grade 0"},
            {"id": "SP-06", "offset_x": 50, "offset_y": 240, "disease": "Healthy", "damage_pct": 1.2, "severity_grade": "Grade 0"},
            {"id": "SP-07", "offset_x": 90, "offset_y": 20, "disease": "Downy Mildew", "damage_pct": 22.4, "severity_grade": "Grade 2"}
        ]
    },
    "Crestview Tomato Basin (Plot 4-East)": {
        "crop": "Tomato",
        "primary_threat": "Tomato Early Blight",
        "center_lat": 36.7783,
        "center_lon": -119.4179,
        "width_meters": 200.0,
        "length_meters": 200.0,
        "area_acres": 9.88,
        "recommended_product": "Chlorothalonil 75 WP (Daconil Broad-Spectrum)",
        "base_application_rate_l_ha": 2.5,
        "scout_points": [
            {"id": "SP-01", "offset_x": 20, "offset_y": 20, "disease": "Early Blight", "damage_pct": 42.0, "severity_grade": "Grade 4"},
            {"id": "SP-02", "offset_x": 40, "offset_y": 60, "disease": "Early Blight", "damage_pct": 31.5, "severity_grade": "Grade 3"},
            {"id": "SP-03", "offset_x": 80, "offset_y": 100, "disease": "Early Blight", "damage_pct": 16.0, "severity_grade": "Grade 2"},
            {"id": "SP-04", "offset_x": 140, "offset_y": 140, "disease": "Healthy", "damage_pct": 3.0, "severity_grade": "Grade 0"},
            {"id": "SP-05", "offset_x": 180, "offset_y": 180, "disease": "Healthy", "damage_pct": 0.8, "severity_grade": "Grade 0"}
        ]
    },
    "West Ridge Apple Orchard (Quadrant B)": {
        "crop": "Apple",
        "primary_threat": "Apple Scab",
        "center_lat": 44.0582,
        "center_lon": -123.0868,
        "width_meters": 300.0,
        "length_meters": 350.0,
        "area_acres": 25.9,
        "recommended_product": "Captan 50 WP",
        "base_application_rate_l_ha": 3.0,
        "scout_points": [
            {"id": "SP-01", "offset_x": 270, "offset_y": 40, "disease": "Apple Scab", "damage_pct": 38.0, "severity_grade": "Grade 3"},
            {"id": "SP-02", "offset_x": 240, "offset_y": 80, "disease": "Apple Scab", "damage_pct": 26.5, "severity_grade": "Grade 3"},
            {"id": "SP-03", "offset_x": 160, "offset_y": 150, "disease": "Apple Scab", "damage_pct": 12.0, "severity_grade": "Grade 2"},
            {"id": "SP-04", "offset_x": 50, "offset_y": 280, "disease": "Healthy", "damage_pct": 1.5, "severity_grade": "Grade 0"},
            {"id": "SP-05", "offset_x": 100, "offset_y": 220, "disease": "Healthy", "damage_pct": 2.0, "severity_grade": "Grade 0"}
        ]
    }
}


def meters_to_lat_lon_offsets(dx_meters: float, dy_meters: float, base_lat: float) -> Tuple[float, float]:
    """
    Converts local Cartesian coordinate offsets (meters east dx, meters north dy)
    to WGS84 geographic coordinate shifts (dLon, dLat).
    """
    # 1 deg latitude approx 111,139 meters
    d_lat = dy_meters / 111139.0
    # 1 deg longitude approx 111,139 * cos(lat) meters
    lat_rad = math.radians(base_lat)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        cos_lat = 1.0
    d_lon = dx_meters / (111139.0 * cos_lat)
    return round(d_lat, 7), round(d_lon, 7)


def interpolate_idw(
    target_x: float,
    target_y: float,
    sample_points: List[Dict[str, Any]],
    power: float = 2.0
) -> float:
    """
    Computes Inverse Distance Weighting (IDW) interpolation of disease severity (%)
    at target coordinates (target_x, target_y) given a list of scouted sampling points.
    """
    if not sample_points:
        return 0.0

    sum_weights = 0.0
    weighted_sum = 0.0

    for pt in sample_points:
        px = float(pt.get("offset_x", 0.0))
        py = float(pt.get("offset_y", 0.0))
        val = float(pt.get("damage_pct", 0.0))

        dist = math.hypot(target_x - px, target_y - py)

        # Exact coincidence with sample location
        if dist < 1.0:
            return round(val, 2)

        weight = 1.0 / (dist ** power)
        sum_weights += weight
        weighted_sum += weight * val

    if sum_weights > 0.0:
        return round(weighted_sum / sum_weights, 2)
    return 0.0


def generate_field_prescription_grid(
    farm_block_name: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    grid_resolution_meters: float = 25.0,
    curative_threshold_pct: float = 20.0,
    protective_threshold_pct: float = 8.0,
    chemical_cost_per_liter: float = 34.0
) -> Dict[str, Any]:
    """
    Discretizes a farm block into spatial micro-zones, computes IDW interpolated disease severity,
    assigns precision Variable-Rate Application (VRA) zones, and calculates agrochemical & financial savings.
    """
    if farm_block_name and farm_block_name in DEMO_FARM_BLOCKS:
        block = DEMO_FARM_BLOCKS[farm_block_name]
    elif custom_config:
        block = custom_config
    else:
        block = list(DEMO_FARM_BLOCKS.values())[0]

    center_lat = float(block.get("center_lat", 38.5432))
    center_lon = float(block.get("center_lon", -121.7456))
    width_m = float(block.get("width_meters", 200.0))
    length_m = float(block.get("length_meters", 200.0))
    crop_name = block.get("crop", "Crops")
    primary_threat = block.get("primary_threat", "Fungal Pathogen")
    product_name = block.get("recommended_product", "Quadris / Heritage (Azoxystrobin)")
    full_rate_l_ha = float(block.get("base_application_rate_l_ha", 2.5))
    scout_pts = block.get("scout_points", [])

    # Enrich scout points with geographic lat/lon
    enriched_scouts = []
    for sp in scout_pts:
        ox = float(sp["offset_x"])
        oy = float(sp["offset_y"])
        dlat, dlon = meters_to_lat_lon_offsets(ox - (width_m / 2.0), oy - (length_m / 2.0), center_lat)
        enriched_scouts.append({
            "id": sp.get("id", "SP"),
            "offset_x": ox,
            "offset_y": oy,
            "latitude": round(center_lat + dlat, 6),
            "longitude": round(center_lon + dlon, 6),
            "disease": sp.get("disease", primary_threat),
            "damage_pct": float(sp.get("damage_pct", 0.0)),
            "severity_grade": sp.get("severity_grade", "Grade 1")
        })

    # Discretize field into grid cells
    cols = max(2, int(width_m / grid_resolution_meters))
    rows = max(2, int(length_m / grid_resolution_meters))
    cell_w = width_m / cols
    cell_h = length_m / rows
    cell_area_m2 = cell_w * cell_h
    cell_area_ha = cell_area_m2 / 10000.0
    total_area_ha = (width_m * length_m) / 10000.0
    total_area_acres = total_area_ha * 2.47105

    grid_cells = []
    zone_counts = {"Zone 1 (Curative 100%)": 0, "Zone 2 (Protective 50%)": 0, "Zone 3 (No-Spray 0%)": 0}
    total_vra_chemical_liters = 0.0

    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            
            # Interpolated damage %
            interp_damage = interpolate_idw(cx, cy, scout_pts, power=2.0)

            # Assign VRA zone
            if interp_damage >= curative_threshold_pct:
                zone_name = "Zone 1 (Curative 100%)"
                rate_multiplier = 1.00
                color_hex = "#ef4444"
                action_text = "100% Curative Rate + Penetrant Adjuvant"
            elif interp_damage >= protective_threshold_pct:
                zone_name = "Zone 2 (Protective 50%)"
                rate_multiplier = 0.50
                color_hex = "#f59e0b"
                action_text = "50% Preventive Rate Barrier"
            else:
                zone_name = "Zone 3 (No-Spray 0%)"
                rate_multiplier = 0.00
                color_hex = "#10b981"
                action_text = "0% Skip Spray (Refuge / Clean)"

            zone_counts[zone_name] += 1
            cell_chem_liters = full_rate_l_ha * rate_multiplier * cell_area_ha
            total_vra_chemical_liters += cell_chem_liters

            # Cell 4-corner polygon bounding box in Lat/Lon
            x_min = c * cell_w - (width_m / 2.0)
            x_max = (c + 1) * cell_w - (width_m / 2.0)
            y_min = r * cell_h - (length_m / 2.0)
            y_max = (r + 1) * cell_h - (length_m / 2.0)

            lat_min, lon_min = meters_to_lat_lon_offsets(x_min, y_min, center_lat)
            lat_max, lon_max = meters_to_lat_lon_offsets(x_max, y_max, center_lat)

            # Polygon ring (closed loop)
            poly_coords = [
                [center_lon + lon_min, center_lat + lat_min],
                [center_lon + lon_max, center_lat + lat_min],
                [center_lon + lon_max, center_lat + lat_max],
                [center_lon + lon_min, center_lat + lat_max],
                [center_lon + lon_min, center_lat + lat_min]
            ]

            cell_dlat, cell_dlon = meters_to_lat_lon_offsets(cx - (width_m / 2.0), cy - (length_m / 2.0), center_lat)

            grid_cells.append({
                "row": r,
                "col": c,
                "center_lat": round(center_lat + cell_dlat, 6),
                "center_lon": round(center_lon + cell_dlon, 6),
                "damage_pct": interp_damage,
                "zone_name": zone_name,
                "rate_multiplier": rate_multiplier,
                "rate_l_ha": round(full_rate_l_ha * rate_multiplier, 2),
                "chemical_liters": round(cell_chem_liters, 3),
                "color_hex": color_hex,
                "action_text": action_text,
                "polygon_coords": poly_coords
            })

    # Economic and Environmental Savings
    uniform_total_liters = round(full_rate_l_ha * total_area_ha, 2)
    vra_total_liters = round(total_vra_chemical_liters, 2)
    liters_saved = max(0.0, round(uniform_total_liters - vra_total_liters, 2))
    percent_saved = round((liters_saved / uniform_total_liters * 100.0) if uniform_total_liters > 0 else 0.0, 1)

    uniform_cost = round(uniform_total_liters * chemical_cost_per_liter, 2)
    vra_cost = round(vra_total_liters * chemical_cost_per_liter, 2)
    cost_saved = round(uniform_cost - vra_cost, 2)
    savings_per_acre = round((cost_saved / total_area_acres) if total_area_acres > 0 else 0.0, 2)

    # Perimeter polygon of the whole field block
    half_w = width_m / 2.0
    half_l = length_m / 2.0
    bl_lat, bl_lon = meters_to_lat_lon_offsets(-half_w, -half_l, center_lat)
    br_lat, br_lon = meters_to_lat_lon_offsets(half_w, -half_l, center_lat)
    tr_lat, tr_lon = meters_to_lat_lon_offsets(half_w, half_l, center_lat)
    tl_lat, tl_lon = meters_to_lat_lon_offsets(-half_w, half_l, center_lat)

    field_boundary = [
        [center_lon + bl_lon, center_lat + bl_lat],
        [center_lon + br_lon, center_lat + br_lat],
        [center_lon + tr_lon, center_lat + tr_lat],
        [center_lon + tl_lon, center_lat + tl_lat],
        [center_lon + bl_lon, center_lat + bl_lat]
    ]

    return {
        "farm_block_name": farm_block_name or "Custom Field Block",
        "crop": crop_name,
        "primary_threat": primary_threat,
        "recommended_product": product_name,
        "base_application_rate_l_ha": full_rate_l_ha,
        "center_latitude": center_lat,
        "center_longitude": center_lon,
        "width_meters": width_m,
        "length_meters": length_m,
        "total_area_hectares": round(total_area_ha, 2),
        "total_area_acres": round(total_area_acres, 2),
        "grid_resolution_meters": grid_resolution_meters,
        "total_cells": len(grid_cells),
        "zone_distribution": zone_counts,
        "scout_points": enriched_scouts,
        "grid_cells": grid_cells,
        "field_boundary": field_boundary,
        "economics": {
            "chemical_cost_per_liter": chemical_cost_per_liter,
            "uniform_broadcast_liters": uniform_total_liters,
            "vra_precision_liters": vra_total_liters,
            "liters_saved": liters_saved,
            "chemical_reduction_pct": percent_saved,
            "uniform_total_cost": uniform_cost,
            "vra_total_cost": vra_cost,
            "dollars_saved": cost_saved,
            "savings_per_acre": savings_per_acre
        }
    }


def export_field_prescription_geojson(field_prescription: Dict[str, Any]) -> str:
    """
    Exports a field prescription model to standard RFC 7946 compliant GeoJSON.
    Includes polygon features for each VRA prescription cell and point features for scouted leaves.
    """
    features = []

    # 1. Field Perimeter Polygon
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [field_prescription["field_boundary"]]
        },
        "properties": {
            "feature_type": "Field_Boundary",
            "block_name": field_prescription["farm_block_name"],
            "crop": field_prescription["crop"],
            "area_acres": field_prescription["total_area_acres"],
            "chemical": field_prescription["recommended_product"]
        }
    })

    # 2. VRA Grid Cells
    for idx, cell in enumerate(field_prescription["grid_cells"]):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [cell["polygon_coords"]]
            },
            "properties": {
                "feature_type": "VRA_Prescription_Zone",
                "cell_id": f"CELL-{cell['row']:02d}-{cell['col']:02d}",
                "zone_name": cell["zone_name"],
                "rate_multiplier": cell["rate_multiplier"],
                "target_rate_l_ha": cell["rate_l_ha"],
                "chemical_liters": cell["chemical_liters"],
                "interpolated_damage_pct": cell["damage_pct"],
                "action": cell["action_text"],
                "stroke": cell["color_hex"],
                "fill": cell["color_hex"]
            }
        })

    # 3. Scout Points
    for sp in field_prescription.get("scout_points", []):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [sp["longitude"], sp["latitude"]]
            },
            "properties": {
                "feature_type": "Scout_Sample_Marker",
                "scout_id": sp["id"],
                "pathology": sp["disease"],
                "foliar_damage_pct": sp["damage_pct"],
                "severity_grade": sp["severity_grade"]
            }
        })

    geojson_doc = {
        "type": "FeatureCollection",
        "metadata": {
            "generator": "PlantGuard AI Precision GIS Engine",
            "generated_at": datetime.datetime.now().isoformat(),
            "farm_block": field_prescription["farm_block_name"],
            "crop": field_prescription["crop"],
            "chemical_savings_pct": field_prescription["economics"]["chemical_reduction_pct"],
            "dollars_saved": field_prescription["economics"]["dollars_saved"]
        },
        "features": features
    }

    return json.dumps(geojson_doc, indent=2)


def export_field_prescription_kml(field_prescription: Dict[str, Any]) -> str:
    """
    Exports the field prescription model to standard Google Earth KML format
    with styled color-coded polygons and inspection placemarks.
    """
    block_name = field_prescription.get("farm_block_name", "Field")
    crop = field_prescription.get("crop", "Crop")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '  <Document>',
        f'    <name>PlantGuard AI - {block_name} VRA Prescription</name>',
        f'    <description>Variable-rate spray prescription zones and scouting markers for {crop}. Generated on {now_str}.</description>',
        '    <Style id="polyRed">',
        '      <LineStyle><color>ff4444ef</color><width>2</width></LineStyle>',
        '      <PolyStyle><color>7f4444ef</color></PolyStyle>',
        '    </Style>',
        '    <Style id="polyAmber">',
        '      <LineStyle><color>ff0b9ef5</color><width>2</width></LineStyle>',
        '      <PolyStyle><color>7f0b9ef5</color></PolyStyle>',
        '    </Style>',
        '    <Style id="polyGreen">',
        '      <LineStyle><color>ff81b910</color><width>2</width></LineStyle>',
        '      <PolyStyle><color>7f81b910</color></PolyStyle>',
        '    </Style>',
        '    <Folder>',
        '      <name>VRA Prescription Zones</name>'
    ]

    for cell in field_prescription["grid_cells"]:
        style = "polyRed" if cell["rate_multiplier"] >= 1.0 else ("polyAmber" if cell["rate_multiplier"] > 0.0 else "polyGreen")
        coords_str = " ".join([f"{pt[0]},{pt[1]},0" for pt in cell["polygon_coords"]])

        kml_lines.extend([
            '      <Placemark>',
            f'        <name>{cell["zone_name"]} (Row {cell["row"]}, Col {cell["col"]})</name>',
            f'        <description>Target Rate: {cell["rate_l_ha"]} L/ha | Damage: {cell["damage_pct"]}% | Action: {cell["action_text"]}</description>',
            f'        <styleUrl>#{style}</styleUrl>',
            '        <Polygon>',
            '          <outerBoundaryIs>',
            '            <LinearRing>',
            f'              <coordinates>{coords_str}</coordinates>',
            '            </LinearRing>',
            '          </outerBoundaryIs>',
            '        </Polygon>',
            '      </Placemark>'
        ])

    kml_lines.append('    </Folder>')
    kml_lines.append('    <Folder>')
    kml_lines.append('      <name>Scout Sampling Points</name>')

    for sp in field_prescription.get("scout_points", []):
        kml_lines.extend([
            '      <Placemark>',
            f'        <name>{sp["id"]}: {sp["disease"]}</name>',
            f'        <description>Foliar Damage: {sp["damage_pct"]}% ({sp["severity_grade"]})</description>',
            '        <Point>',
            f'          <coordinates>{sp["longitude"]},{sp["latitude"]},0</coordinates>',
            '        </Point>',
            '      </Placemark>'
        ])

    kml_lines.extend([
        '    </Folder>',
        '  </Document>',
        '</kml>'
    ])

    return "\n".join(kml_lines)


def export_rate_controller_csv(field_prescription: Dict[str, Any]) -> str:
    """
    Exports a machine-ready CSV matrix for tractor/drone rate controllers (Raven, Trimble, John Deere GreenStar).
    Format complies with standard RFC-4180 with UTF-8 BOM.
    """
    output = io.StringIO()
    output.write("\ufeff")
    output.write("Grid_ID,Row,Col,Latitude,Longitude,Interpolated_Damage_Pct,VRA_Zone,Target_Rate_L_ha,Chemical_Volume_Liters\r\n")

    for cell in field_prescription["grid_cells"]:
        grid_id = f"CELL-{cell['row']:02d}-{cell['col']:02d}"
        output.write(
            f"{grid_id},{cell['row']},{cell['col']},{cell['center_lat']:.6f},{cell['center_lon']:.6f},"
            f"{cell['damage_pct']:.1f},\"{cell['zone_name']}\",{cell['rate_l_ha']:.2f},{cell['chemical_liters']:.3f}\r\n"
        )

    return output.getvalue()
