"""
PlantGuard AI - Offline Agricultural Field Scouting Cards & Export Utilities
Generates printable inspection audit cards, IPM scouting sheets,
and CSV diagnostic telemetry for farm certification records.
"""

import csv
import io
import datetime

def export_predictions_to_csv_string(predictions: list) -> str:
    """
    Exports a list of prediction records to a standard RFC 4180 CSV string with UTF-8 BOM.
    """
    output = io.StringIO()
    # Write UTF-8 BOM for Microsoft Excel auto-encoding
    output.write('\ufeff')
    
    fieldnames = ["ID", "Timestamp", "Specimen Filename", "Diagnosis", "Confidence (%)", "Health Status", "Canopy Damage (%)", "Severity Grade"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for idx, row in enumerate(predictions, 1):
        writer.writerow({
            "ID": idx,
            "Timestamp": row.get("timestamp", "N/A"),
            "Specimen Filename": row.get("filename", "N/A"),
            "Diagnosis": row.get("class", "Unknown"),
            "Confidence (%)": f"{float(row.get('conf', 0.0)) * 100:.1f}",
            "Health Status": "Healthy" if row.get("is_healthy") else "Infected",
            "Canopy Damage (%)": f"{float(row.get('damage_pct', 0.0)):.1f}",
            "Severity Grade": row.get("severity_grade", "N/A")
        })
        
    return output.getvalue()

def generate_field_audit_checklist_html(
    farm_name: str = "Valley Crest Agricultural Farm",
    inspector_name: str = "Certified Crop Advisor (CCA)",
    target_crops: list = None,
    plot_id: str = "Block 4-B (North Quadrant)"
) -> str:
    """
    Generates a high-contrast, printable HTML/CSS Field Scouting Checklist
    ready for physical clipboard printing or offline tablet auditing.
    """
    if not target_crops:
        target_crops = ["Tomato", "Apple", "Potato", "Corn", "Bell Pepper"]
        
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    crops_html = "".join([f"<span class='badge'>{c}</span>" for c in target_crops])
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PlantGuard AI - Field Scouting & IPM Audit Sheet</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b; line-height: 1.5; padding: 30px; }}
    .header {{ border-bottom: 3px solid #10b981; padding-bottom: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; }}
    h1 {{ margin: 0; color: #065f46; font-size: 24px; }}
    .badge {{ background: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-right: 5px; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 10px 12px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; font-weight: 600; color: #334155; }}
    .checkbox-box {{ width: 16px; height: 16px; border: 1.5px solid #64748b; display: inline-block; vertical-align: middle; margin-right: 6px; }}
    .section-title {{ font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 25px; border-left: 4px solid #10b981; padding-left: 10px; }}
    .notes-box {{ border: 1px dashed #94a3b8; height: 70px; margin-top: 10px; border-radius: 4px; }}
    @media print {{
        body {{ padding: 0; }}
        button {{ display: none; }}
    }}
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>🌿 PlantGuard AI — Field Scouting & IPM Audit Sheet</h1>
        <div style="color: #64748b; font-size: 13px; margin-top: 5px;">Integrated Pest Management (IPM) & Phytosanitary Protocol</div>
    </div>
    <div style="text-align: right; font-size: 12px; color: #64748b;">
        <div><strong>Date:</strong> {now_str}</div>
        <div><strong>Plot / Block:</strong> {plot_id}</div>
    </div>
</div>

<table style="margin-bottom: 20px;">
    <tr>
        <th style="width: 25%;">Farm / Orchard Name</th>
        <td>{farm_name}</td>
        <th style="width: 25%;">Field Scout / Inspector</th>
        <td>{inspector_name}</td>
    </tr>
    <tr>
        <th>Target Crop(s) Monitored</th>
        <td colspan="3">{crops_html}</td>
    </tr>
</table>

<div class="section-title">1. Visual Foliar Lesion & Disease Scouting Matrix</div>
<table>
    <thead>
        <tr>
            <th style="width: 20%;">Scout Sample Point</th>
            <th style="width: 25%;">Symptoms Observed</th>
            <th style="width: 25%;">Suspected Pathogen / Abiotic Stress</th>
            <th style="width: 15%;">Severity Grade</th>
            <th style="width: 15%;">Action Triggered</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Point A (Border / Perimeter)</td>
            <td>&nbsp;</td>
            <td>&nbsp;</td>
            <td>[ ] 0 &nbsp; [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4</td>
            <td>[ ] Routine &nbsp; [ ] Spray</td>
        </tr>
        <tr>
            <td>Point B (Canopy Center)</td>
            <td>&nbsp;</td>
            <td>&nbsp;</td>
            <td>[ ] 0 &nbsp; [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4</td>
            <td>[ ] Routine &nbsp; [ ] Spray</td>
        </tr>
        <tr>
            <td>Point C (Low Elevation / Wet)</td>
            <td>&nbsp;</td>
            <td>&nbsp;</td>
            <td>[ ] 0 &nbsp; [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4</td>
            <td>[ ] Routine &nbsp; [ ] Spray</td>
        </tr>
        <tr>
            <td>Point D (Windward Edge)</td>
            <td>&nbsp;</td>
            <td>&nbsp;</td>
            <td>[ ] 0 &nbsp; [ ] 1 &nbsp; [ ] 2 &nbsp; [ ] 3 &nbsp; [ ] 4</td>
            <td>[ ] Routine &nbsp; [ ] Spray</td>
        </tr>
    </tbody>
</table>

<div class="section-title">2. Chemical Application & WALES Tank-Mix Compliance Verification</div>
<table>
    <tr>
        <td><span class="checkbox-box"></span> Water carrier volume calibrated per canopy area</td>
        <td><span class="checkbox-box"></span> Tank sequence follows strict WALES order (WP &rarr; A &rarr; L &rarr; E &rarr; S)</td>
    </tr>
    <tr>
        <td><span class="checkbox-box"></span> Water carrier pH verified between 6.0 &ndash; 7.0</td>
        <td><span class="checkbox-box"></span> 30-day Sulfur and Oil spacing rule observed (No phytotoxicity hazard)</td>
    </tr>
    <tr>
        <td><span class="checkbox-box"></span> Wind speed &lt; 15 km/h (No off-target drift risk)</td>
        <td><span class="checkbox-box"></span> Required PPE (Respirator, Nitrile gloves, Face shield) donned</td>
    </tr>
</table>

<div class="section-title">3. Agronomic Scout Notes & Recommendations</div>
<div class="notes-box"></div>

<div style="margin-top: 25px; display: flex; justify-content: space-between; font-size: 13px;">
    <div><strong>Inspector Signature:</strong> ___________________________</div>
    <div><strong>Farm Manager Approval:</strong> ___________________________</div>
</div>

<div style="text-align: center; margin-top: 30px;">
    <button onclick="window.print()" style="background: #10b981; color: white; border: none; padding: 10px 24px; font-weight: 600; border-radius: 6px; cursor: pointer;">🖨️ Print Scouting Audit Card</button>
</div>
</body>
</html>
"""
