"""
PlantGuard AI - Comprehensive Automated Test Pipeline
Validates Computer Vision, Agronomic Logic, Financial Economics,
Recovery Tracking, Weather Modeling, and Database Operations.
"""

import unittest
import os
import sys
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.leaf_cv import (
    analyze_leaf_damage,
    calculate_canopy_spectral_indices,
    estimate_physical_leaf_dimensions,
)
from utils.cost_estimator import estimate_treatment_economics, DEFAULT_PRODUCT_PRICING
from utils.recovery_tracker import evaluate_treatment_recovery
from utils.weather_risk import calculate_disease_outbreak_risk, get_regional_monitoring_stations
from utils.agronomy_brain import query_agronomy_brain
from utils.crop_calendar import get_crop_calendar, PHENOLOGY_CALENDAR
from utils.dosage_calculator import compute_spray_dosage, get_all_products
from utils.db import add_prediction, get_recent_predictions, get_summary_stats


class TestLeafCV(unittest.TestCase):
    """Tests for leaf damage computer vision segmentation."""

    def test_synthetic_leaf_damage(self):
        # Create a synthetic image: Green background with a brown square
        img_arr = np.zeros((200, 200, 3), dtype=np.uint8)
        # Fill with healthy leaf green
        img_arr[:, :] = [34, 139, 34]
        # Draw a brown necrotic spot in the center
        img_arr[80:120, 80:120] = [139, 69, 19]
        
        pil_img = Image.fromarray(img_arr)
        result = analyze_leaf_damage(pil_img)
        
        self.assertIn("damage_percentage", result)
        self.assertIn("stage_label", result)
        self.assertIn("severity_color", result)
        self.assertIn("severity_badge", result)
        self.assertGreater(result["damage_percentage"], 0.0)
        self.assertGreaterEqual(result["lesion_count"], 1)
        self.assertIsNotNone(result["overlay_image"])

    def test_sample_leaves_if_present(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "assets", "samples", "healthy_leaf.jpg")
        if os.path.exists(sample_path):
            img = Image.open(sample_path)
            res = analyze_leaf_damage(img)
            self.assertLess(res["damage_percentage"], 15.0)


class TestCostEstimator(unittest.TestCase):
    """Tests for treatment commercial economics and ROI estimator."""

    def test_cost_calculation_basic(self):
        res = estimate_treatment_economics(
            product_name="Neem Oil (Cold-Pressed 70%)",
            active_amount=250.0,
            active_unit="ml",
            area_type="Garden Bed (Sq. Feet)",
            area_val=250.0
        )
        self.assertEqual(res["packs_needed"], 1)
        self.assertGreater(res["total_product_cost"], 0.0)
        self.assertGreater(res["labor_cost"], 0.0)
        self.assertGreater(res["roi_ratio"], 0.0)
        self.assertIn("pack_size", res)

    def test_cost_calculation_farm_scale(self):
        res = estimate_treatment_economics(
            product_name="Copper Octanoate (Liquid Copper Soap)",
            active_amount=4000.0,
            active_unit="ml",
            area_type="Farm Acreage (Acres)",
            area_val=2.0,
            custom_labor_rate=20.0
        )
        self.assertEqual(res["packs_needed"], 4)
        self.assertEqual(res["total_product_cost"], 4 * 29.0)
        self.assertGreater(res["potential_crop_loss"], 1000.0)
        self.assertGreaterEqual(res["roi_ratio"], 1.0)


class TestRecoveryTracker(unittest.TestCase):
    """Tests for dual scan post-treatment healing evaluation."""

    def test_recovery_evaluation(self):
        # Specimen 1 (high damage)
        img1_arr = np.zeros((150, 150, 3), dtype=np.uint8)
        img1_arr[:, :] = [34, 139, 34]
        img1_arr[40:110, 40:110] = [139, 69, 19]
        
        # Specimen 2 (healed/low damage)
        img2_arr = np.zeros((150, 150, 3), dtype=np.uint8)
        img2_arr[:, :] = [34, 139, 34]
        img2_arr[70:80, 70:80] = [139, 69, 19]

        res = evaluate_treatment_recovery(Image.fromarray(img1_arr), Image.fromarray(img2_arr))
        
        self.assertGreater(res["dmg_before"], res["dmg_after"])
        self.assertGreater(res["recovery_pct"], 50.0)
        self.assertIn("status", res)
        self.assertIn("guidance", res)
        self.assertIsNotNone(res["composite_image"])


class TestWeatherRisk(unittest.TestCase):
    """Tests for meteorological outbreak forecasting and regional stations."""

    def test_high_risk_weather(self):
        # Warm, high humidity, rainfall, prolonged wetness
        res = calculate_disease_outbreak_risk(
            temp=23.0, humidity=92.0, rainfall=12.0, leaf_wetness=10.0, wind_speed=8.0
        )
        self.assertGreaterEqual(res["overall_score"], 60.0)
        self.assertIn("Severe Threat", res["overall_level"])
        self.assertIsNotNone(res["top_threat"])

    def test_regional_monitoring_stations(self):
        stations = get_regional_monitoring_stations(38.54, -121.74, "Davis", 65.0)
        self.assertEqual(len(stations), 5)
        for s in stations:
            self.assertIn("name", s)
            self.assertIn("score", s)
            self.assertIn("color", s)
            self.assertIn("radius", s)


class TestAgronomyBrain(unittest.TestCase):
    """Tests for local offline agronomic brain and tank mix rules."""

    def test_query_disease(self):
        res = query_agronomy_brain("tomato early blight")
        self.assertIn("Early Blight", res)
        self.assertIn("Organic", res)
        self.assertIn("Chemical", res)

    def test_sulfur_oil_compatibility(self):
        comp = query_agronomy_brain("Can I mix sulfur and neem oil?")
        self.assertIn("Incompatibility", comp)
        self.assertIn("30-day", comp)


class TestCropCalendar(unittest.TestCase):
    """Tests for phenological growth calendars."""

    def test_crops_supported(self):
        for crop in PHENOLOGY_CALENDAR.keys():
            stages = get_crop_calendar(crop)
            self.assertGreater(len(stages), 0)
            for stage in stages:
                self.assertIn("stage", stage)
                self.assertIn("threats", stage)
                self.assertIn("preventive_spray", stage)

    def test_phenology_stage_by_gdd(self):
        from utils.crop_calendar import get_phenology_stage_by_gdd
        stage_info = get_phenology_stage_by_gdd("Apple", cumulative_gdd=185.0)
        self.assertEqual(stage_info["crop"], "Apple")
        self.assertIn("Pink Bud", stage_info["stage_name"])
        self.assertGreater(len(stage_info["threats"]), 0)
        self.assertGreaterEqual(stage_info["gdd_to_next_stage"], 0.0)


class TestDosageCalculator(unittest.TestCase):
    """Tests for carrier volume and active ingredient calculations."""

    def test_compute_dosage(self):
        prods = get_all_products()
        self.assertGreater(len(prods), 0)
        
        res = compute_spray_dosage(
            product_name=prods[0],
            area_type="Garden Bed (Sq. Feet)",
            area_val=100.0,
            plant_stage="Medium Canopy"
        )
        self.assertGreater(res["water_liters"], 0.0)
        self.assertGreater(res["active_amount"], 0.0)
        self.assertGreater(len(res["mixing_steps"]), 0)


class TestDatabase(unittest.TestCase):
    """Tests for database persistence and schema migrations."""

    def test_add_and_retrieve_prediction(self):
        add_prediction(
            filename="test_leaf_sample.jpg",
            predicted_class="Tomato Early blight",
            confidence=0.95,
            is_healthy=False,
            damage_pct=14.5,
            severity_grade="Grade 2 (Moderate Infection)"
        )
        recent = get_recent_predictions(limit=1)
        self.assertGreater(len(recent), 0)
        latest = recent[0]
        self.assertEqual(latest["class"], "Tomato Early blight")
        self.assertEqual(latest["damage_pct"], 14.5)

        stats = get_summary_stats()
        self.assertGreater(stats["total"], 0)


class TestChemicalScanner(unittest.TestCase):
    """Tests for EPA chemical registration and FRAC resistance engine."""

    def test_lookup_by_epa(self):
        from utils.chemical_scanner import lookup_chemical_by_epa_or_name
        res = lookup_chemical_by_epa_or_name("100-1098")
        self.assertTrue(res["match_found"])
        self.assertEqual(res["record"]["trade_name"], "Heritage 50 WDG")
        self.assertIn("Group 11", res["record"]["frac_group"])

    def test_lookup_by_trade_name(self):
        from utils.chemical_scanner import lookup_chemical_by_epa_or_name
        res = lookup_chemical_by_epa_or_name("Trilogy")
        self.assertTrue(res["match_found"])
        self.assertIn("Neem", res["record"]["active_ingredient"])

    def test_frac_rotation_guidance(self):
        from utils.chemical_scanner import get_frac_rotation_guidance
        rot = get_frac_rotation_guidance("Group 11")
        self.assertEqual(rot["risk_level"], "HIGH")
        self.assertGreater(len(rot["recommended_rotation"]), 0)


class TestNutrientDiagnostics(unittest.TestCase):
    """Tests for nutrient deficiency diagnosis and disease differential."""

    def test_iron_chlorosis_diagnosis(self):
        from utils.nutrient_diagnostics import diagnose_nutrient_deficiency
        matches = diagnose_nutrient_deficiency(leaf_position="Upper young leaves", primary_symptom="Interveinal chlorosis yellow")
        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertEqual(top["nutrient"], "Iron (Fe)")
        self.assertIn("Immobile", top["mobility"])

    def test_disease_comparison(self):
        from utils.nutrient_diagnostics import get_deficiency_vs_disease_comparison
        comp = get_deficiency_vs_disease_comparison("Tomato Early Blight")
        self.assertTrue(comp["matched"])
        self.assertEqual(comp["mimicking_nutrient"], "Potassium (K)")
        self.assertIn("differential_rule", comp)

    def test_soil_ph_bioavailability_acidic(self):
        from utils.nutrient_diagnostics import calculate_soil_ph_bioavailability
        res = calculate_soil_ph_bioavailability(5.2, soil_texture="Loam")
        self.assertEqual(res["ph"], 5.2)
        self.assertIn("Acidic Soil Lockout", res["status"])
        self.assertLess(res["bioavailability"]["Phosphorus (P)"], 60.0)
        self.assertLess(res["bioavailability"]["Calcium (Ca)"], 60.0)
        self.assertGreater(res["lime_lbs_1000sqft"], 0.0)
        self.assertGreater(res["lime_kg_ha"], 0.0)
        self.assertEqual(res["sulfur_lbs_1000sqft"], 0.0)
        self.assertIn("Limestone", res["action_summary"])

    def test_soil_ph_bioavailability_alkaline(self):
        from utils.nutrient_diagnostics import calculate_soil_ph_bioavailability
        res = calculate_soil_ph_bioavailability(8.0, soil_texture="Clay")
        self.assertEqual(res["ph"], 8.0)
        self.assertIn("Alkaline Chlorosis Hazard", res["status"])
        self.assertLess(res["bioavailability"]["Iron (Fe)"], 35.0)
        self.assertLess(res["bioavailability"]["Manganese (Mn)"], 35.0)
        self.assertIn("Iron (Fe)", res["lockouts"])
        self.assertIn("Manganese (Mn)", res["lockouts"])
        self.assertGreater(res["sulfur_lbs_1000sqft"], 0.0)
        self.assertGreater(res["sulfur_kg_ha"], 0.0)
        self.assertEqual(res["lime_lbs_1000sqft"], 0.0)
        self.assertIn("Sulfur", res["action_summary"])

    def test_soil_ph_bioavailability_optimal(self):
        from utils.nutrient_diagnostics import calculate_soil_ph_bioavailability
        res = calculate_soil_ph_bioavailability(6.5, soil_texture="Sandy Loam")
        self.assertEqual(res["status"], "Optimal Bioavailability Range")
        self.assertEqual(len(res["lockouts"]), 0)
        self.assertEqual(res["lime_lbs_1000sqft"], 0.0)
        self.assertEqual(res["sulfur_lbs_1000sqft"], 0.0)
        self.assertGreaterEqual(res["bioavailability"]["Nitrogen (N)"], 90.0)
        self.assertGreaterEqual(res["bioavailability"]["Phosphorus (P)"], 85.0)


class TestTankMixCompatibility(unittest.TestCase):
    """Tests for spray tank chemical incompatibility detection and jar test simulator."""

    def test_calcium_sulfate_precipitation(self):
        from utils.dosage_calculator import check_tank_mix_compatibility
        res = check_tank_mix_compatibility(["Calcium Nitrate", "Epsom Salt (Magnesium Sulfate)"])
        self.assertIn(res["overall_safety"], ["WARNING", "DANGER"])
        self.assertGreaterEqual(len(res["conflicts"]), 1)
        conflict = res["conflicts"][0]
        self.assertIn("Precipitation", conflict["title"])
        self.assertIn("Gypsum", conflict["hazard"])

    def test_sulfur_oil_phytotoxicity(self):
        from utils.dosage_calculator import check_tank_mix_compatibility
        res = check_tank_mix_compatibility(["Wettable Sulfur (Micronized 80%)", "Neem Oil (Cold-Pressed 70%)"])
        self.assertEqual(res["overall_safety"], "DANGER")
        self.assertGreaterEqual(len(res["conflicts"]), 1)
        conflict = res["conflicts"][0]
        self.assertIn("Phytotoxicity", conflict["title"])
        self.assertEqual(conflict["withholding_days"], 30)

    def test_copper_biological_antagonism(self):
        from utils.dosage_calculator import check_tank_mix_compatibility
        res = check_tank_mix_compatibility(["Liquid Copper Fungicide", "Bacillus subtilis Bio-Fungicide"])
        self.assertIn(res["overall_safety"], ["WARNING", "DANGER"])
        self.assertGreaterEqual(len(res["conflicts"]), 1)
        conflict = res["conflicts"][0]
        self.assertIn("Biological", conflict["title"])

    def test_jar_test_protocol_generation(self):
        from utils.dosage_calculator import get_jar_test_protocol
        protocol = get_jar_test_protocol(["Liquid Copper Fungicide", "Neem Oil (Cold-Pressed 70%)"])
        self.assertEqual(protocol["jar_volume_ml"], 500.0)
        self.assertEqual(len(protocol["jar_dosages"]), 2)
        self.assertGreaterEqual(len(protocol["protocol_steps"]), 4)
        self.assertIn("pass_criteria", protocol["evaluation"])
        self.assertIn("fail_criteria", protocol["evaluation"])
        self.assertIn("cottage cheese", protocol["evaluation"]["fail_criteria"].lower())


class TestExportUtils(unittest.TestCase):
    """Tests for CSV telemetry export and HTML field checklist generation."""

    def test_export_csv_string(self):
        from utils.export_utils import export_predictions_to_csv_string
        sample_preds = [
            {
                "timestamp": "2026-10-17 12:00:00", 
                "filename": "sample.jpg", 
                "class": "Tomato Early blight", 
                "conf": 0.92, 
                "is_healthy": False, 
                "damage_pct": 14.2, 
                "severity_grade": "Grade 2",
                "farm_zone": "Greenhouse Bay 4",
                "notes": "Targeted with Neem spray"
            }
        ]
        csv_str = export_predictions_to_csv_string(sample_preds)
        self.assertIn("Tomato Early blight", csv_str)
        self.assertIn("14.2", csv_str)
        self.assertIn("Greenhouse Bay 4", csv_str)
        self.assertIn("Targeted with Neem spray", csv_str)
        self.assertTrue(csv_str.startswith("\ufeff"))

    def test_generate_checklist_html(self):
        from utils.export_utils import generate_field_audit_checklist_html
        html = generate_field_audit_checklist_html("Test Vineyard", "Agronomist John", ["Apple", "Grape"])
        self.assertIn("Test Vineyard", html)
        self.assertIn("Agronomist John", html)
        self.assertIn("WALES", html)

    def test_generate_pdf_report_with_decay(self):
        from utils.pdf_generator import generate_pdf_report
        from PIL import Image
        import os
        img = Image.new("RGB", (64, 64), color=(34, 139, 34))
        pred_data = {
            "predicted_class": "Tomato___Early_blight",
            "confidence": 0.95,
            "is_healthy": False,
            "all_probabilities": {"Tomato___Early_blight": 0.95, "Tomato___healthy": 0.05}
        }
        disease_info = {
            "title": "Early Blight",
            "urgency": "High",
            "description": "Fungal disease",
            "symptoms": "Concentric rings",
            "causes": "Warm and wet",
            "immediate_actions": ["Prune lower leaves"],
            "organic_treatment": ["Neem Oil"],
            "chemical_treatment": ["Chlorothalonil"],
            "preventive_measures": ["Crop rotation"],
            "nutrition_and_care": "Avoid high nitrogen"
        }
        decay_data = {
            "product_name": "Chlorothalonil 75 WP",
            "action_type": "Surface Protectant",
            "frac_code": "Group M05",
            "effective_half_life_days": 7.5,
            "rainfast_hours": 2.0,
            "days_to_threshold": 9.2,
            "critical_threshold_pct": 20.0,
            "optimal_ph_range": "5.5 - 6.8",
            "hydrolysis_hazard": "MODERATE",
            "recommended_adjuvant": "Spreader-Sticker",
            "adjuvant_purpose": "Resists rain erosion"
        }
        pdf_path = generate_pdf_report(
            original_img=img,
            heatmap_img=None,
            prediction_data=pred_data,
            disease_info=disease_info,
            decay_data=decay_data
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)
        try:
            os.remove(pdf_path)
        except OSError:
            pass


class TestSensorTelemetry(unittest.TestCase):
    """Tests for IoT sensor parsing, VPD calculation, and spore hazard modeling."""

    def test_calculate_vpd(self):
        from utils.sensor_telemetry import calculate_vpd
        res = calculate_vpd(temp_c=25.0, relative_humidity=60.0)
        self.assertGreater(res["vpd_kpa"], 0.5)
        self.assertLess(res["vpd_kpa"], 2.0)
        self.assertIn("vp_sat", res)
        self.assertIn("vp_act", res)

    def test_evaluate_vpd_hazard(self):
        from utils.sensor_telemetry import evaluate_vpd_disease_hazard
        hazard_low = evaluate_vpd_disease_hazard(0.25)
        self.assertIn("CRITICAL", hazard_low["hazard_level"])
        self.assertIn("Botrytis", hazard_low["target_pathogens"][0])

        hazard_optimal = evaluate_vpd_disease_hazard(1.0)
        self.assertIn("OPTIMAL", hazard_optimal["hazard_level"])

    def test_mock_iot_stream(self):
        from utils.sensor_telemetry import generate_mock_iot_telemetry_stream
        stream = generate_mock_iot_telemetry_stream("Sensor-01", hours=8)
        self.assertEqual(len(stream), 8)
        self.assertIn("vpd_kpa", stream[0])
        self.assertIn("soil_moisture_pct", stream[0])


class TestBatchScanner(unittest.TestCase):
    """Tests for multi-leaf canopy batch processor, CHI index, and CSV export."""

    def test_batch_processing_synthetic(self):
        from utils.batch_scanner import process_leaf_batch
        from PIL import Image

        # Create 1 healthy green leaf
        h_arr = np.zeros((120, 120, 3), dtype=np.uint8)
        h_arr[:, :] = [34, 139, 34]
        img_healthy = Image.fromarray(h_arr)

        # Create 1 diseased leaf with brown necrosis
        d_arr = np.zeros((120, 120, 3), dtype=np.uint8)
        d_arr[:, :] = [34, 139, 34]
        d_arr[30:90, 30:90] = [139, 69, 19]
        img_diseased = Image.fromarray(d_arr)

        res = process_leaf_batch([img_healthy, img_diseased])
        self.assertEqual(res["total_scanned"], 2)
        self.assertEqual(res["valid_scanned"], 2)
        self.assertGreaterEqual(res["canopy_health_index"], 0.0)
        self.assertLessEqual(res["canopy_health_index"], 100.0)
        self.assertEqual(len(res["items"]), 2)
        self.assertIn("triage", res["items"][0])

    def test_batch_csv_summary_formatting(self):
        from utils.batch_scanner import generate_batch_csv_summary
        mock_batch = {
            "items": [
                {
                    "filename": "leaf_01.jpg",
                    "crop": "Tomato",
                    "disease": "Early Blight",
                    "is_healthy": False,
                    "confidence": 0.945,
                    "damage_pct": 18.2,
                    "lesion_count": 4,
                    "stage_label": "Grade 2: Moderate",
                    "triage": "Warning"
                },
                {
                    "filename": "leaf_02.jpg",
                    "crop": "Apple",
                    "disease": "Healthy",
                    "is_healthy": True,
                    "confidence": 0.991,
                    "damage_pct": 0.0,
                    "lesion_count": 0,
                    "stage_label": "Grade 0: Pristine",
                    "triage": "Healthy"
                }
            ]
        }
        csv_out = generate_batch_csv_summary(mock_batch)
        self.assertTrue(csv_out.startswith("\ufeff"))
        self.assertIn("Tomato", csv_out)
        self.assertIn("Early Blight", csv_out)
        self.assertIn("Warning", csv_out)
        self.assertIn("Healthy", csv_out)

    def test_parse_crop_and_disease(self):
        from predict import parse_crop_and_disease
        crop, disease = parse_crop_and_disease("Apple___Apple_scab")
        self.assertEqual(crop, "Apple")
        self.assertEqual(disease, "Apple scab")

        crop, disease = parse_crop_and_disease("Tomato_Early_blight")
        self.assertEqual(crop, "Tomato")
        self.assertEqual(disease, "Early blight")

        crop, disease = parse_crop_and_disease("Tomato_healthy")
        self.assertEqual(crop, "Tomato")
        self.assertEqual(disease, "Healthy")


class TestDewPointAndMultiZoneIoT(unittest.TestCase):
    """Tests for psychrometric dew point calculation and multi-zone farm spatial modeling."""

    def test_dew_point_psychrometrics(self):
        from utils.sensor_telemetry import calculate_dew_point, calculate_vpd
        dp = calculate_dew_point(22.0, 88.0)
        self.assertAlmostEqual(dp, 19.92, delta=0.5)
        
        vpd_res = calculate_vpd(22.0, 88.0)
        self.assertIn("dew_point_c", vpd_res)
        self.assertIn("abs_humidity_gm3", vpd_res)
        self.assertIn("condensation_risk", vpd_res)

    def test_multizone_farm_telemetry(self):
        from utils.sensor_telemetry import simulate_multizone_farm_telemetry
        res = simulate_multizone_farm_telemetry()
        self.assertEqual(res["total_zones"], 4)
        self.assertIn("zones", res)
        self.assertGreaterEqual(res["high_risk_zones"], 0)
        z0 = res["zones"][0]
        self.assertIn("dew_point_c", z0)
        self.assertIn("condensation_margin_c", z0)


class TestSeasonalFRACRotation(unittest.TestCase):
    """Tests for multi-season FRAC/IRAC rotation compliance and resistance risk scoring."""

    def test_consecutive_frac_violation(self):
        from utils.chemical_scanner import evaluate_seasonal_rotation_plan
        plan = ["Heritage 50 WDG", "Heritage 50 WDG"]
        res = evaluate_seasonal_rotation_plan(plan)
        self.assertGreaterEqual(len(res["violations"]), 1)
        self.assertEqual(res["violations"][0]["type"], "CONSECUTIVE_OVEREXPOSURE")
        self.assertGreaterEqual(res["score"], 35)

    def test_compliant_rotation(self):
        from utils.chemical_scanner import evaluate_seasonal_rotation_plan
        plan = ["Heritage 50 WDG", "Bonide Liquid Copper Octanoate", "Luna Tranquility Fungicide"]
        res = evaluate_seasonal_rotation_plan(plan)
        self.assertEqual(len(res["violations"]), 0)
        self.assertEqual(res["status"], "ANTI-RESISTANCE COMPLIANT")
        self.assertLessEqual(res["score"], 30)


class TestSeasonalSprayBudget(unittest.TestCase):
    """Tests for multi-treatment seasonal spray budgeting and economics."""

    def test_seasonal_program_calculation(self):
        from utils.cost_estimator import calculate_seasonal_spray_program_budget
        res = calculate_seasonal_spray_program_budget(area_type="Farm Acreage (Acres)", area_val=2.0)
        self.assertGreaterEqual(res["total_applications"], 4)
        self.assertGreater(res["grand_total_investment"], 0.0)
        self.assertGreater(res["potential_loss_avoided"], 0.0)
        self.assertGreaterEqual(res["roi_multiplier"], 1.0)
        self.assertEqual(len(res["stages"]), 4)


class TestDatabaseLotTaggingAndSearch(unittest.TestCase):
    """Tests for farm zone lot tagging, historical search, and zone health aggregation."""

    def test_lot_tagging_and_search(self):
        from utils.db import add_prediction, search_prediction_history, get_zone_health_summary
        add_prediction(
            filename="lot_leaf_99.jpg",
            predicted_class="Apple___Apple_scab",
            confidence=0.98,
            is_healthy=False,
            damage_pct=21.0,
            severity_grade="Grade 3",
            farm_zone="Zone-C South Orchard",
            notes="Observed on Gala trees"
        )
        matches = search_prediction_history(query="Apple_scab", farm_zone="Zone-C South Orchard")
        self.assertGreaterEqual(len(matches), 1)
        top = matches[0]
        self.assertEqual(top["farm_zone"], "Zone-C South Orchard")
        self.assertIn("Gala", top["notes"])
        
        summary = get_zone_health_summary()
        self.assertGreater(len(summary), 0)
        self.assertTrue(any(z["zone"] == "Zone-C South Orchard" for z in summary))


class TestDatasetValidationUtility(unittest.TestCase):
    """Tests for dataset directory integrity verification and corruption scanner."""

    def test_dataset_validation_missing_dir(self):
        from download_dataset import validate_dataset_integrity
        res = validate_dataset_integrity("non_existent_folder_xyz_123")
        self.assertEqual(res["status"], "NOT_FOUND")
        self.assertEqual(res["classes_found"], 0)

    def test_dataset_validation_synthetic(self):
        import shutil
        from download_dataset import validate_dataset_integrity
        test_dir = "temp_unittest_dataset"
        class_dir = os.path.join(test_dir, "Tomato___Early_blight")
        os.makedirs(class_dir, exist_ok=True)
        try:
            sample_img = Image.new("RGB", (30, 30), color="green")
            sample_img.save(os.path.join(class_dir, "sample_leaf.jpg"))
            
            res = validate_dataset_integrity(test_dir, verify_images=True)
            self.assertEqual(res["status"], "VALID")
            self.assertEqual(res["classes_found"], 1)
            self.assertEqual(res["total_images"], 1)
            self.assertEqual(res["corrupted_count"], 0)
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


class TestCropPathologyVectorAndIncubation(unittest.TestCase):
    """Tests for disease incubation periods, vector insect carriers, and secondary spread dynamics."""

    def test_epidemiology_profile_lookup(self):
        from utils.disease_expert import get_disease_epidemiology_profile
        prof = get_disease_epidemiology_profile("Apple___Apple_scab")
        self.assertEqual(prof["crop"], "Apple")
        self.assertEqual(prof["incubation_days_min"], 8)
        self.assertEqual(prof["incubation_days_max"], 14)
        self.assertIn("Wind-borne ascospores", prof["primary_vectors"])
        self.assertGreaterEqual(len(prof["quarantine_measures"]), 1)

    def test_epidemiology_healthy_fallback(self):
        from utils.disease_expert import get_disease_epidemiology_profile
        prof = get_disease_epidemiology_profile("Tomato_healthy")
        self.assertEqual(prof["incubation_days_min"], 0)
        self.assertEqual(prof["secondary_spread_rate"], "None (No pathogen present)")

    def test_find_diseases_by_vector(self):
        from utils.disease_expert import find_diseases_by_vector
        whitefly_diseases = find_diseases_by_vector("whitefly")
        self.assertGreaterEqual(len(whitefly_diseases), 1)
        self.assertTrue(any("Yellow_Leaf_Curl" in d["disease_key"] for d in whitefly_diseases))

        rain_diseases = find_diseases_by_vector("rain splash")
        self.assertGreaterEqual(len(rain_diseases), 3)


class TestSprayDriftAndNozzlePhysics(unittest.TestCase):
    """Tests for ASABE S572.1 droplet classification, boom pressure VMD, and drift hazard modeling."""

    def test_nozzle_types_available(self):
        from utils.dosage_calculator import get_available_nozzle_types
        types = get_available_nozzle_types()
        self.assertGreaterEqual(len(types), 4)
        self.assertIn("Air Induction Flat Fan (AI / AIC)", types)

    def test_spray_drift_standard_vs_air_induction(self):
        from utils.dosage_calculator import calculate_spray_drift_and_nozzle_specs
        xr_spec = calculate_spray_drift_and_nozzle_specs(
            nozzle_type="Extended Range Flat Fan (XR)",
            boom_pressure_psi=50.0,
            wind_speed_mph=6.0
        )
        ai_spec = calculate_spray_drift_and_nozzle_specs(
            nozzle_type="Air Induction Flat Fan (AI / AIC)",
            boom_pressure_psi=50.0,
            wind_speed_mph=6.0
        )
        self.assertGreater(ai_spec["vmd_microns"], xr_spec["vmd_microns"])
        self.assertLess(ai_spec["drift_hazard_score"], xr_spec["drift_hazard_score"])
        self.assertTrue(any(cat in xr_spec["asabe_category"].lower() for cat in ["fine", "medium"]))

    def test_high_wind_drift_hazard(self):
        from utils.dosage_calculator import calculate_spray_drift_and_nozzle_specs
        spec = calculate_spray_drift_and_nozzle_specs(
            nozzle_type="Hollow Cone (TX ConeJet)",
            boom_pressure_psi=60.0,
            wind_speed_mph=14.0,
            boom_height_inches=32.0
        )
        self.assertGreaterEqual(spec["drift_hazard_score"], 60.0)
        self.assertIn("PROHIBITED", spec["wind_status"])
        self.assertGreater(spec["setback_buffer_ft"], 25.0)
        self.assertGreaterEqual(len(spec["warnings"]), 1)


class TestChemicalCatalogFiltering(unittest.TestCase):
    """Tests for multi-criteria chemical catalog queries by FRAC, REI, PHI, and pollinator safety."""

    def test_filter_by_frac(self):
        from utils.chemical_scanner import filter_chemicals
        g11 = filter_chemicals(frac_group="Group 11")
        self.assertGreaterEqual(len(g11), 1)
        for prod in g11:
            self.assertIn("11", prod["frac_group"])

    def test_filter_by_rei_and_pollinator(self):
        from utils.chemical_scanner import filter_chemicals
        results = filter_chemicals(max_rei_hours=12, pollinator_safe_only=True)
        self.assertGreaterEqual(len(results), 2)
        for prod in results:
            self.assertLessEqual(prod["rei_hours"], 12)

    def test_filter_by_phi(self):
        from utils.chemical_scanner import filter_chemicals
        harvest_ready = filter_chemicals(max_phi_days=1)
        self.assertGreaterEqual(len(harvest_ready), 3)
        for prod in harvest_ready:
            self.assertLessEqual(prod["phi_days"], 1)


class TestSpectralVegetativeAndPhysicalLeafCV(unittest.TestCase):
    """Tests for multispectral chlorophyll indices, physical leaf dimensions, and recovery tracking."""

    def test_spectral_indices_healthy_vs_necrotic(self):
        # 1. Healthy green leaf image
        green_arr = np.zeros((150, 150, 3), dtype=np.uint8)
        green_arr[:, :] = [34, 180, 34]  # High green channel
        green_img = Image.fromarray(green_arr)
        green_metrics = calculate_canopy_spectral_indices(green_img)

        self.assertIn("mean_gli", green_metrics)
        self.assertIn("spad_chlorophyll_proxy", green_metrics)
        self.assertIn("spectral_map", green_metrics)
        self.assertGreater(green_metrics["mean_gli"], 0.05)
        self.assertGreater(green_metrics["spad_chlorophyll_proxy"], 25.0)
        self.assertIn(green_metrics["chlorophyll_status"], ["Optimal / High Nitrogen Vigor", "Moderate / Normal Vegetative"])

        # 2. Severely chlorotic/brown tissue
        brown_arr = np.zeros((150, 150, 3), dtype=np.uint8)
        brown_arr[:, :] = [180, 100, 40]  # High red, low green
        brown_img = Image.fromarray(brown_arr)
        brown_metrics = calculate_canopy_spectral_indices(brown_img)

        self.assertLess(brown_metrics["mean_gli"], green_metrics["mean_gli"])
        self.assertLess(brown_metrics["spad_chlorophyll_proxy"], green_metrics["spad_chlorophyll_proxy"])

    def test_estimate_physical_leaf_dimensions(self):
        # Synthetic green leaf with a central lesion
        img_arr = np.zeros((200, 200, 3), dtype=np.uint8)
        img_arr[30:170, 40:160] = [34, 139, 34]  # Leaf region
        img_arr[80:120, 80:120] = [139, 69, 19]  # Lesion region
        pil_img = Image.fromarray(img_arr)

        dims = estimate_physical_leaf_dimensions(pil_img, assumed_dpi=96)
        self.assertIn("leaf_area_cm2", dims)
        self.assertIn("lesion_area_cm2", dims)
        self.assertIn("lesion_area_mm2", dims)
        self.assertIn("leaf_length_cm", dims)
        self.assertIn("aspect_ratio", dims)
        self.assertIn("circularity_form_factor", dims)

        self.assertGreater(dims["leaf_area_cm2"], 0.0)
        self.assertGreater(dims["lesion_area_cm2"], 0.0)
        self.assertGreater(dims["lesion_area_mm2"], 0.0)
        self.assertGreater(dims["leaf_length_cm"], 0.0)

    def test_recovery_tracker_spectral_regreening(self):
        # Day 0 (severe damage, brown spots)
        before_arr = np.zeros((150, 150, 3), dtype=np.uint8)
        before_arr[:, :] = [120, 100, 50]  # Stressed leaf
        before_img = Image.fromarray(before_arr)

        # Day 14 (recovered, lush green)
        after_arr = np.zeros((150, 150, 3), dtype=np.uint8)
        after_arr[:, :] = [30, 180, 40]  # Healthy green
        after_img = Image.fromarray(after_arr)

        recovery = evaluate_treatment_recovery(before_img, after_img)
        self.assertIn("delta_spad", recovery)
        self.assertIn("spad_before", recovery)
        self.assertIn("spad_after", recovery)
        self.assertIn("spectral_before", recovery)
        self.assertIn("spectral_after", recovery)
        self.assertGreater(recovery["delta_spad"], 0.0)
        self.assertGreater(recovery["spad_after"], recovery["spad_before"])


class TestPathogenPhenologyAndThermalGDD(unittest.TestCase):
    """Tests for Growing Degree Day (GDD) accumulation and pathogen biofix phenology models."""

    def test_daily_gdd_calculation(self):
        from utils.gdd_phenology import calculate_daily_gdd
        # Standard average: (22 + 10) / 2 - 10 = 6.0
        gdd = calculate_daily_gdd(t_max=22.0, t_min=10.0, base_temp=10.0, cap_temp=30.0)
        self.assertEqual(gdd, 6.0)

        # Cold day below base: (8 + 4) / 2 - 10 = -4 -> 0.0
        cold_gdd = calculate_daily_gdd(t_max=8.0, t_min=4.0, base_temp=10.0)
        self.assertEqual(cold_gdd, 0.0)

        # Hot day capped at 30: (30 + 18) / 2 - 10 = 14.0
        capped_gdd = calculate_daily_gdd(t_max=38.0, t_min=18.0, base_temp=10.0, cap_temp=30.0)
        self.assertEqual(capped_gdd, 14.0)

    def test_cumulative_gdd_series(self):
        from utils.gdd_phenology import calculate_cumulative_gdd_series
        records = [
            {"date": "2026-04-01", "t_max": 20.0, "t_min": 10.0},
            {"date": "2026-04-02", "t_max": 22.0, "t_min": 12.0},
            {"date": "2026-04-03", "t_max": 18.0, "t_min": 8.0}
        ]
        res = calculate_cumulative_gdd_series(records, base_temp=10.0)
        self.assertEqual(res["days_accumulated"], 3)
        self.assertGreater(res["total_cumulative_gdd"], 10.0)
        self.assertEqual(len(res["series"]), 3)
        self.assertEqual(res["series"][-1]["cumulative_gdd"], res["total_cumulative_gdd"])

    def test_pathogen_phenology_apple_scab(self):
        from utils.gdd_phenology import evaluate_pathogen_phenology
        # Early stage
        early = evaluate_pathogen_phenology("Apple Scab", cumulative_gdd=25.0)
        self.assertIn("Dormant", early["stage"])
        self.assertEqual(early["base_temp_c"], 0.0)

        # Peak ascospore discharge window
        peak = evaluate_pathogen_phenology("Apple Scab", cumulative_gdd=195.0)
        self.assertIn("Accelerated Ascospore Discharge", peak["stage"])
        self.assertIn("Critical Primary Alert", peak["risk_level"])
        self.assertGreater(peak["stage_progress_pct"], 0.0)
        self.assertGreater(peak["gdd_to_next_stage"], 0.0)

    def test_pathogen_phenology_tomato_early_blight_fast(self):
        from utils.gdd_phenology import evaluate_pathogen_phenology
        # 300 GDD FAST threshold
        fast = evaluate_pathogen_phenology("Tomato Early Blight", cumulative_gdd=320.0)
        self.assertIn("FAST Critical Action Threshold", fast["stage"])
        self.assertIn("High Treatment Trigger", fast["risk_level"])
        self.assertIn("Mancozeb", fast["action_advisory"])

    def test_weather_risk_with_gdd_integration(self):
        from utils.weather_risk import calculate_disease_outbreak_risk
        res = calculate_disease_outbreak_risk(
            temp=19.0,
            humidity=92.0,
            rainfall=6.0,
            leaf_wetness=10.0,
            wind_speed=12.0,
            crop_filter="Apple",
            cumulative_gdd=195.0
        )
        self.assertGreaterEqual(len(res["diseases"]), 1)
        apple_disease = res["diseases"][0]
        self.assertIn("phenology", apple_disease)
        self.assertEqual(apple_disease["phenology"]["cumulative_gdd"], 195.0)
        self.assertIn("Peak Inoculum", apple_disease["phenology"]["stage"])

    def test_upcoming_milestones_projection(self):
        from utils.gdd_phenology import estimate_upcoming_milestones
        milestones = estimate_upcoming_milestones("Apple Scab", current_gdd=80.0, projected_daily_gdd=10.0)
        self.assertGreater(len(milestones), 0)
        first_next = milestones[0]
        self.assertGreater(first_next["days_remaining"], 0)
        self.assertIn("target_gdd", first_next)
        self.assertIn("projected_date", first_next)


class TestFungicideKineticsAndWaterQuality(unittest.TestCase):
    """Tests for fungicide residue first-order dissipation, rainfastness, and spray water conditioning."""

    def test_decay_products_registered(self):
        from utils.fungicide_decay import get_all_decay_products, get_product_kinetics
        prods = get_all_decay_products()
        self.assertGreaterEqual(len(prods), 9)
        for p in prods:
            k = get_product_kinetics(p)
            self.assertIsNotNone(k)
            self.assertGreater(k["base_half_life_days"], 0.0)
            self.assertGreater(k["rainfast_hours"], 0.0)
            self.assertIn("action_type", k)
            self.assertIn("frac_code", k)

    def test_first_order_residue_decay_basic(self):
        from utils.fungicide_decay import calculate_foliar_chemical_decay
        res = calculate_foliar_chemical_decay(
            product_name="Azoxystrobin 22.9% (Quadris / Heritage)",
            initial_concentration_pct=100.0,
            simulation_days=10,
            avg_temperature_c=20.0,
            avg_uv_index=5.0,
            canopy_growth_rate_pct=0.0
        )
        self.assertEqual(len(res["trajectory"]), 11)  # Day 0 to 10
        traj = res["trajectory"]
        # Day 0 is 100%
        self.assertEqual(traj[0]["concentration_pct"], 100.0)
        # Monotonic decrease
        for i in range(1, len(traj)):
            self.assertLess(traj[i]["concentration_pct"], traj[i - 1]["concentration_pct"])
        self.assertAlmostEqual(res["effective_half_life_days"], res["base_half_life_days"], places=1)

    def test_temperature_and_uv_acceleration(self):
        from utils.fungicide_decay import calculate_foliar_chemical_decay
        cool_res = calculate_foliar_chemical_decay(
            product_name="Mancozeb 75 WP (Dithane M-45 Multi-Site)",
            avg_temperature_c=15.0,
            avg_uv_index=3.0
        )
        hot_res = calculate_foliar_chemical_decay(
            product_name="Mancozeb 75 WP (Dithane M-45 Multi-Site)",
            avg_temperature_c=30.0,
            avg_uv_index=9.0
        )
        self.assertLess(hot_res["effective_half_life_days"], cool_res["effective_half_life_days"])
        self.assertLess(hot_res["trajectory"][-1]["concentration_pct"], cool_res["trajectory"][-1]["concentration_pct"])

    def test_rainfall_washoff_impact(self):
        from utils.fungicide_decay import calculate_foliar_chemical_decay
        # Mancozeb (contact protectant, high wash-off coefficient) vs Azoxystrobin (systemic penetrant)
        rain_event = [0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0]
        manc_decay = calculate_foliar_chemical_decay(
            product_name="Mancozeb 75 WP (Dithane M-45 Multi-Site)",
            daily_rainfall_mm=rain_event,
            simulation_days=6
        )
        azoxy_decay = calculate_foliar_chemical_decay(
            product_name="Azoxystrobin 22.9% (Quadris / Heritage)",
            daily_rainfall_mm=rain_event,
            simulation_days=6
        )
        # Mancozeb washoff loss on day 3 should be greater than Azoxystrobin
        manc_wash = manc_decay["trajectory"][3]["washoff_loss_pct"]
        azoxy_wash = azoxy_decay["trajectory"][3]["washoff_loss_pct"]
        self.assertGreater(manc_wash, azoxy_wash)

    def test_canopy_vegetative_growth_dilution(self):
        from utils.fungicide_decay import calculate_foliar_chemical_decay
        static_canopy = calculate_foliar_chemical_decay(
            product_name="Chlorothalonil 75 WP (Daconil Broad-Spectrum)",
            canopy_growth_rate_pct=0.0,
            simulation_days=7
        )
        growing_canopy = calculate_foliar_chemical_decay(
            product_name="Chlorothalonil 75 WP (Daconil Broad-Spectrum)",
            canopy_growth_rate_pct=5.0,
            simulation_days=7
        )
        self.assertLess(growing_canopy["trajectory"][-1]["concentration_pct"], static_canopy["trajectory"][-1]["concentration_pct"])

    def test_reapplication_prediction_schedule(self):
        from utils.fungicide_decay import predict_reapplication_schedule
        import datetime
        app_d = datetime.date(2026, 6, 1)
        sched = predict_reapplication_schedule(
            product_name="Chlorothalonil 75 WP (Daconil Broad-Spectrum)",
            application_date=app_d,
            current_day_offset=2,
            forecasted_rain_day=4
        )
        self.assertIn("recommended_respray_date", sched)
        self.assertGreater(sched["days_remaining_protection"], 0.0)
        self.assertIn(sched["urgency_level"], ["ROUTINE", "ADVANCED_BEFORE_RAIN", "RESPRAY_WINDOW_ACTIVE"])

    def test_rainfastness_curing_states(self):
        from utils.fungicide_decay import evaluate_rainfastness
        # Azoxystrobin requires 2.0h
        cured = evaluate_rainfastness("Azoxystrobin 22.9% (Quadris / Heritage)", hours_since_spray=3.0)
        self.assertEqual(cured["state"], "SECURE_RAINFAST")
        self.assertEqual(cured["curing_pct"], 100.0)

        partial = evaluate_rainfastness("Azoxystrobin 22.9% (Quadris / Heritage)", hours_since_spray=1.2)
        self.assertEqual(partial["state"], "PARTIAL_CURE")

        uncured = evaluate_rainfastness("Azoxystrobin 22.9% (Quadris / Heritage)", hours_since_spray=0.4)
        self.assertEqual(uncured["state"], "UNCURED_VULNERABLE")
        self.assertGreater(uncured["estimated_washoff_loss_pct"], 50.0)

    def test_spray_water_alkaline_hydrolysis_and_acidifier(self):
        from utils.fungicide_decay import optimize_spray_water_and_adjuvant
        # Captan at alkaline pH 8.2 has extreme hazard
        captan_opt = optimize_spray_water_and_adjuvant(
            product_name="Captan 50 WP",
            water_ph=8.2,
            water_hardness_ppm=280.0,
            tank_volume_gallons=100.0
        )
        self.assertEqual(captan_opt["hydrolysis_hazard"], "EXTREME")
        self.assertIn("ALKALINE HYDROLYSIS DANGER", captan_opt["hydrolysis_alert"])
        self.assertEqual(captan_opt["buffer_status"], "ACIDIFIER_REQUIRED")
        self.assertGreater(captan_opt["citric_acid_oz_total"], 0.0)
        self.assertIn("Ammonium Sulfate", captan_opt["ams_recommendation"])

        # Optimal neutral water
        opt_water = optimize_spray_water_and_adjuvant(
            product_name="Chlorothalonil 75 WP (Daconil Broad-Spectrum)",
            water_ph=6.2,
            water_hardness_ppm=80.0
        )
        self.assertEqual(opt_water["buffer_status"], "OPTIMAL_PH")
        self.assertEqual(opt_water["citric_acid_oz_total"], 0.0)


class TestBatchScannerMultispectralAndPhysicalAggregation(unittest.TestCase):
    """Tests for batch scanner spectral and physical metric aggregation."""

    def test_batch_scanner_spectral_and_physical_metrics(self):
        from utils.batch_scanner import process_leaf_batch, generate_batch_csv_summary
        from PIL import Image

        img1 = Image.new("RGB", (100, 100), color=(34, 139, 34))
        img2 = Image.new("RGB", (100, 100), color=(139, 69, 19))

        batch_res = process_leaf_batch([img1, img2])
        self.assertIn("mean_spad_chlorophyll", batch_res)
        self.assertIn("mean_gli", batch_res)
        self.assertIn("total_canopy_area_cm2", batch_res)
        self.assertIn("total_lesion_area_cm2", batch_res)
        self.assertIn("canopy_photosynthetic_efficiency", batch_res)
        self.assertGreaterEqual(batch_res["mean_spad_chlorophyll"], 0.0)

        # Verify CSV summary includes new biometric headers
        csv_out = generate_batch_csv_summary(batch_res)
        self.assertIn("SPAD Chlorophyll", csv_out)
        self.assertIn("Leaf Area (cm2)", csv_out)
        self.assertIn("Lesion Area (cm2)", csv_out)


class TestSpatialMappingAndVariableRatePrescription(unittest.TestCase):
    """Tests for spatial disease IDW interpolation, VRA zone classification, and GIS exports."""

    def test_demo_farm_blocks_exist(self):
        from utils.spatial_mapping import DEMO_FARM_BLOCKS
        self.assertGreaterEqual(len(DEMO_FARM_BLOCKS), 3)
        for name, blk in DEMO_FARM_BLOCKS.items():
            self.assertIn("crop", blk)
            self.assertIn("center_lat", blk)
            self.assertIn("center_lon", blk)
            self.assertIn("scout_points", blk)
            self.assertGreater(len(blk["scout_points"]), 0)

    def test_meters_to_lat_lon_offsets(self):
        from utils.spatial_mapping import meters_to_lat_lon_offsets
        dlat_zero, dlon_zero = meters_to_lat_lon_offsets(0.0, 0.0, 38.5)
        self.assertEqual(dlat_zero, 0.0)
        self.assertEqual(dlon_zero, 0.0)

        dlat_pos, dlon_pos = meters_to_lat_lon_offsets(100.0, 100.0, 38.5)
        self.assertGreater(dlat_pos, 0.0)
        self.assertGreater(dlon_pos, 0.0)

    def test_idw_interpolation_accuracy(self):
        from utils.spatial_mapping import interpolate_idw
        pts = [
            {"offset_x": 0.0, "offset_y": 0.0, "damage_pct": 50.0},
            {"offset_x": 100.0, "offset_y": 100.0, "damage_pct": 10.0}
        ]
        # At exact position of pt 0
        val_0 = interpolate_idw(0.0, 0.0, pts)
        self.assertEqual(val_0, 50.0)

        # Midpoint
        val_mid = interpolate_idw(50.0, 50.0, pts)
        self.assertAlmostEqual(val_mid, 30.0, delta=1.0)

    def test_prescription_grid_generation(self):
        from utils.spatial_mapping import generate_field_prescription_grid
        presc = generate_field_prescription_grid("North Valley Vineyard (Block 7)", grid_resolution_meters=30.0)
        self.assertIn("grid_cells", presc)
        self.assertIn("economics", presc)
        self.assertIn("zone_distribution", presc)
        self.assertIn("field_boundary", presc)
        self.assertGreater(len(presc["grid_cells"]), 0)

        # Verify cell schema
        cell0 = presc["grid_cells"][0]
        self.assertIn("center_lat", cell0)
        self.assertIn("center_lon", cell0)
        self.assertIn("damage_pct", cell0)
        self.assertIn("zone_name", cell0)
        self.assertIn("rate_multiplier", cell0)
        self.assertIn("rate_l_ha", cell0)
        self.assertIn("polygon_coords", cell0)
        self.assertEqual(len(cell0["polygon_coords"]), 5)

    def test_chemical_savings_calculation(self):
        from utils.spatial_mapping import generate_field_prescription_grid
        presc = generate_field_prescription_grid("Crestview Tomato Basin (Plot 4-East)", grid_resolution_meters=25.0)
        econ = presc["economics"]
        self.assertLess(econ["vra_precision_liters"], econ["uniform_broadcast_liters"])
        self.assertGreater(econ["liters_saved"], 0.0)
        self.assertGreater(econ["chemical_reduction_pct"], 15.0)
        self.assertGreater(econ["dollars_saved"], 0.0)
        self.assertGreater(econ["savings_per_acre"], 0.0)

    def test_geojson_export_rfc7946(self):
        from utils.spatial_mapping import generate_field_prescription_grid, export_field_prescription_geojson
        import json
        presc = generate_field_prescription_grid("West Ridge Apple Orchard (Quadrant B)")
        geojson_str = export_field_prescription_geojson(presc)
        doc = json.loads(geojson_str)
        self.assertEqual(doc["type"], "FeatureCollection")
        self.assertGreater(len(doc["features"]), 5)
        # Check boundary feature
        boundary = doc["features"][0]
        self.assertEqual(boundary["geometry"]["type"], "Polygon")
        self.assertEqual(boundary["properties"]["feature_type"], "Field_Boundary")

    def test_kml_export_format(self):
        from utils.spatial_mapping import generate_field_prescription_grid, export_field_prescription_kml
        presc = generate_field_prescription_grid("North Valley Vineyard (Block 7)")
        kml_str = export_field_prescription_kml(presc)
        self.assertTrue(kml_str.startswith('<?xml version="1.0" encoding="UTF-8"?>'))
        self.assertIn("<kml", kml_str)
        self.assertIn("<Polygon>", kml_str)
        self.assertIn("<LinearRing>", kml_str)
        self.assertIn("<Placemark>", kml_str)
        self.assertIn("polyRed", kml_str)

    def test_rate_controller_csv(self):
        from utils.spatial_mapping import generate_field_prescription_grid, export_rate_controller_csv
        presc = generate_field_prescription_grid("Crestview Tomato Basin (Plot 4-East)")
        csv_str = export_rate_controller_csv(presc)
        self.assertTrue(csv_str.startswith("\ufeff"))
        self.assertIn("Grid_ID,Row,Col,Latitude,Longitude", csv_str)
        self.assertIn("CELL-00-00", csv_str)
        lines = [line for line in csv_str.strip().split("\r\n") if line]
        self.assertEqual(len(lines), 1 + len(presc["grid_cells"]))

    def test_pdf_report_with_spatial_data(self):
        from utils.pdf_generator import generate_pdf_report
        from utils.spatial_mapping import generate_field_prescription_grid
        from PIL import Image
        import os

        img = Image.new("RGB", (64, 64), color=(34, 139, 34))
        pred_data = {
            "predicted_class": "Grape___Black_rot",
            "confidence": 0.94,
            "is_healthy": False,
            "all_probabilities": {"Grape___Black_rot": 0.94, "Grape___healthy": 0.06}
        }
        disease_info = {
            "title": "Black Rot",
            "urgency": "High",
            "description": "Fungal rot",
            "symptoms": "Black shriveled fruit mummies",
            "causes": "Warm and wet",
            "immediate_actions": ["Strip mummies"],
            "organic_treatment": ["Copper"],
            "chemical_treatment": ["Myclobutanil"],
            "preventive_measures": ["Canopy airflow"],
            "nutrition_and_care": "Balanced compost"
        }
        presc = generate_field_prescription_grid("North Valley Vineyard (Block 7)")
        pdf_path = generate_pdf_report(
            original_img=img,
            heatmap_img=None,
            prediction_data=pred_data,
            disease_info=disease_info,
            spatial_data=presc
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    def test_calculate_field_bounding_box(self):
        from utils.spatial_mapping import calculate_field_bounding_box
        poly = [
            [-121.748, 38.541],
            [-121.742, 38.541],
            [-121.742, 38.545],
            [-121.748, 38.545],
            [-121.748, 38.541]
        ]
        bbox = calculate_field_bounding_box(poly)
        self.assertEqual(bbox["min_lat"], 38.541)
        self.assertEqual(bbox["max_lat"], 38.545)
        self.assertEqual(bbox["min_lon"], -121.748)
        self.assertEqual(bbox["max_lon"], -121.742)
        self.assertGreater(bbox["delta_lat"], 0.0)
        self.assertGreater(bbox["delta_lon"], 0.0)

    def test_calculate_polygon_acreage(self):
        from utils.spatial_mapping import calculate_polygon_acreage
        poly = [
            [-121.748, 38.541],
            [-121.742, 38.541],
            [-121.742, 38.545],
            [-121.748, 38.545],
            [-121.748, 38.541]
        ]
        acres = calculate_polygon_acreage(poly)
        self.assertGreater(acres, 10.0)
        self.assertLess(acres, 100.0)


class TestSoilbornePathologyAndDampingOff(unittest.TestCase):
    """Automated tests for Soilborne Pathogen & Seedling Damping-Off Engine."""

    def test_soilborne_pathogens_database_integrity(self):
        from utils.soilborne_pathology import SOILBORNE_PATHOGENS_DATABASE
        expected_keys = [
            "Pythium_ultimum",
            "Pythium_aphanidermatum",
            "Rhizoctonia_solani",
            "Fusarium_oxysporum",
            "Phytophthora_capsici",
            "Verticillium_dahliae",
            "Macrophomina_phaseolina"
        ]
        for key in expected_keys:
            self.assertIn(key, SOILBORNE_PATHOGENS_DATABASE)
            entry = SOILBORNE_PATHOGENS_DATABASE[key]
            self.assertIn("common_name", entry)
            self.assertIn("organism_class", entry)
            self.assertLess(entry["temp_min_c"], entry["temp_opt_c"])
            self.assertLess(entry["temp_opt_c"], entry["temp_max_c"])
            self.assertGreater(len(entry["primary_symptoms"]), 0)
            self.assertIn("frac_chemical_group", entry)

    def test_cardinal_temperature_thermal_suitability(self):
        from utils.soilborne_pathology import calculate_thermal_suitability
        # Pythium ultimum: min 8, opt 17.5, max 26
        suit_opt = calculate_thermal_suitability(17.5, 8.0, 17.5, 26.0)
        self.assertAlmostEqual(suit_opt, 1.0, places=2)

        suit_min = calculate_thermal_suitability(7.0, 8.0, 17.5, 26.0)
        self.assertEqual(suit_min, 0.0)

        suit_max = calculate_thermal_suitability(27.0, 8.0, 17.5, 26.0)
        self.assertEqual(suit_max, 0.0)

        suit_mid = calculate_thermal_suitability(14.0, 8.0, 17.5, 26.0)
        self.assertGreater(suit_mid, 0.5)
        self.assertLess(suit_mid, 1.0)

    def test_moisture_suitability_and_oomycete_motility(self):
        from utils.soilborne_pathology import calculate_moisture_suitability
        # Oomycete at 90% FC (saturated) -> high zoospore motility
        m_suit, z_mot = calculate_moisture_suitability(90.0, 75.0, 100.0, is_oomycete=True)
        self.assertGreaterEqual(m_suit, 0.9)
        self.assertGreater(z_mot, 0.7)

        # Fungal pathogen (non-oomycete) -> no zoospores
        m_suit_f, z_mot_f = calculate_moisture_suitability(60.0, 45.0, 75.0, is_oomycete=False)
        self.assertAlmostEqual(m_suit_f, 1.0, places=1)
        self.assertEqual(z_mot_f, 0.0)

    def test_soilborne_risk_calculation(self):
        from utils.soilborne_pathology import calculate_soilborne_pathogen_risk
        # Cold, wet soil favoring Pythium ultimum
        res = calculate_soilborne_pathogen_risk(
            soil_temp_c=17.0,
            soil_moisture_fc_pct=92.0,
            soil_texture="Clay",
            compaction_level="Moderate",
            crop="Tomato",
            seed_depth_cm=3.0
        )
        self.assertIn("overall_risk_score", res)
        self.assertIn("overall_tier", res)
        self.assertIn("primary_pathogen_key", res)
        self.assertIn("pathogen_breakdown", res)
        self.assertEqual(res["primary_pathogen_key"], "Pythium_ultimum")
        self.assertGreater(res["overall_risk_score"], 60.0)
        self.assertIn("CRITICAL", res["overall_tier"].upper() + res["primary_pathogen"]["risk_tier"].upper())

    def test_soil_compaction_and_emergence_penalty(self):
        from utils.soilborne_pathology import calculate_soilborne_pathogen_risk
        res_loose = calculate_soilborne_pathogen_risk(
            soil_temp_c=22.0,
            soil_moisture_fc_pct=60.0,
            compaction_level="Low / Loose",
            seed_depth_cm=2.5
        )
        res_compact = calculate_soilborne_pathogen_risk(
            soil_temp_c=22.0,
            soil_moisture_fc_pct=60.0,
            compaction_level="Severe / Plow-Pan",
            seed_depth_cm=5.0
        )
        self.assertGreater(res_compact["projected_emergence_days"], res_loose["projected_emergence_days"])
        self.assertGreater(res_compact["exposure_vulnerability_index"], res_loose["exposure_vulnerability_index"])

    def test_biological_suppressiveness_scoring(self):
        from utils.soilborne_pathology import evaluate_soil_biological_suppressiveness
        # High suppressive profile
        res_supp = evaluate_soil_biological_suppressiveness(
            soil_organic_matter_pct=5.5,
            soil_ph=6.8,
            beneficials_inoculated=True,
            crop_rotation_years=4,
            tillage_practice="No-Till / Direct Drill"
        )
        self.assertGreaterEqual(res_supp["suppressiveness_score"], 80.0)
        self.assertEqual(res_supp["suppressiveness_level"], "HIGHLY DISEASE-SUPPRESSIVE")

        # Conducive depleted profile
        res_cond = evaluate_soil_biological_suppressiveness(
            soil_organic_matter_pct=1.2,
            soil_ph=5.0,
            beneficials_inoculated=False,
            crop_rotation_years=1,
            tillage_practice="Conventional Deep Inversion Moldboard"
        )
        self.assertLess(res_cond["suppressiveness_score"], 40.0)
        self.assertEqual(res_cond["suppressiveness_level"], "HIGHLY DISEASE-CONDUCIVE")

    def test_seed_and_drench_regimens(self):
        from utils.soilborne_pathology import prescribe_seed_and_drench_regimen
        # Oomycete prescription
        rx_pyth = prescribe_seed_and_drench_regimen("Pythium_ultimum", crop="Tomato")
        self.assertIn("Mefenoxam", rx_pyth["chemical_seed_treatment"]["primary_active"])
        self.assertIn("FRAC 4", rx_pyth["chemical_seed_treatment"]["frac_code"])
        self.assertIn("Trichoderma", rx_pyth["biological_seed_treatment"]["active_biologicals"])
        self.assertIn("Brown Mustard", rx_pyth["biofumigation_protocol"]["cover_crop_species"])

        # Basidiomycete prescription
        rx_rhiz = prescribe_seed_and_drench_regimen("Rhizoctonia_solani", crop="Potato")
        self.assertIn("Sedaxane", rx_rhiz["chemical_seed_treatment"]["primary_active"])
        self.assertIn("FRAC 7", rx_rhiz["chemical_seed_treatment"]["frac_code"])

    def test_radar_and_json_export(self):
        from utils.soilborne_pathology import (
            calculate_soilborne_pathogen_risk,
            evaluate_soil_biological_suppressiveness,
            prescribe_seed_and_drench_regimen,
            generate_suppressiveness_radar_data,
            export_soilborne_assessment_json
        )
        import json

        risk_res = calculate_soilborne_pathogen_risk(20.0, 70.0, "Silt Loam", "Moderate", "Tomato")
        supp_res = evaluate_soil_biological_suppressiveness(3.5, 6.5, True, 3, "Reduced / Minimum Till")
        presc_res = prescribe_seed_and_drench_regimen("Pythium_ultimum", "Tomato")

        radar = generate_suppressiveness_radar_data(supp_res)
        self.assertEqual(len(radar["categories"]), 5)
        self.assertEqual(len(radar["values"]), 5)
        for val in radar["values"]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 100.0)

        json_str = export_soilborne_assessment_json(risk_res, supp_res, presc_res)
        parsed = json.loads(json_str)
        self.assertIn("audit_metadata", parsed)
        self.assertIn("soil_telemetry", parsed)
        self.assertIn("epidemiological_risk", parsed)
        self.assertIn("biological_suppressiveness", parsed)
        self.assertIn("agronomic_prescription", parsed)

    def test_pdf_report_with_soil_data(self):
        from utils.pdf_generator import generate_pdf_report
        from utils.soilborne_pathology import (
            calculate_soilborne_pathogen_risk,
            evaluate_soil_biological_suppressiveness,
            prescribe_seed_and_drench_regimen
        )
        img = Image.new("RGB", (256, 256), color=(40, 160, 40))
        pred_data = {
            "predicted_class": "Tomato___Early_blight",
            "confidence": 0.94,
            "is_healthy": False,
            "all_probabilities": {"Tomato___Early_blight": 0.94, "Tomato___healthy": 0.06}
        }
        disease_info = {
            "disease_name": "Tomato Early Blight",
            "description": "Fungal infection caused by Alternaria solani.",
            "symptoms": "Concentric rings and brown target lesions on lower foliage.",
            "causes": "Warm temperatures and high humidity.",
            "immediate_actions": "Prune lower infected leaves.",
            "organic_treatment": "Copper octanoate.",
            "chemical_treatment": ["Chlorothalonil", "Mancozeb"],
            "preventive_measures": ["Drip irrigation", "Crop rotation"],
            "nutrition_and_care": "Avoid excess nitrogen."
        }
        risk_res = calculate_soilborne_pathogen_risk(19.0, 80.0, "Clay Loam", "Moderate", "Tomato")
        supp_res = evaluate_soil_biological_suppressiveness(4.0, 6.6, True, 3, "Reduced / Minimum Till")
        presc_res = prescribe_seed_and_drench_regimen("Pythium_ultimum", "Tomato")
        soil_payload = {
            "overall_risk_score": risk_res["overall_risk_score"],
            "overall_tier": risk_res["overall_tier"],
            "primary_pathogen": risk_res["primary_pathogen"],
            "projected_emergence_days": risk_res["projected_emergence_days"],
            "exposure_vulnerability_index": risk_res["exposure_vulnerability_index"],
            "suppressiveness_score": supp_res["suppressiveness_score"],
            "suppressiveness_level": supp_res["suppressiveness_level"],
            "prescription": presc_res
        }
        pdf_path = generate_pdf_report(
            original_img=img,
            heatmap_img=None,
            prediction_data=pred_data,
            disease_info=disease_info,
            soil_data=soil_payload
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)
        try:
            os.remove(pdf_path)
        except OSError:
            pass


class TestEvapotranspirationAndCropWaterStress(unittest.TestCase):
    """Automated tests for FAO-56 Penman-Monteith, CWSI & Disease-Irrigation Scheduler."""

    def test_crop_coefficients_database_integrity(self):
        from utils.evapotranspiration_cwsi import CROP_COEFFICIENT_DATABASE
        expected_crops = ["Tomato", "Potato", "Pepper", "Apple", "Corn", "Grape", "Strawberry"]
        for crop in expected_crops:
            self.assertIn(crop, CROP_COEFFICIENT_DATABASE)
            cdata = CROP_COEFFICIENT_DATABASE[crop]
            self.assertIn("kc_ini", cdata)
            self.assertIn("kc_mid", cdata)
            self.assertIn("kc_end", cdata)
            self.assertGreater(cdata["kc_mid"], cdata["kc_ini"])
            self.assertGreater(cdata["max_rooting_depth_m"], 0.2)
            self.assertGreater(cdata["depletion_fraction_p"], 0.1)
            self.assertLess(cdata["depletion_fraction_p"], 0.8)
            self.assertIn("idso_nwsb_intercept_a", cdata)
            self.assertIn("idso_nwsb_slope_b", cdata)
            self.assertIn("idso_upper_limit_deg_c", cdata)

    def test_fao56_penman_monteith_baseline(self):
        from utils.evapotranspiration_cwsi import calculate_fao56_penman_monteith
        res = calculate_fao56_penman_monteith(
            temp_mean_c=20.0,
            relative_humidity_pct=50.0,
            solar_radiation_mj_m2_day=20.0,
            wind_speed_2m_m_s=2.0,
            elevation_m=100.0
        )
        self.assertIn("et0_mm_day", res)
        self.assertGreaterEqual(res["et0_mm_day"], 3.5)
        self.assertLessEqual(res["et0_mm_day"], 7.5)
        self.assertGreater(res["vapor_pressure_deficit_kpa"], 0.5)
        self.assertGreater(res["slope_vapor_pressure_delta"], 0.1)
        self.assertGreater(res["psychrometric_constant_gamma"], 0.05)
        self.assertGreater(res["radiation_component_mm"], 0.0)
        self.assertGreater(res["aerodynamic_component_mm"], 0.0)

    def test_fao56_temperature_and_radiation_sensitivity(self):
        from utils.evapotranspiration_cwsi import calculate_fao56_penman_monteith
        res_cool = calculate_fao56_penman_monteith(temp_mean_c=12.0, relative_humidity_pct=70.0, solar_radiation_mj_m2_day=12.0)
        res_hot = calculate_fao56_penman_monteith(temp_mean_c=32.0, relative_humidity_pct=35.0, solar_radiation_mj_m2_day=26.0)
        self.assertGreater(res_hot["et0_mm_day"], res_cool["et0_mm_day"])
        self.assertGreater(res_hot["vapor_pressure_deficit_kpa"], res_cool["vapor_pressure_deficit_kpa"])

    def test_crop_water_stress_index_calculation(self):
        from utils.evapotranspiration_cwsi import calculate_crop_water_stress_index
        # Well-watered transpiring canopy: Tc cooler than Ta
        res_cool_canopy = calculate_crop_water_stress_index(
            canopy_temp_c=21.0,
            air_temp_c=25.0,
            vpd_kpa=1.5,
            crop="Tomato"
        )
        self.assertLessEqual(res_cool_canopy["cwsi"], 0.25)
        self.assertIn("NO STRESS", res_cool_canopy["status"])

        # Stressed heated canopy: Tc significantly warmer than Ta
        res_hot_canopy = calculate_crop_water_stress_index(
            canopy_temp_c=31.0,
            air_temp_c=26.0,
            vpd_kpa=1.5,
            crop="Tomato"
        )
        self.assertGreaterEqual(res_hot_canopy["cwsi"], 0.70)
        self.assertTrue(any(k in (res_hot_canopy["status"] + " " + res_hot_canopy["advisory"]).upper() for k in ["STRESS", "DESICCATION", "HAZARD"]))

    def test_cwsi_bounds_and_clamping(self):
        from utils.evapotranspiration_cwsi import calculate_crop_water_stress_index
        res_low = calculate_crop_water_stress_index(canopy_temp_c=10.0, air_temp_c=30.0, vpd_kpa=3.0, crop="Potato")
        res_high = calculate_crop_water_stress_index(canopy_temp_c=45.0, air_temp_c=25.0, vpd_kpa=0.5, crop="Potato")
        self.assertGreaterEqual(res_low["cwsi"], 0.0)
        self.assertLessEqual(res_low["cwsi"], 1.0)
        self.assertGreaterEqual(res_high["cwsi"], 0.0)
        self.assertLessEqual(res_high["cwsi"], 1.0)

    def test_irrigation_prescription_and_efficiency(self):
        from utils.evapotranspiration_cwsi import optimize_irrigation_and_disease_schedule
        sched_drip = optimize_irrigation_and_disease_schedule(
            crop="Tomato",
            growth_stage="Mid-Season / Fruit Bulking",
            et0_mm_day=5.0,
            effective_rainfall_mm=0.0,
            delivery_system="Subsurface Drip",
            target_area_ha=2.0
        )
        # Tomato Kc mid is 1.15 -> ETc = 5.75 mm
        self.assertAlmostEqual(sched_drip["actual_crop_etc_mm"], 5.75, places=1)
        self.assertGreater(sched_drip["gross_irrigation_mm"], sched_drip["net_irrigation_mm"])
        self.assertAlmostEqual(sched_drip["volume_m3"], sched_drip["gross_irrigation_mm"] * 10.0 * 2.0, places=1)
        self.assertGreater(sched_drip["volume_gallons"], 1000.0)

    def test_disease_timing_and_overhead_hazard(self):
        from utils.evapotranspiration_cwsi import optimize_irrigation_and_disease_schedule
        # Overnight irrigation -> dangerous leaf wetness
        sched_dusk = optimize_irrigation_and_disease_schedule(
            crop="Tomato",
            delivery_system="Center Pivot / Overhead Sprinkler",
            planned_irrigation_time="08:30 PM (Dusk)"
        )
        self.assertGreater(sched_dusk["combined_disease_hazard_multiplier"], 3.0)
        self.assertIn("TRAP", sched_dusk["timing_risk_status"])

        # Dawn irrigation -> optimal drying
        sched_dawn = optimize_irrigation_and_disease_schedule(
            crop="Tomato",
            delivery_system="Subsurface Drip",
            planned_irrigation_time="05:00 AM (Dawn)"
        )
        self.assertEqual(sched_dawn["combined_disease_hazard_multiplier"], 1.0)
        self.assertIn("OPTIMAL", sched_dawn["timing_risk_status"])

    def test_depletion_trajectory_and_raw_threshold(self):
        from utils.evapotranspiration_cwsi import generate_irrigation_depletion_curve
        traj = generate_irrigation_depletion_curve(
            daily_etc_mm=6.0,
            rooting_depth_m=1.0,
            soil_water_capacity_mm_per_m=120.0,
            depletion_fraction_p=0.40,
            horizon_days=7
        )
        self.assertEqual(len(traj), 7)
        self.assertGreater(traj[-1]["cumulative_depletion_mm"], traj[0]["cumulative_depletion_mm"])
        self.assertLess(traj[-1]["moisture_remaining_pct"], traj[0]["moisture_remaining_pct"])
        # Day 7: 42 mm depletion, RAW = 48 mm -> should be nearing stress
        self.assertIn("raw_threshold_mm", traj[0])
        self.assertEqual(traj[0]["raw_threshold_mm"], 48.0)

    def test_pdf_report_with_irrigation_data(self):
        from utils.pdf_generator import generate_pdf_report
        from utils.evapotranspiration_cwsi import (
            calculate_fao56_penman_monteith,
            calculate_crop_water_stress_index,
            optimize_irrigation_and_disease_schedule
        )
        img = Image.new("RGB", (256, 256), color=(40, 160, 40))
        pred_data = {
            "predicted_class": "Tomato___Early_blight",
            "confidence": 0.95,
            "is_healthy": False,
            "all_probabilities": {"Tomato___Early_blight": 0.95, "Tomato___healthy": 0.05}
        }
        disease_info = {
            "disease_name": "Tomato Early Blight",
            "description": "Fungal infection caused by Alternaria solani.",
            "symptoms": "Concentric rings and brown target lesions on lower foliage.",
            "causes": "Warm temperatures and high humidity.",
            "immediate_actions": "Prune lower infected leaves.",
            "organic_treatment": "Copper octanoate.",
            "chemical_treatment": ["Chlorothalonil"],
            "preventive_measures": ["Drip irrigation", "Crop rotation"],
            "nutrition_and_care": "Avoid excess nitrogen."
        }
        et0_res = calculate_fao56_penman_monteith(24.0, 55.0)
        cwsi_res = calculate_crop_water_stress_index(23.5, 24.0, et0_res["vapor_pressure_deficit_kpa"], "Tomato")
        sched_res = optimize_irrigation_and_disease_schedule(crop="Tomato", et0_mm_day=et0_res["et0_mm_day"])
        irrig_payload = {
            "et0": et0_res,
            "cwsi": cwsi_res,
            "schedule": sched_res
        }
        pdf_path = generate_pdf_report(
            original_img=img,
            heatmap_img=None,
            prediction_data=pred_data,
            disease_info=disease_info,
            irrigation_data=irrig_payload
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)
        try:
            os.remove(pdf_path)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()

