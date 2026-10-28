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

from utils.leaf_cv import analyze_leaf_damage
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
            {"timestamp": "2026-10-17 12:00:00", "filename": "sample.jpg", "class": "Tomato Early blight", "conf": 0.92, "is_healthy": False, "damage_pct": 14.2, "severity_grade": "Grade 2"}
        ]
        csv_str = export_predictions_to_csv_string(sample_preds)
        self.assertIn("Tomato Early blight", csv_str)
        self.assertIn("14.2", csv_str)
        self.assertTrue(csv_str.startswith("\ufeff"))

    def test_generate_checklist_html(self):
        from utils.export_utils import generate_field_audit_checklist_html
        html = generate_field_audit_checklist_html("Test Vineyard", "Agronomist John", ["Apple", "Grape"])
        self.assertIn("Test Vineyard", html)
        self.assertIn("Agronomist John", html)
        self.assertIn("WALES", html)


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


if __name__ == "__main__":
    unittest.main()

