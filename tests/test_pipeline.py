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


if __name__ == "__main__":
    unittest.main()
