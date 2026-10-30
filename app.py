import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import google.generativeai as genai
import time
import os
import cv2
import numpy as np
from predict import PlantDiseasePredictor
from utils.db import (
    add_prediction, get_recent_predictions, get_summary_stats, 
    get_disease_distribution, get_daily_trends, get_all_predictions_df,
    clear_prediction_history, search_prediction_history, get_zone_health_summary,
    export_history_json
)
from utils.pdf_generator import generate_pdf_report
from utils.disease_expert import (
    get_disease_expert_info, get_all_disease_guides,
    get_disease_epidemiology_profile, find_diseases_by_vector
)
from utils.weather_risk import (
    fetch_live_weather, calculate_disease_outbreak_risk, WEATHER_PRESETS,
    get_regional_monitoring_stations
)
from utils.dosage_calculator import (
    compute_spray_dosage, get_all_products, get_product_details, 
    TREATMENT_DATABASE, check_tank_mix_compatibility, get_jar_test_protocol,
    calculate_spray_drift_and_nozzle_specs, get_available_nozzle_types, AVAILABLE_NOZZLE_TYPES
)
from utils.leaf_cv import analyze_leaf_damage, calculate_canopy_spectral_indices, estimate_physical_leaf_dimensions
from utils.agronomy_brain import query_agronomy_brain, SUGGESTED_QUESTIONS
from utils.crop_calendar import get_crop_calendar, PHENOLOGY_CALENDAR
from utils.cost_estimator import (
    estimate_treatment_economics, DEFAULT_PRODUCT_PRICING, BENCHMARK_CROP_VALUES,
    calculate_seasonal_spray_program_budget
)
from utils.recovery_tracker import evaluate_treatment_recovery
from utils.chemical_scanner import (
    lookup_chemical_by_epa_or_name, get_all_chemical_catalog, get_frac_rotation_guidance,
    evaluate_seasonal_rotation_plan, filter_chemicals
)
from utils.nutrient_diagnostics import (
    diagnose_nutrient_deficiency, get_deficiency_vs_disease_comparison, 
    NUTRIENT_DEFICIENCIES, calculate_soil_ph_bioavailability
)
from utils.export_utils import export_predictions_to_csv_string, generate_field_audit_checklist_html
from utils.sensor_telemetry import (
    calculate_vpd, evaluate_vpd_disease_hazard, generate_mock_iot_telemetry_stream,
    simulate_multizone_farm_telemetry, calculate_dew_point, calculate_absolute_humidity
)
from utils.batch_scanner import process_leaf_batch, generate_batch_csv_summary
from utils.gdd_phenology import (
    calculate_daily_gdd, calculate_cumulative_gdd_series,
    evaluate_pathogen_phenology, estimate_upcoming_milestones,
    get_supported_phenology_models, PATHOGEN_PHENOLOGY_MODELS
)
import pydeck as pdk
import pandas as pd
import datetime

# Set page configuration for a modern look
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Premium Dark Theme Background */
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(16, 185, 129, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(59, 130, 246, 0.08), transparent 25%);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header hero text */
    .hero-title {
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(to right, #34d399, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
        letter-spacing: -2px;
        line-height: 1.1;
        text-shadow: 0 0 40px rgba(16, 185, 129, 0.4);
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 10px;
        margin-bottom: 40px;
        letter-spacing: 0.5px;
    }
    
    /* Text color overrides for Dark Mode */
    h1, h2, h3, h4, h5, h6, span, div {
        color: #f1f5f9;
    }
    .stMarkdown p, .stMarkdown div {
        color: #cbd5e1 !important;
    }
    
    /* Sleek Buttons with Neon Glow */
    .stButton>button {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981 !important;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: rgba(16, 185, 129, 0.2);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4), inset 0 0 10px rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.8);
        color: #fff !important;
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 2px dashed rgba(16, 185, 129, 0.3) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.3s ease !important;
        padding: 40px !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: #10b981 !important;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15) !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.4);
        padding: 8px;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 10px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #34d399 !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
    }
    
    /* Dashboard Custom Metrics Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 30px 24px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.5), transparent);
    }
    .metric-card:hover {
        transform: translateY(-8px);
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.1);
    }
    .metric-title {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34d399 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
    }
    
    /* Modern Glassmorphism Cards */
    .disease-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        padding: 40px;
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin: 30px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out forwards;
        position: relative;
        overflow: hidden;
    }
    .disease-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.03), transparent);
        transform: skewX(-25deg);
        animation: shine 8s infinite;
    }
    .disease-card:hover {
        transform: translateY(-5px);
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(16, 185, 129, 0.05);
    }
    
    /* Typography */
    .healthy-text {
        color: #34d399 !important;
        font-weight: 800;
        font-size: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(52, 211, 153, 0.4);
    }
    .disease-text {
        color: #f87171 !important;
        font-weight: 800;
        font-size: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(248, 113, 113, 0.4);
        animation: pulseRed 2s infinite;
    }
    
    /* Streamlit specific overrides */
    .stRadio > label {
        color: #94a3b8;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.6);
        border-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes shine {
        0% { left: -100%; }
        20% { left: 200%; }
        100% { left: 200%; }
    }
    @keyframes pulseRed {
        0% { text-shadow: 0 0 10px rgba(248, 113, 113, 0.2); }
        50% { text-shadow: 0 0 25px rgba(248, 113, 113, 0.6); }
        100% { text-shadow: 0 0 10px rgba(248, 113, 113, 0.2); }
    }
    </style>
""", unsafe_allow_html=True)

# --- Disease Information & Treatment Expert System ---
def get_disease_info(class_name):
    return get_disease_expert_info(class_name)

class LocalImageFile:
    def __init__(self, path, name=None):
        self.path = path
        self.name = name if name else os.path.basename(path)

# Shared Cached Model Predictor
@st.cache_resource
def load_predictor():
    predictor = PlantDiseasePredictor()
    try:
        predictor.load_model()
        return predictor, None
    except Exception as e:
        return None, str(e)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2928/2928883.png", width=100) # Placeholder icon
        st.title("PlantGuard AI")
        st.info(
            "Upload images of plant leaves, and our AI model will detect "
            "if the plants are healthy or suffering from a disease."
        )
        st.markdown("---")
        
        page = st.radio(
            "Navigation", 
            [
                "📸 Disease Scanner", 
                "📦 Multi-Leaf Canopy Batch Scanner",
                "🔬 Recovery & Healing Tracker",
                "🧪 Abiotic Nutrient Diagnostic Matrix",
                "📡 IoT Microclimate & VPD Telemetry",
                "🌦️ Disease Risk Forecaster", 
                "🌡️ Pathogen Phenology & Degree-Day Engine",
                "⚖️ Spray & Dosage Calculator",
                "📅 Seasonal Spray Calendar",
                "📊 Analytics Dashboard", 
                "📖 Care & Treatment Handbook", 
                "💬 AI Plant Assistant"
            ]
        )
        st.markdown("---")
        
        # Display Prediction History from Database
        st.markdown("### 🕒 Recent History")
        history = get_recent_predictions(limit=5)
        if len(history) == 0:
            st.write("No predictions yet.")
        else:
            for item in history:
                status_icon = "✅" if item['is_healthy'] else "🦠"
                st.write(f"{status_icon} **{item['class']}** ({item['conf']*100:.1f}%)")
                dmg_text = f" • {item['damage_pct']:.1f}% damage" if item.get('damage_pct', 0.0) > 0 else ""
                st.caption(f"📅 {item['timestamp']}{dmg_text}")
        
        st.markdown("---")
        st.warning("⚠️ **Disclaimer:** This is an educational tool and should not replace professional agricultural advice.")

    # Multi-Leaf Canopy Batch Diagnostic Scanner View
    if page == "📦 Multi-Leaf Canopy Batch Scanner":
        st.title("📦 Multi-Leaf Canopy Batch Diagnostic Scanner")
        st.markdown("### High-throughput foliage batch processor with aggregate canopy health metrics, automated triage, and field inspection export.")
        st.caption("Quantifies overall Canopy Health Index (CHI), identifies localized disease hotspots across orchard/vineyard rows, and triages critical foliar infections.")

        predictor, error_msg = load_predictor()
        if error_msg:
            st.warning(f"⚠️ Predictor model note: {error_msg}. Running with fallback biological rules.")

        col_bctrl1, col_bctrl2 = st.columns([1.8, 2.2])
        with col_bctrl1:
            load_batch_demo = st.button("🧪 Load Field Canopy Demo (6 Leaf Scans Across Rows)", use_container_width=True)
        with col_bctrl2:
            st.info("💡 Rapidly survey entire tree canopies or crop beds by uploading multiple foliage photos at once.")

        if "batch_files_queue" not in st.session_state:
            st.session_state["batch_files_queue"] = []
        if "batch_results" not in st.session_state:
            st.session_state["batch_results"] = None

        sample_apple = "assets/samples/apple_scab.jpg"
        sample_healthy = "assets/samples/healthy_leaf.jpg"
        sample_tomato = "assets/samples/tomato_early_blight.jpg"

        if load_batch_demo:
            demo_queue = []
            if os.path.exists(sample_apple):
                demo_queue.append(LocalImageFile(sample_apple, "Canopy_Row1_SpecimenA_AppleScab.jpg"))
            if os.path.exists(sample_healthy):
                demo_queue.append(LocalImageFile(sample_healthy, "Canopy_Row1_SpecimenB_HealthyLeaf.jpg"))
            if os.path.exists(sample_tomato):
                demo_queue.append(LocalImageFile(sample_tomato, "Canopy_Row2_SpecimenC_EarlyBlight.jpg"))
            if os.path.exists(sample_apple):
                demo_queue.append(LocalImageFile(sample_apple, "Canopy_Row2_SpecimenD_AppleScab.jpg"))
            if os.path.exists(sample_healthy):
                demo_queue.append(LocalImageFile(sample_healthy, "Canopy_Row3_SpecimenE_HealthyLeaf.jpg"))
            if os.path.exists(sample_tomato):
                demo_queue.append(LocalImageFile(sample_tomato, "Canopy_Row3_SpecimenF_EarlyBlight.jpg"))
            st.session_state["batch_files_queue"] = demo_queue
            st.session_state["batch_results"] = None
            st.toast("Loaded 6 canopy field specimens!")

        up_batch = st.file_uploader(
            "📁 Select Multiple Leaf Images (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        if up_batch:
            st.session_state["batch_files_queue"] = up_batch
            st.session_state["batch_results"] = None

        active_batch = st.session_state["batch_files_queue"]
        if active_batch:
            st.markdown(f"**Queued Foliage Scans:** `{len(active_batch)} images ready for canopy analysis.`")
            if st.button("🚀 Execute Canopy Batch Diagnosis", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                batch_res = process_leaf_batch(active_batch, predictor=predictor, progress_callback=update_progress)
                st.session_state["batch_results"] = batch_res
                status_text.empty()
                progress_bar.empty()
                st.toast("Batch diagnosis completed successfully! 🎉")

        if st.session_state.get("batch_results"):
            b_res = st.session_state["batch_results"]
            chi = b_res["canopy_health_index"]
            avg_dmg = b_res["average_damage_pct"]
            healthy_pct = b_res["healthy_percentage"]
            crit_count = b_res["high_severity_count"]

            if chi >= 75.0:
                chi_color = "#10b981"
                chi_badge = "Vigorous / Healthy Canopy"
            elif chi >= 45.0:
                chi_color = "#f59e0b"
                chi_badge = "Moderate Foliar Stress"
            else:
                chi_color = "#ef4444"
                chi_badge = "Severe Canopy Outbreak Threat"

            # CHI Gauge Banner
            st.markdown(f"""
            <div class='disease-card' style='border-color: {chi_color}; margin: 24px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h2 style='margin:0; color: #f8fafc;'>Canopy Health Index (CHI)</h2>
                        <p style='color: #cbd5e1; margin: 6px 0 0 0;'>Composite foliar vigor scoring based on {b_res['total_scanned']} evaluated canopy leaf specimens.</p>
                    </div>
                    <div style='text-align: right;'>
                        <span style='font-size: 3rem; font-weight: 800; color: {chi_color};'>{chi:.1f} / 100</span>
                        <div style='font-weight: 700; color: {chi_color}; text-transform: uppercase;'>{chi_badge}</div>
                    </div>
                </div>
                <div style='width: 100%; background-color: #334155; border-radius: 999px; height: 12px; margin: 18px 0 6px 0;'>
                    <div style='background-color: {chi_color}; width: {min(100.0, max(4.0, chi))}%; height: 12px; border-radius: 999px; box-shadow: 0 0 15px {chi_color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4 Summary Metric Cards
            col_bk1, col_bk2, col_bk3, col_bk4 = st.columns(4)
            with col_bk1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Total Foliage Scanned</div>
                    <div class='metric-value' style='font-size: 2.2rem;'>{b_res['total_scanned']} Leaves</div>
                    <div style='color: #94a3b8; font-size: 0.85rem;'>{b_res['valid_scanned']} Processed</div>
                </div>
                """, unsafe_allow_html=True)
            with col_bk2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Canopy Health Ratio</div>
                    <div class='metric-value' style='font-size: 2.2rem; color: #34d399;'>{healthy_pct:.1f}%</div>
                    <div style='color: #94a3b8; font-size: 0.85rem;'>{b_res['healthy_count']} Healthy / {b_res['diseased_count']} Diseased</div>
                </div>
                """, unsafe_allow_html=True)
            with col_bk3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Average Damaged Area</div>
                    <div class='metric-value' style='font-size: 2.2rem; color: #f59e0b;'>{avg_dmg:.1f}%</div>
                    <div style='color: #94a3b8; font-size: 0.85rem;'>Foliar Lesion Surface</div>
                </div>
                """, unsafe_allow_html=True)
            with col_bk4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Triage Outbreak Alerts</div>
                    <div class='metric-value' style='font-size: 2.2rem; color: #ef4444;'>{crit_count}</div>
                    <div style='color: #94a3b8; font-size: 0.85rem;'>Critical / Warning Leaves</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Triage Filter Tabs
            st.markdown("#### 🔍 Triage & Foliar Inspection Gallery")
            all_items = b_res.get("items", [])
            crit_items = [it for it in all_items if it.get("triage") == "Critical"]
            warn_items = [it for it in all_items if it.get("triage") == "Warning"]
            hlth_items = [it for it in all_items if it.get("triage") == "Healthy"]

            tab_all, tab_crit, tab_warn, tab_hlth = st.tabs([
                f"📋 All Specimens ({len(all_items)})",
                f"🚨 Critical ({len(crit_items)})",
                f"⚠️ Warning ({len(warn_items)})",
                f"🌱 Healthy ({len(hlth_items)})"
            ])

            def render_specimen_list(items_list):
                if not items_list:
                    st.info("No leaf scans in this triage category.")
                    return
                for s_idx, item in enumerate(items_list):
                    t_col = item.get("triage_color", "#10b981")
                    with st.expander(f"🍂 **{item.get('filename')}** — {item.get('crop')} {item.get('disease')} | Triage: {item.get('triage')} ({item.get('confidence', 0.0)*100:.1f}%)", expanded=(s_idx == 0)):
                        col_sp1, col_sp2, col_sp3 = st.columns([1, 1, 1.2])
                        with col_sp1:
                            if item.get("image"):
                                st.image(item["image"], caption="Original Specimen", use_column_width=True)
                        with col_sp2:
                            if item.get("mask_overlay"):
                                st.image(item["mask_overlay"], caption=f"Lesion Mask ({item.get('damage_pct', 0.0):.1f}% Damaged)", use_column_width=True)
                        with col_sp3:
                            st.markdown(f"<span style='background:{t_col}22; color:{t_col}; padding:4px 10px; border-radius:8px; font-weight:700; border:1px solid {t_col}66;'>TRIAGE: {item.get('triage', '').upper()}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Primary Diagnosis:** `{item.get('class', '').replace('_', ' ')}`")
                            st.markdown(f"**Confidence:** `{item.get('confidence', 0.0)*100:.1f}%`")
                            st.markdown(f"**Foliar Damage:** `{item.get('damage_pct', 0.0):.1f}%` ({item.get('stage_label', 'N/A')})")
                            st.markdown(f"**Lesion Clusters:** `{item.get('lesion_count', 0)}`")
                            if item.get("top_3"):
                                st.markdown("##### Differential Diagnoses:")
                                for t3 in item["top_3"]:
                                    st.write(f"- {t3.get('disease', t3.get('class'))} ({t3.get('confidence', 0.0)*100:.1f}%)")

            with tab_all:
                render_specimen_list(all_items)
            with tab_crit:
                render_specimen_list(crit_items)
            with tab_warn:
                render_specimen_list(warn_items)
            with tab_hlth:
                render_specimen_list(hlth_items)

            # Export Telemetry CSV
            st.markdown("<br>", unsafe_allow_html=True)
            batch_csv_data = generate_batch_csv_summary(b_res)
            st.download_button(
                label="📥 Download Canopy Field Inspection Telemetry (RFC-4180 CSV)",
                data=batch_csv_data.encode("utf-8"),
                file_name=f'plantguard_canopy_batch_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        else:
            st.info("👆 Upload leaf images or click the 1-Click Field Canopy Demo above to begin batch diagnosis.")
        return

    # Recovery & Healing Tracker View
    if page == "🔬 Recovery & Healing Tracker":
        st.title("🔬 Post-Treatment Recovery & Healing Tracker")
        st.markdown("### Dual-scan computer vision evaluation comparing baseline vs post-treatment foliar canopy.")
        st.caption("Quantifies lesion surface reduction, tissue callousing, and curative therapeutic efficacy.")
        
        col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])
        with col_ctrl1:
            load_demo = st.button("🧪 Load Clinical Demo (Tomato Early Blight → Treated)", use_container_width=True)
        with col_ctrl2:
            st.info("💡 Compare pre-treatment diseased foliar tissue with post-treatment foliar recovery to quantitatively confirm pathogen eradication.")

        if "recovery_before" not in st.session_state:
            st.session_state["recovery_before"] = None
        if "recovery_after" not in st.session_state:
            st.session_state["recovery_after"] = None

        if load_demo:
            st.session_state["recovery_before"] = "assets/samples/tomato_early_blight.jpg"
            st.session_state["recovery_after"] = "assets/samples/healthy_leaf.jpg"

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            st.markdown("##### 1️⃣ Baseline Scan (Pre-Treatment)")
            up_before = st.file_uploader("Upload Pre-Treatment Leaf", type=["jpg", "jpeg", "png"], key="uploader_before")
            if up_before:
                st.session_state["recovery_before"] = up_before

            if st.session_state["recovery_before"]:
                img_before_show = Image.open(st.session_state["recovery_before"])
                st.image(img_before_show, caption="Specimen A: Baseline Infection", use_column_width=True)

        with col_up2:
            st.markdown("##### 2️⃣ Follow-up Scan (Post-Treatment)")
            up_after = st.file_uploader("Upload Post-Treatment Leaf", type=["jpg", "jpeg", "png"], key="uploader_after")
            if up_after:
                st.session_state["recovery_after"] = up_after

            if st.session_state["recovery_after"]:
                img_after_show = Image.open(st.session_state["recovery_after"])
                st.image(img_after_show, caption="Specimen B: Follow-up Canopy", use_column_width=True)

        st.markdown("---")
        if st.session_state["recovery_before"] and st.session_state["recovery_after"]:
            if st.button("🔬 Execute Quantitative Recovery Analysis", use_container_width=True):
                with st.spinner("Executing ExG color space segmentation, lesion count delta, and healing evaluation..."):
                    img_b = Image.open(st.session_state["recovery_before"])
                    img_a = Image.open(st.session_state["recovery_after"])
                    
                    rec = evaluate_treatment_recovery(img_b, img_a)
                    
                    # Status Banner
                    st.markdown(f"""
                    <div class='disease-card' style='border-color: {rec["status_color"]}; margin: 20px 0;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <h3 style='margin:0; color: {rec["status_color"]};'>{rec["status"]}</h3>
                                <p style='color: #cbd5e1; margin: 8px 0 0 0;'>{rec["guidance"]}</p>
                            </div>
                            <div style='text-align: right; min-width: 140px;'>
                                <span style='font-size: 2.8rem; font-weight: 800; color: {rec["status_color"]};'>+{rec["recovery_pct"]}%</span>
                                <div style='font-weight: 600; color: #94a3b8;'>EFFICACY RATING</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 5 KPI Cards
                    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
                    with col_k1:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-title'>Pre-Treatment Damage</div>
                            <div class='metric-value' style='font-size: 2.2rem; color: #f87171;'>{rec["dmg_before"]}%</div>
                            <div style='color: #94a3b8; font-size: 0.85rem;'>{rec["stage_before"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k2:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-title'>Post-Treatment Damage</div>
                            <div class='metric-value' style='font-size: 2.2rem; color: #34d399;'>{rec["dmg_after"]}%</div>
                            <div style='color: #94a3b8; font-size: 0.85rem;'>{rec["stage_after"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k3:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-title'>Lesion Area Reduction</div>
                            <div class='metric-value' style='font-size: 2.2rem;'>{rec["delta_damage"]}%</div>
                            <div style='color: #94a3b8; font-size: 0.85rem;'>Net Foliar Reclamation</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k4:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-title'>Active Lesion Count</div>
                            <div class='metric-value' style='font-size: 2.2rem;'>{rec["lesions_before"]} → {rec["lesions_after"]}</div>
                            <div style='color: #94a3b8; font-size: 0.85rem;'>Individual Spots</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k5:
                        spad_color = "#10b981" if rec.get("delta_spad", 0) >= 0 else "#f87171"
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-title'>Chlorophyll Re-Greening</div>
                            <div class='metric-value' style='font-size: 2.2rem; color:{spad_color};'>{rec.get('delta_spad', 0.0):+0.1f}</div>
                            <div style='color: #94a3b8; font-size: 0.85rem;'>SPAD {rec.get('spad_before', 0)} → {rec.get('spad_after', 0)}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 🔬 Computer Vision Dual Overlay Comparison")
                    st.image(rec["composite_image"], caption="Left: Baseline Lesion Detection (Red Mask) | Right: Post-Treatment Foliar Surface", use_column_width=True)
                    
                    if "spectral_before" in rec and "spectral_after" in rec:
                        with st.expander("🌿 Spectral Chlorophyll Dual-Map Comparison (Viridis)", expanded=False):
                            col_sp1, col_sp2 = st.columns(2)
                            with col_sp1:
                                st.image(rec["spectral_before"]["spectral_map"], caption=f"Baseline Spectral Map: SPAD {rec['spad_before']} ({rec['spectral_before']['chlorophyll_status']})", use_column_width=True)
                            with col_sp2:
                                st.image(rec["spectral_after"]["spectral_map"], caption=f"Post-Treatment Spectral Map: SPAD {rec['spad_after']} ({rec['spectral_after']['chlorophyll_status']})", use_column_width=True)
        else:
            st.info("👆 Please upload or load both Baseline and Follow-up leaf images above to perform healing evaluation.")
        return

    # Abiotic Nutrient Diagnostic Matrix View
    if page == "🧪 Abiotic Nutrient Diagnostic Matrix":
        st.title("🧪 Abiotic Plant Nutrient Deficiency & Cross-Diagnostic Matrix")
        st.markdown("### Distinguish physiological nutrient disorders from biotic foliar pathogens and model root-zone soil pH bioavailability.")
        st.caption("Up to 40% of foliar discolorations and stunted growth are abiotic nutritional imbalances rather than infectious fungi or bacteria.")
        
        tab_symp, tab_soil = st.tabs([
            "🌿 Foliar Symptom & Pathology Differential", 
            "🧪 Soil pH Bioavailability & Amendment Calculator"
        ])
        
        with tab_symp:
            col_nd1, col_nd2 = st.columns([1, 1], gap="large")
            with col_nd1:
                st.markdown("#### 1️⃣ Symptom Observation on Specimen")
                leaf_loc = st.selectbox(
                    "🌿 Canopy Leaf Position:",
                    [
                        "Upper young leaves / Shoots (Immobile elements: Fe, Ca, S, Zn, Mn, B)",
                        "Lower mature leaves (Mobile elements: N, P, K, Mg)",
                        "Fruit / Blossom End (Calcium & Boron reproductive mobility)"
                    ]
                )
                
                symptom_choice = st.selectbox(
                    "🔍 Primary Foliar Pattern:",
                    [
                        "Interveinal chlorosis (veins emerald green, leaf blade yellow)",
                        "Marginal scorch / crispy brown leaf edges ('tip burn')",
                        "Uniform pale yellowing starting from leaf tip",
                        "Reddish-purple or bronze pigmentation on undersides",
                        "Distorted, hooked tips with blossom end rot on fruit",
                        "Little leaf syndrome with rosetted stunted internodes",
                        "Checkered interveinal chlorosis with dark necrotic speckles",
                        "Terminal shoot dieback with thick brittle leaves"
                    ]
                )
                
                st.markdown("---")
                st.markdown("#### 🔍 Cross-Disease Differential Check")
                suspected_disease = st.selectbox(
                    "Compare against suspected infectious pathogen:",
                    ["Tomato Early Blight", "Late Blight", "Powdery Mildew", "Fusarium Wilt", "Septoria Leaf Spot", "Tomato Mosaic Virus"]
                )
                comp_diff = get_deficiency_vs_disease_comparison(suspected_disease)
                st.info(f"**Commonly Mimics:** `{comp_diff['mimicking_nutrient']}`\n\n**Differential Rule:** {comp_diff['differential_rule']}")

            with col_nd2:
                st.markdown("#### 2️⃣ Ranked Nutritional Diagnosis")
                diagnoses = diagnose_nutrient_deficiency(leaf_loc, symptom_choice)
                
                if diagnoses:
                    for idx, d in enumerate(diagnoses[:4]):
                        with st.expander(f"🔬 Rank #{idx+1}: **{d['nutrient']}** — Match: {d['confidence_score']}%", expanded=(idx==0)):
                            st.markdown(f"**Mobility:** `{d['mobility']}`")
                            st.markdown(f"**Visual Symptom:** {d['symptoms']}")
                            st.markdown(f"**Confused with Diseases:** {', '.join(d['confused_with'])}")
                            st.warning(f"💡 **How to Tell Apart:** {d['differential_rule']}")
                            st.markdown(f"**🌱 Soil Amendment:** {d['remedy']}")
                            st.success(f"⚡ **Emergency Foliar Spray:** {d['emergency_spray']}")
                else:
                    st.info("Select symptoms on the left to view ranked nutritional diagnoses.")

        with tab_soil:
            st.markdown("#### 🧪 Root-Zone Soil pH Bioavailability Simulator (Truog Curves)")
            st.markdown("Soil pH governs root-zone chemical solubility and ion exchange capacity. Model nutrient lockouts and calculate agricultural limestone or elemental sulfur soil amendment requirements.")
            
            col_sp1, col_sp2 = st.columns([1, 1.2], gap="large")
            with col_sp1:
                st.markdown("##### 🎛️ Field Soil Parameters")
                soil_ph_val = st.slider("Soil pH (H₂O saturated paste test):", min_value=4.0, max_value=9.5, value=6.5, step=0.1)
                
                soil_tex_choice = st.selectbox(
                    "Soil Texture / Cation Exchange Capacity (CEC):",
                    ["Sandy / Coarse (Low CEC)", "Sandy Loam", "Loam / Silt (Standard)", "Clay Loam", "Heavy Clay (High CEC)"],
                    index=2
                )
                tex_key = "Sandy" if "Sandy /" in soil_tex_choice else ("Sandy Loam" if "Sandy Loam" in soil_tex_choice else ("Clay" if "Clay (" in soil_tex_choice else ("Clay Loam" if "Clay Loam" in soil_tex_choice else "Loam")))
                
                field_sqft = st.number_input("Target Growing Area (Sq. Feet):", min_value=50.0, max_value=500000.0, value=1000.0, step=100.0)
                
                ph_res = calculate_soil_ph_bioavailability(soil_ph_val, tex_key)
                
                st.markdown(f"""
                <div class='disease-card' style='border-color: {ph_res["status_color"]}; margin-top: 15px;'>
                    <h3 style='margin:0; color:{ph_res["status_color"]};'>{ph_res["status"]}</h3>
                    <p style='color:#cbd5e1; margin-top: 8px; line-height:1.5;'>{ph_res["action_summary"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Amendment Dosage Cards
                if ph_res["lime_lbs_1000sqft"] > 0:
                    tot_lime = round(ph_res["lime_lbs_1000sqft"] * (field_sqft / 1000.0), 1)
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 18px; margin-top: 15px; border-left: 4px solid #f59e0b;'>
                        <div class='metric-title' style='color:#f59e0b;'>Agricultural Dolomitic Limestone Required</div>
                        <div class='metric-value' style='font-size: 2rem;'>{tot_lime} lbs</div>
                        <div style='color:#94a3b8; font-size: 0.88rem; margin-top: 6px;'>
                            Rate: <strong>{ph_res["lime_lbs_1000sqft"]} lbs / 1,000 sq ft</strong> ({ph_res["lime_kg_ha"]:.0f} kg/ha).<br>
                            Supplies both <em>Calcium Carbonate</em> and <em>Magnesium Carbonate</em> to neutralize toxic free aluminum (Al³⁺) and restore phosphorus uptake.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif ph_res["sulfur_lbs_1000sqft"] > 0:
                    tot_sulfur = round(ph_res["sulfur_lbs_1000sqft"] * (field_sqft / 1000.0), 1)
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 18px; margin-top: 15px; border-left: 4px solid #ef4444;'>
                        <div class='metric-title' style='color:#ef4444;'>Elemental Agricultural Sulfur Required</div>
                        <div class='metric-value' style='font-size: 2rem;'>{tot_sulfur} lbs</div>
                        <div style='color:#94a3b8; font-size: 0.88rem; margin-top: 6px;'>
                            Rate: <strong>{ph_res["sulfur_lbs_1000sqft"]} lbs / 1,000 sq ft</strong> ({ph_res["sulfur_kg_ha"]:.0f} kg/ha).<br>
                            Soil bacteria oxidize sulfur into sulfuric acid to reduce root-zone alkalinity and release precipitated micronutrients.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 18px; margin-top: 15px; border-left: 4px solid #10b981;'>
                        <div class='metric-title' style='color:#10b981;'>Optimal Agronomic Balance</div>
                        <div class='metric-value' style='font-size: 1.6rem;'>Zero Amendments Needed</div>
                        <div style='color:#94a3b8; font-size: 0.88rem; margin-top: 6px;'>
                            Soil pH is in the prime biological window. Maintain organic matter with finished compost and monitor annually.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_sp2:
                st.markdown("##### 📊 10-Nutrient Bioavailability & Lockout Status")
                if ph_res["lockouts"]:
                    st.error(f"⚠️ **Severe Lockout Hazard (<50% Available):** {', '.join(ph_res['lockouts'])}")
                
                for nutr_name, pct in ph_res["bioavailability"].items():
                    if pct >= 80.0:
                        bar_color = "#10b981"
                        status_tag = "Optimal"
                    elif pct >= 50.0:
                        bar_color = "#f59e0b"
                        status_tag = "Reduced"
                    else:
                        bar_color = "#ef4444"
                        status_tag = "Lockout"
                        
                    st.markdown(f"""
                    <div style='margin-bottom: 10px; background: rgba(30, 41, 59, 0.4); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>
                        <div style='display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;'>
                            <span style='font-weight: 600; color: #f1f5f9;'>{nutr_name}</span>
                            <span style='color: {bar_color}; font-weight: 700;'>{pct:.1f}% ({status_tag})</span>
                        </div>
                        <div style='background: rgba(255,255,255,0.1); height: 7px; border-radius: 4px; overflow: hidden;'>
                            <div style='background: {bar_color}; width: {pct}%; height: 100%; border-radius: 4px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
        return

    # IoT Microclimate & VPD Telemetry View
    if page == "📡 IoT Microclimate & VPD Telemetry":
        st.title("📡 Real-time IoT Microclimate & VPD Outbreak Monitor")
        st.markdown("### Psychrometric Vapor Pressure Deficit (VPD) modeling and farm sensor node telemetry.")
        st.caption("Low VPD (<0.4 kPa) creates stagnant moisture films that trigger exponential fungal zoospore germination.")
        
        tab_iot1, tab_iot2 = st.tabs([
            "🎛️ Single Sensor Calibration & LoRaWAN Stream", 
            "🗺️ Multi-Zone Farm Spatial Grid & Dew Point Simulator"
        ])
        
        with tab_iot1:
            col_vpd1, col_vpd2 = st.columns([1, 1], gap="large")
            with col_vpd1:
                st.markdown("#### 🎛️ Live Sensor Calibration")
                t_input = st.slider("Ambient Canopy Temperature (°C):", min_value=10.0, max_value=45.0, value=22.0, step=0.5)
                rh_input = st.slider("Relative Canopy Humidity (%):", min_value=20.0, max_value=100.0, value=88.0, step=1.0)
                
                vpd_data = calculate_vpd(t_input, rh_input)
                hazard_info = evaluate_vpd_disease_hazard(vpd_data["vpd_kpa"])
                
                st.markdown(f"""
                <div class='disease-card' style='border-color: {hazard_info["hazard_color"]}; margin: 20px 0;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <h3 style='margin:0; color:{hazard_info["hazard_color"]};'>{hazard_info["hazard_badge"]}</h3>
                            <p style='color:#cbd5e1; margin:6px 0 0 0;'>{hazard_info["summary"]}</p>
                        </div>
                        <div style='text-align:right;'>
                            <span style='font-size:2.8rem; font-weight:800; color:{hazard_info["hazard_color"]};'>{vpd_data["vpd_kpa"]}</span>
                            <div style='color:#94a3b8; font-weight:600;'>kPa VPD</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_psi1, col_psi2 = st.columns(2)
                with col_psi1:
                    st.metric("Thermodynamic Dew Point", f"{vpd_data['dew_point_c']} °C", delta=f"{vpd_data['condensation_margin_c']} °C Margin")
                with col_psi2:
                    st.metric("Absolute Humidity", f"{vpd_data['abs_humidity_gm3']} g/m³", delta="Condensing!" if vpd_data['condensation_risk'] else "Safe", delta_color="inverse" if vpd_data['condensation_risk'] else "normal")
                
                if vpd_data['condensation_risk']:
                    st.error("🚨 **Foliar Condensation Alert:** Leaf temperature is within 1.8°C of dew point. Free water droplets are condensing on leaf tissue.")
                
                st.warning(f"🚨 **High-Risk Fungal Pathogens:** {', '.join(hazard_info['target_pathogens'])}")
                st.info(f"⚙️ **Automated Actuator / Field Action:** {hazard_info['field_action']}")
                
            with col_vpd2:
                st.markdown("#### 📈 8-Hour LoRaWAN Sensor Node Stream")
                st.caption("Node #04 — South Greenhouse / Valley Block 2")
                stream_data = generate_mock_iot_telemetry_stream(hours=8)
                df_stream = pd.DataFrame(stream_data).set_index("timestamp")
                
                st.line_chart(df_stream[["vpd_kpa", "temperature_c", "humidity_pct"]])
                st.markdown("##### 📡 Latest Probe Sensor Telemetry")
                st.dataframe(df_stream[["temperature_c", "humidity_pct", "soil_moisture_pct", "vpd_kpa", "hazard_level"]].rename(
                    columns={"temperature_c": "Temp (°C)", "humidity_pct": "Humidity (%)", "soil_moisture_pct": "Soil Moisture (%)", "vpd_kpa": "VPD (kPa)", "hazard_level": "Status"}
                ), use_container_width=True)
                
        with tab_iot2:
            st.markdown("#### 🗺️ Multi-Zone Farm Microclimate Spatial Grid")
            st.markdown("Real-time telemetry across 4 distinct horticultural zones. Identifies microclimate pockets where free leaf condensation accelerates fungal infection.")
            
            multizone = simulate_multizone_farm_telemetry()
            
            st.markdown(f"""
            <div class='disease-card' style='border-color: {multizone["farm_status_color"]}; margin-bottom: 20px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <h3 style='margin:0; color:{multizone["farm_status_color"]};'>Farm Executive Status: {multizone["farm_status"]}</h3>
                        <p style='color:#cbd5e1; margin-top:5px;'>High-Risk Disease Microclimates: <strong>{multizone["high_risk_zones"]} of {multizone["total_zones"]} Zones</strong></p>
                    </div>
                    <span style='background:{multizone["farm_status_color"]}; color:#000; font-weight:800; padding:6px 14px; border-radius:12px;'>SYNCED AT {multizone["timestamp"].split()[-1]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            z_col1, z_col2 = st.columns(2)
            for idx, z in enumerate(multizone["zones"]):
                target_col = z_col1 if idx % 2 == 0 else z_col2
                with target_col:
                    border_c = "#ef4444" if z["condensation_risk"] else ("#f59e0b" if "ELEVATED" in z["hazard_level"] else "#10b981")
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 18px; margin-bottom: 15px; border-left: 5px solid {border_c};'>
                        <div style='display:flex; justify-content:space-between;'>
                            <strong style='font-size:1.1rem; color:#f1f5f9;'>{z["zone_name"]}</strong>
                            <span style='background:rgba(255,255,255,0.1); padding:2px 8px; border-radius:6px; font-size:0.75rem;'>{z["crop"]}</span>
                        </div>
                        <div style='display:flex; gap: 15px; margin-top:10px;'>
                            <div>Temp: <strong>{z["temp_c"]}°C</strong></div>
                            <div>Humidity: <strong>{z["humidity_pct"]}%</strong></div>
                            <div>VPD: <strong style='color:{border_c};'>{z["vpd_kpa"]} kPa</strong></div>
                        </div>
                        <div style='margin-top:8px; font-size:0.85rem; color:#94a3b8;'>
                            Dew Point: <strong>{z["dew_point_c"]}°C</strong> (Margin: <strong>{z["condensation_margin_c"]}°C</strong>) | Abs Humidity: <strong>{z["abs_humidity_gm3"]} g/m³</strong>
                        </div>
                        <div style='margin-top:10px; font-size:0.88rem; color:#f8fafc;'>
                            <strong>Actuator Action:</strong> {z["action"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        return

    # Seasonal Spray Calendar View
    if page == "📅 Seasonal Spray Calendar":
        st.title("📅 Seasonal Crop Protection & Spray Calendar")
        st.markdown("### Phenological stage-by-stage guide to disease prevention, scouting, and proactive spray timing.")
        
        col_c1, col_c2 = st.columns([1.5, 2.5])
        with col_c1:
            selected_crop = st.selectbox("🌱 Choose Crop Phenology:", list(PHENOLOGY_CALENDAR.keys()))
        with col_c2:
            st.info(f"Showing seasonal growth timeline and protection milestones for **{selected_crop}**.")
            
        stages = get_crop_calendar(selected_crop)
        
        for idx, s in enumerate(stages, 1):
            with st.container():
                st.markdown(f"""
                <div class='disease-card' style='padding: 24px; margin: 15px 0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h3 style='margin:0; color: #34d399;'>{s['stage']}</h3>
                        <span style='background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;'>{s['timing']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_st1, col_st2 = st.columns([1, 1])
                with col_st1:
                    st.markdown("##### 🚨 High-Risk Pathogens:")
                    for threat in s["threats"]:
                        st.markdown(f"- 🦠 **{threat}**")
                    st.markdown("##### 🧪 Recommended Preventative Spray:")
                    st.success(f"{s['preventive_spray']}")
                with col_st2:
                    st.markdown("##### 📋 Actionable Field & Cultural Practices:")
                    st.write(s["action"])
                    st.caption("💡 Sprays applied at this stage stop exponential spore amplification before canopy damage occurs.")
                st.markdown("---")
        return

    # Analytics Dashboard View
    if page == "📊 Analytics Dashboard":
        st.title("📊 Analytics Dashboard")
        st.markdown("### Monitor crop health trends, severity metrics, and historical scan data.")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            date_range = st.selectbox("📅 Select Time Range", ["Last 7 Days", "Last 30 Days", "All Time"], index=2)
        with col_f2:
            crop_filter = st.selectbox("🌱 Filter by Crop", ["All Crops", "Apple", "Potato", "Tomato", "Grape", "Corn", "Bell Pepper"])
            
        st.markdown(f"**Showing data for:** {date_range} | {crop_filter}")
        st.markdown("---")
        
        stats = get_summary_stats()
        df_all = get_all_predictions_df()
        avg_damage = 0.0
        if not df_all.empty and "damage_pct" in df_all.columns:
            avg_damage = df_all["damage_pct"].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Scans</div><div class="metric-value">{stats["total"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Diseased Plants</div><div class="metric-value" style="color:#ef4444">{stats["diseased"]}</div></div>', unsafe_allow_html=True)
        with col3:
            health_ratio = (stats["healthy"] / stats["total"] * 100) if stats["total"] > 0 else 0
            st.markdown(f'<div class="metric-card"><div class="metric-title">Health Ratio</div><div class="metric-value">{health_ratio:.1f}%</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Foliar Damage</div><div class="metric-value" style="color:#f59e0b">{avg_damage:.1f}%</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### 🦠 Disease Distribution")
            dist = get_disease_distribution()
            if dist:
                df_dist = pd.DataFrame(list(dist.items()), columns=['Disease', 'Count']).set_index('Disease')
                st.bar_chart(df_dist, color="#f87171")
            else:
                st.info("No disease data available yet.")
                
        with col_chart2:
            st.markdown("#### 📈 Scanning Activity Over Time")
            trends = get_daily_trends()
            if trends:
                df_trends = pd.DataFrame(list(trends.items()), columns=['Date', 'Scans']).set_index('Date')
                st.line_chart(df_trends, color="#10b981")
            else:
                st.info("No trend data available yet.")
                
        st.markdown("---")
        st.markdown("#### 🌾 Farm Lot & Zone Health Matrix")
        zone_summary = get_zone_health_summary()
        if zone_summary:
            df_zones = pd.DataFrame(zone_summary).rename(columns={
                "zone": "Farm Zone / Field Block",
                "total_scans": "Total Scans",
                "healthy": "Healthy Plants",
                "diseased": "Diseased Plants",
                "health_index": "Health Index (%)",
                "avg_damage_pct": "Avg Damage Area (%)"
            })
            st.dataframe(df_zones, use_container_width=True)
        else:
            st.info("No farm zone scans recorded yet.")

        st.markdown("---")
        st.markdown("#### 🗄️ Raw Data Export & History Management")
        
        st.dataframe(df_all, use_container_width=True)
        
        col_dl, col_chk, col_json, col_clr = st.columns([1.2, 1.2, 1.2, 1])
        with col_dl:
            if not df_all.empty:
                recent_recs = get_recent_predictions(limit=1000)
                csv_rfc = export_predictions_to_csv_string(recent_recs)
                st.download_button(
                    label="📥 Download RFC CSV",
                    data=csv_rfc.encode("utf-8"),
                    file_name=f'plantguard_telemetry_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
        with col_chk:
            checklist_html = generate_field_audit_checklist_html()
            st.download_button(
                label="📋 Print Field Sheet",
                data=checklist_html,
                file_name=f'plantguard_field_scouting_sheet_{datetime.datetime.now().strftime("%Y%m%d")}.html',
                mime='text/html',
                use_container_width=True
            )
        with col_json:
            json_export = export_history_json()
            st.download_button(
                label="📦 Backup (.json)",
                data=json_export,
                file_name=f'plantguard_backup_{datetime.datetime.now().strftime("%Y%m%d")}.json',
                mime='application/json',
                use_container_width=True
            )
        with col_clr:
            with st.expander("🗑️ Reset DB"):
                st.warning("This will erase all past scans from the local database.")
                if st.button("Confirm Reset History", type="secondary", use_container_width=True):
                    clear_prediction_history()
                    st.toast("Scan history cleared successfully!")
                    st.rerun()
        
        return # Stop execution of scanner

    # AI Assistant View
    if page == "💬 AI Plant Assistant":
        st.title("💬 AI Plant Assistant & Agronomist")
        st.markdown("### Expert plant pathology, organic remedies, soil fertility, and chemical compatibility intelligence.")
        
        with st.sidebar:
            st.markdown("### ⚙️ Agronomist Engine")
            api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Enter your Gemini API key from Google AI Studio for open-ended conversation. Leave blank to use the built-in offline Agronomy Brain!")
            if api_key:
                st.success("🟢 Mode: Gemini Generative AI Active")
            else:
                st.info("🧠 Mode: Built-in PlantGuard Agronomy Brain Active (No API key needed!)")
                
        # Suggestion chips
        st.markdown("##### 💡 Quick Inquiries:")
        cols_q = st.columns(3)
        selected_prompt = None
        for i, q_text in enumerate(SUGGESTED_QUESTIONS[:3]):
            if cols_q[i].button(q_text, use_container_width=True):
                selected_prompt = q_text
                
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your PlantGuard AI Agronomist. Ask me anything about plant diseases, organic remedies, spray dilution, or soil nutrition. I work offline out-of-the-box or with your Gemini API key!"}
            ]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "🧑‍🌾"):
                st.markdown(msg["content"])
                
        prompt = st.chat_input("Ask about fertilizers, watering, specific diseases...")
        active_input = selected_prompt or prompt
        
        if active_input:
            st.session_state.messages.append({"role": "user", "content": active_input})
            with st.chat_message("user", avatar="🧑‍🌾"):
                st.write(active_input)
                
            with st.chat_message("assistant", avatar="🌿"):
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(
                            'gemini-1.5-flash',
                            system_instruction="You are a certified professional agricultural pathologist and agronomist. Provide structured, actionable, and safety-compliant advice regarding crop diseases, IPM (Integrated Pest Management), organic treatments, and safe chemical applications."
                        )
                        history = []
                        for msg in st.session_state.messages[:-1]:
                            if msg["role"] == "user":
                                history.append({"role": "user", "parts": [msg["content"]]})
                            elif msg["role"] == "assistant":
                                history.append({"role": "model", "parts": [msg["content"]]})
                                
                        chat = model.start_chat(history=history)
                        with st.spinner("AI Agronomist analyzing with Gemini..."):
                            resp = chat.send_message(active_input)
                            response = resp.text
                            st.markdown(response)
                    except Exception as e:
                        st.warning(f"Gemini API note: {e}. Falling back to Local Agronomy Brain.")
                        response = query_agronomy_brain(active_input)
                        st.markdown(response)
                else:
                    with st.spinner("Querying Local Agronomic Knowledge Brain..."):
                        time.sleep(0.3)
                        response = query_agronomy_brain(active_input)
                        st.markdown(response)
                        
            st.session_state.messages.append({"role": "assistant", "content": response})
        return

    # Care & Treatment Handbook View
    if page == "📖 Care & Treatment Handbook":
        st.title("📖 Plant Care & Disease Handbook")
        st.markdown("### Comprehensive encyclopedia of crop diseases, organic remedies, and care guides.")
        
        all_guides = get_all_disease_guides()
        
        # Interactive Search & Filters
        col_hb1, col_hb2, col_hb3 = st.columns([2, 1, 1])
        with col_hb1:
            search_query = st.text_input("🔍 Search by disease name, symptom, or pathogen...", "")
        with col_hb2:
            crop_filter = st.selectbox("🌱 Crop Filter", ["All Crops", "Apple", "Corn (Maize)", "Bell Pepper", "Potato", "Tomato"])
        with col_hb3:
            category_filter = st.selectbox("🔬 Category Filter", ["All Categories", "Fungal Infection", "Bacterial Infection", "Viral Disease", "Pest Infestation", "Healthy Plant"])
            
        with st.expander("🔬 Crop Pathology Vector & Transmission Lookup", expanded=False):
            st.markdown("##### Filter crop pathogens by insect carrier, meteorological vector, or mechanical pathway:")
            col_v1, col_v2 = st.columns([1.5, 2.5])
            with col_v1:
                vector_preset = st.selectbox(
                    "Select Transmission Pathway:",
                    ["All Vectors", "Whitefly", "Spider Mites", "Aphids / Flea Beetles", "Rain Splash", "Wind", "Pruning Tools", "Seed Borne"]
                )
                custom_vec_query = st.text_input("Or enter custom vector keyword:", "")
            with col_v2:
                active_vector = custom_vec_query.strip() if custom_vec_query else (vector_preset if vector_preset != "All Vectors" else "")
                if "Flea" in active_vector:
                    active_vector = "flea"
                if active_vector:
                    v_matches = find_diseases_by_vector(active_vector)
                    st.markdown(f"**Found {len(v_matches)} diseases transmitted via '{active_vector}':**")
                    for vm in v_matches[:6]:
                        st.markdown(f"""
                        <div class='metric-card' style='margin-bottom:6px; padding:8px 12px;'>
                            <div style='display:flex; justify-content:space-between;'>
                                <strong>{vm['title']}</strong>
                                <span style='color:#34d399; font-weight:700;'>⏱️ {vm['incubation_range']}</span>
                            </div>
                            <div style='font-size:0.8rem; color:#94a3b8;'>Pathogen: {vm['pathogen_type']} • Spread: {vm['secondary_spread_rate']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Select a transmission vector or type a keyword to see which crop pathogens share that dispersal mechanism.")

        st.markdown("---")
        
        # Filter guides
        filtered_guides = {}
        for k, guide in all_guides.items():
            # Crop match
            if crop_filter != "All Crops" and guide.get("crop", "").lower() != crop_filter.lower():
                continue
            # Category match
            if category_filter != "All Categories" and category_filter.lower() not in guide.get("category", "").lower():
                continue
            # Search query
            if search_query:
                q = search_query.lower()
                text_pool = f"{guide.get('title', '')} {guide.get('description', '')} {guide.get('symptoms', '')} {guide.get('causes', '')}".lower()
                if q not in text_pool:
                    continue
            filtered_guides[k] = guide
            
        st.markdown(f"**Found {len(filtered_guides)} guides matching your criteria.**")
        
        for k, guide in filtered_guides.items():
            is_hlthy = guide.get("severity_level") == "None"
            icon = "🌿" if is_hlthy else "🦠"
            expander_title = f"{icon} {guide.get('title', k)} ({guide.get('crop')} - {guide.get('category')})"
            
            with st.expander(expander_title, expanded=False):
                # Header badges
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.markdown(f"**Host Crop:** {guide.get('crop')}")
                with col_b2:
                    st.markdown(f"**Classification:** `{guide.get('category')}`")
                with col_b3:
                    sev = guide.get('severity_level')
                    badge_color = "#10b981" if is_hlthy else ("#ef4444" if sev in ["High", "Critical"] else "#f59e0b")
                    st.markdown(f"**Severity Level:** <span style='color:{badge_color}; font-weight:700;'>{sev}</span>", unsafe_allow_html=True)
                    
                st.info(f"**Urgency:** {guide.get('urgency')}")
                st.markdown(f"**Overview:** {guide.get('description')}")
                
                tab_g1, tab_g2, tab_g3, tab_g4, tab_g5, tab_g6 = st.tabs([
                    "🔍 Symptoms & Causes",
                    "⚡ Immediate Actions",
                    "🌿 Organic Remedies",
                    "🧪 Chemical Treatments",
                    "🛡️ Prevention & Nutrition",
                    "🔬 Vector & Incubation"
                ])
                
                with tab_g1:
                    st.markdown(f"**Visual Symptoms:**\n{guide.get('symptoms')}")
                    st.markdown(f"**Causes & Environmental Triggers:**\n{guide.get('causes')}")
                    
                with tab_g2:
                    st.markdown("##### ⚡ Urgent Containment (First-Aid):")
                    actions = guide.get("immediate_actions", [])
                    if isinstance(actions, list):
                        for act in actions:
                            st.markdown(f"- {act}")
                    else:
                        st.write(actions)
                        
                with tab_g3:
                    st.markdown("##### 🌿 Organic & Eco-Friendly Controls:")
                    organic = guide.get("organic_treatment", [])
                    if isinstance(organic, list):
                        for org in organic:
                            st.markdown(f"- {org}")
                    else:
                        st.write(organic)
                        
                with tab_g4:
                    st.markdown("##### 🧪 Chemical & Conventional Treatments:")
                    chem = guide.get("chemical_treatment", [])
                    if isinstance(chem, list):
                        for ch in chem:
                            st.markdown(f"- {ch}")
                    else:
                        st.write(chem)
                        
                with tab_g5:
                    st.markdown("##### 🛡️ Cultural Prevention & Soil Nutrition:")
                    prev = guide.get("preventive_measures", [])
                    if isinstance(prev, list):
                        for p in prev:
                            st.markdown(f"- {p}")
                    else:
                        st.write(prev)
                    st.markdown(f"**Nutrition & Care Regimen:**\n{guide.get('nutrition_and_care', 'N/A')}")
                    
                with tab_g6:
                    epi = get_disease_epidemiology_profile(k)
                    st.markdown("##### 🔬 Pathology Vector & Host Susceptibility Profile:")
                    col_ep1, col_ep2 = st.columns(2)
                    with col_ep1:
                        st.markdown(f"**Pathogen Type:** `{epi.get('pathogen_type', 'Pathogen')}`")
                        inc_min = epi.get('incubation_days_min', 0)
                        inc_max = epi.get('incubation_days_max', 0)
                        if inc_min == 0 and inc_max == 0:
                            st.markdown("**Latent Incubation Period:** `N/A (Healthy)`")
                        else:
                            st.markdown(f"**Latent Incubation Period:** `{inc_min} to {inc_max} Days`")
                        st.markdown(f"**Primary Vectors:** {', '.join(epi.get('primary_vectors', ['N/A']))}")
                        st.markdown(f"**Vector Carrier Insects:** {', '.join(epi.get('vector_carrier_insects', ['None']))}")
                    with col_ep2:
                        st.markdown(f"**Host Vulnerability Stage:** {epi.get('host_susceptibility_stage', 'All stages')}")
                        st.markdown(f"**Secondary Spread Rate:** `{epi.get('secondary_spread_rate', 'Moderate')}`")
                        st.markdown(f"**Environmental Drivers:** {epi.get('environmental_drivers', 'N/A')}")
                    
                    st.markdown("###### 🛑 Biosecurity Quarantine & Containment Checklist:")
                    for q in epi.get("quarantine_measures", []):
                        st.markdown(f"- {q}")
        return

    # Disease Outbreak Risk Forecaster View
    if page == "🌦️ Disease Risk Forecaster":
        st.title("🌦️ Plant Disease Outbreak Risk Forecaster")
        st.markdown("### Meteorological modeling of plant disease outbreaks based on real-time microclimate and pathogen biology.")
        
        # Mode tabs
        mode_tab1, mode_tab2 = st.tabs(["📍 Live City Weather Outbreak Scan", "🎛️ Microclimate Simulator & Preset Scenarios"])
        
        # State management for weather params
        if "weather_data" not in st.session_state:
            st.session_state["weather_data"] = {
                "temp": 20.0,
                "humidity": 85.0,
                "rainfall": 4.5,
                "leaf_wetness": 7.0,
                "wind_speed": 12.0,
                "location_label": "Simulated Field Station"
            }
            
        with mode_tab1:
            st.markdown("#### Real-Time Meteorological Assessment")
            col_search, col_btn = st.columns([3, 1])
            with col_search:
                city_input = st.text_input("Enter your farm/city location (e.g., Sacramento, London, Bengaluru, Nairobi):", value="Sacramento")
            with col_btn:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                fetch_btn = st.button("📡 Fetch Live Climate", use_container_width=True)
                
            st.markdown("##### Quick Location Presets:")
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            preset_city = None
            if col_q1.button("🇺🇸 Salinas Valley, CA", use_container_width=True):
                preset_city = "Salinas"
            if col_q2.button("🇮🇹 Bologna, Italy", use_container_width=True):
                preset_city = "Bologna"
            if col_q3.button("🇮🇳 Nashik, India", use_container_width=True):
                preset_city = "Nashik"
            if col_q4.button("🇪🇸 Valencia, Spain", use_container_width=True):
                preset_city = "Valencia"
                
            active_city = preset_city if preset_city else (city_input if fetch_btn else None)
            
            if active_city:
                with st.spinner(f"Querying meteorological satellite feeds for {active_city}..."):
                    live_res = fetch_live_weather(active_city)
                    if live_res.get("success"):
                        st.session_state["weather_data"] = {
                            "temp": live_res["temp"],
                            "humidity": live_res["humidity"],
                            "rainfall": live_res["rainfall"],
                            "leaf_wetness": live_res["leaf_wetness"],
                            "wind_speed": live_res["wind_speed"],
                            "location_label": live_res["city"]
                        }
                        st.success(f"✅ Meteorological data retrieved for **{live_res['city']}** at {live_res['timestamp']}")
                    else:
                        st.error(f"❌ {live_res.get('error')}")

        with mode_tab2:
            st.markdown("#### Simulate Environmental Conditions & Test Pathogen Triggers")
            selected_preset = st.selectbox("Choose a Climate Scenario Preset:", ["Custom"] + list(WEATHER_PRESETS.keys()))
            if selected_preset != "Custom":
                preset_info = WEATHER_PRESETS[selected_preset]
                st.info(f"ℹ️ **Scenario Profile:** {preset_info['description']}")
                st.session_state["weather_data"] = {
                    "temp": preset_info["temp"],
                    "humidity": preset_info["humidity"],
                    "rainfall": preset_info["rainfall"],
                    "leaf_wetness": preset_info["leaf_wetness"],
                    "wind_speed": preset_info["wind_speed"],
                    "location_label": selected_preset.split(" (")[0]
                }
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.session_state["weather_data"]["temp"] = st.slider("Temperature (°C)", -5.0, 45.0, float(st.session_state["weather_data"]["temp"]), 0.5)
                st.session_state["weather_data"]["humidity"] = st.slider("Relative Humidity (%)", 10.0, 100.0, float(st.session_state["weather_data"]["humidity"]), 1.0)
            with col_s2:
                st.session_state["weather_data"]["rainfall"] = st.slider("Precipitation / 24h Rain (mm)", 0.0, 50.0, float(st.session_state["weather_data"]["rainfall"]), 0.5)
                st.session_state["weather_data"]["leaf_wetness"] = st.slider("Continuous Leaf Wetness (Hours)", 0.0, 24.0, float(st.session_state["weather_data"]["leaf_wetness"]), 0.5)
            with col_s3:
                st.session_state["weather_data"]["wind_speed"] = st.slider("Wind Speed (km/h)", 0.0, 50.0, float(st.session_state["weather_data"]["wind_speed"]), 1.0)
                st.caption("Leaf wetness duration is the primary driver of fungal spore germination (e.g. Mills Apple Scab periods).")

        st.markdown("---")
        
        # Display Current Meteorological Bar
        w = st.session_state["weather_data"]
        st.markdown(f"### 📡 Microclimate Status: **{w.get('location_label', 'Field')}**")
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("🌡️ Temperature", f"{w['temp']} °C", f"{round(w['temp'] * 9/5 + 32, 1)} °F")
        with col_m2:
            st.metric("💧 Humidity", f"{w['humidity']} %")
        with col_m3:
            st.metric("🌧️ Rainfall (24h)", f"{w['rainfall']} mm")
        with col_m4:
            st.metric("🍃 Leaf Wetness", f"{w['leaf_wetness']} hrs")
        with col_m5:
            st.metric("💨 Wind Speed", f"{w['wind_speed']} km/h")
            
        # Target Crop Filter & GDD Thermal Time
        st.markdown("---")
        col_cf, col_gdd = st.columns([1.5, 2.5])
        with col_cf:
            crop_choice = st.selectbox("🌱 Focus on Host Crop:", ["All Crops", "Apple", "Corn (Maize)", "Bell Pepper", "Potato", "Tomato"])
        with col_gdd:
            enable_gdd = st.checkbox("🌡️ Modulate Outbreak Threat by Pathogen GDD Phenology Stage", value=True)
            if enable_gdd:
                farm_gdd = st.slider("Cumulative GDD from Biofix (Thermal Time):", min_value=0.0, max_value=800.0, value=195.0, step=5.0)
                st.session_state["farm_cumulative_gdd"] = farm_gdd
            else:
                farm_gdd = None

        # Calculate Outbreak Risk
        analysis = calculate_disease_outbreak_risk(
            temp=w["temp"],
            humidity=w["humidity"],
            rainfall=w["rainfall"],
            leaf_wetness=w["leaf_wetness"],
            wind_speed=w["wind_speed"],
            crop_filter=crop_choice,
            cumulative_gdd=farm_gdd
        )
        
        # Overall Biosecurity Card
        score = analysis["overall_score"]
        gauge_color = analysis["overall_color"]
        
        st.markdown(f"""
        <div class='disease-card' style='border-color: {gauge_color}; margin: 20px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 style='margin:0; color: #f8fafc;'>Farm Biosecurity Outbreak Index</h2>
                    <p style='color: #94a3b8; margin: 4px 0 0 0;'>Epidemiological probability based on current temperature, leaf wetness, and humidity.</p>
                </div>
                <div style='text-align: right;'>
                    <span style='font-size: 2.5rem; font-weight: 800; color: {gauge_color};'>{analysis["overall_icon"]} {score}%</span>
                    <div style='font-weight: 700; color: {gauge_color}; text-transform: uppercase; letter-spacing: 1px;'>{analysis["overall_level"]}</div>
                </div>
            </div>
            <div style='width: 100%; background-color: #334155; border-radius: 999px; height: 12px; margin: 20px 0 10px 0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);'>
                <div style='background-color: {gauge_color}; width: {min(100.0, max(4.0, score))}%; height: 12px; border-radius: 999px; box-shadow: 0 0 15px {gauge_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if analysis["top_threat"]:
            top_t = analysis["top_threat"]
            st.error(f"🚨 **Primary High-Risk Pathogen Threat:** **{top_t['name']}** on {', '.join(top_t['crops'])} ({top_t['score']}% Outbreak Probability)")
            
        st.markdown("#### 🔬 Pathogen Risk Breakdown")
        
        for d in analysis["diseases"]:
            with st.expander(f"{d['icon']} **{d['name']}** — Outbreak Threat: {d['score']}% ({d['level']})", expanded=(d['score'] >= 50.0)):
                col_d1, col_d2 = st.columns([1, 1])
                with col_d1:
                    st.markdown(f"**Vulnerable Crops:** `{'`, `'.join(d['crops'])}`")
                    st.markdown(f"**Pathogen Classification:** `{d['pathogen_type']}`")
                    st.markdown(f"**Optimal Trigger Climate:** {d['optimum']}")
                with col_d2:
                    st.markdown(f"**Threat Severity:** <span style='color:{d['badge_color']}; font-weight:700;'>{d['level']}</span>", unsafe_allow_html=True)
                    st.warning(f"**Emergency Alert:** {d['emergency']}")
                    
                if d.get("phenology"):
                    p = d["phenology"]
                    st.markdown(f"**🌡️ GDD Phenological Stage:** <span style='color:{p['risk_color']}; font-weight:700;'>{p['stage']}</span> ({p['cumulative_gdd']} GDD) • Inoculum Ripening: `{p['ascospore_maturation_pct']}`", unsafe_allow_html=True)
                    st.caption(f"**IPM Biological Advisory:** {p['action_advisory']}")
                    
                st.markdown(f"**🛡️ Pre-emptive Countermeasure:**\n{d['advice']}")
                
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🗺️ 3D Regional Outbreak Radar & Geospatial Stations")
        st.caption("Real-time telemetry and pathogen pressure contours across adjacent agricultural farm monitoring nodes.")

        # Determine coordinates from current station or defaults
        curr_loc_name = w.get("location_label", "Davis Agricultural Valley, CA")
        lat = 38.54
        lon = -121.74
        for p_name, p_val in WEATHER_PRESETS.items():
            if p_name in curr_loc_name or curr_loc_name in p_name:
                lat = p_val["lat"]
                lon = p_val["lon"]
                break

        stations = get_regional_monitoring_stations(base_lat=lat, base_lon=lon, location_name=curr_loc_name, base_score=score)
        stations_df = pd.DataFrame(stations)

        view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=9.5,
            pitch=35,
            bearing=15
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=stations_df,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=1,
            radius_min_pixels=10,
            radius_max_pixels=75,
            line_width_min_pixels=2,
            get_line_color=[255, 255, 255, 180]
        )

        r = pdk.Deck(
            layers=[scatter_layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}\nThreat Level: {level} ({score}% Outbreak Index)\nCoordinates: {lat}, {lon}"},
            map_style="dark"
        )
        st.pydeck_chart(r)

        with st.expander("📡 Regional Monitoring Station Telemetry", expanded=False):
            st.dataframe(
                stations_df[["name", "level", "score", "lat", "lon"]].rename(
                    columns={"name": "Station Name", "level": "Threat Level", "score": "Risk Index (%)", "lat": "Latitude", "lon": "Longitude"}
                ),
                use_container_width=True
            )
                
        return

    # Pathogen Phenology & Degree-Day Engine View
    if page == "🌡️ Pathogen Phenology & Degree-Day Engine":
        st.title("🌡️ Pathogen Phenology & Growing Degree Day (GDD) Engine")
        st.markdown("### Physiological thermal time modeling for fungal spore maturation, infection windows, and insect vector emergence.")
        st.caption("Tracks physiological degree-day accumulation (thermal time) from a verified biofix date to predict biological vulnerability windows (Gadoury & MacHardy, FAST, and Gubler-Thomas models).")
        
        col_ctl1, col_ctl2 = st.columns([1.5, 2.5], gap="medium")
        with col_ctl1:
            models_list = list(PATHOGEN_PHENOLOGY_MODELS.keys())
            selected_model = st.selectbox("🔬 Target Pathogen / Pest Model:", models_list)
            model_info = PATHOGEN_PHENOLOGY_MODELS[selected_model]
            
            st.markdown(f"**Host Crop:** `{model_info['crop']}`")
            st.markdown(f"**Pathogen:** *{model_info['scientific_name']}*")
            st.markdown(f"**Base Temperature ($T_{{base}}$):** `{model_info['base_temp_c']} °C`")
            st.markdown(f"**Upper Thermal Cap ($T_{{cap}}$):** `{model_info['cap_temp_c']} °C`")
            st.markdown(f"**Biofix Trigger Event:** `{model_info['biofix_event']}`")
            st.caption(f"Standard IPM Model: {model_info['model_reference']}")
            
        with col_ctl2:
            st.markdown("##### ⚙️ Biofix Date & Thermal Accumulation Input")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                biofix_date = st.date_input("🗓️ Field Biofix Date:", value=datetime.date.today() - datetime.timedelta(days=22))
            with col_b2:
                daily_rate = st.slider("📈 Average Daily GDD Rate (°C-days/day):", min_value=3.0, max_value=25.0, value=11.5, step=0.5)
                
            # Quick stage presets
            st.markdown("##### 🧪 Fast-Load Thermal Milestones:")
            col_p1, col_p2, col_p3 = st.columns(3)
            loaded_gdd = None
            if selected_model == "Apple Scab":
                if col_p1.button("🌱 Early Ripening (65 GDD)", use_container_width=True):
                    loaded_gdd = 65.0
                if col_p2.button("🚨 Peak Discharge (195 GDD)", use_container_width=True):
                    loaded_gdd = 195.0
                if col_p3.button("🍂 Late Depletion (320 GDD)", use_container_width=True):
                    loaded_gdd = 320.0
            elif selected_model == "Tomato Early Blight":
                if col_p1.button("🌱 Vegetative (75 GDD)", use_container_width=True):
                    loaded_gdd = 75.0
                if col_p2.button("🚨 FAST 300 GDD Trigger (315 GDD)", use_container_width=True):
                    loaded_gdd = 315.0
                if col_p3.button("🔄 Polycyclic Phase (480 GDD)", use_container_width=True):
                    loaded_gdd = 480.0
            else:
                if col_p1.button("🌱 Early Stage (50 GDD)", use_container_width=True):
                    loaded_gdd = 50.0
                if col_p2.button("🚨 Critical Window (210 GDD)", use_container_width=True):
                    loaded_gdd = 210.0
                if col_p3.button("🔄 Advanced Cycling (360 GDD)", use_container_width=True):
                    loaded_gdd = 360.0

            if "active_gdd_input" not in st.session_state:
                st.session_state["active_gdd_input"] = 195.0
            if loaded_gdd is not None:
                st.session_state["active_gdd_input"] = loaded_gdd
                
            active_gdd = st.slider(
                "Cumulative Growing Degree-Days (GDD) since Biofix:",
                min_value=0.0, max_value=800.0,
                value=float(st.session_state["active_gdd_input"]),
                step=5.0,
                key="gdd_main_slider"
            )
            st.session_state["farm_cumulative_gdd"] = active_gdd

        # Evaluate phenology
        pheno = evaluate_pathogen_phenology(selected_model, active_gdd, biofix_date=str(biofix_date))
        
        st.markdown("<br>", unsafe_allow_html=True)
        # 5 Metric Cards
        col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
        with col_k1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Thermal Time</div>
                <div class='metric-value'>{pheno['cumulative_gdd']}</div>
                <div style='color: #94a3b8; font-size: 0.85rem;'>GDD (Base {pheno['base_temp_c']}°C)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_k2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Life-Cycle Stage</div>
                <div style='font-size: 1.15rem; font-weight: 800; color:{pheno['risk_color']}; line-height:1.2; margin-top:8px;'>{pheno['stage']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_k3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Inoculum Status</div>
                <div style='font-size: 1.25rem; font-weight: 800; color:#f8fafc; margin-top:8px;'>{pheno['ascospore_maturation_pct']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_k4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Next Stage In</div>
                <div class='metric-value' style='font-size: 2.2rem;'>{pheno['gdd_to_next_stage']}</div>
                <div style='color: #94a3b8; font-size: 0.85rem;'>GDD Remaining</div>
            </div>
            """, unsafe_allow_html=True)
        with col_k5:
            days_est = max(0, int(math.ceil(pheno['gdd_to_next_stage'] / max(1.0, daily_rate)))) if pheno['gdd_to_next_stage'] > 0 else 0
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Days to Next Stage</div>
                <div class='metric-value' style='font-size: 2.2rem; color:#38bdf8;'>~{days_est}d</div>
                <div style='color: #94a3b8; font-size: 0.85rem;'>@ {daily_rate}°C-days/d</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Prescription Banner
        st.markdown(f"""
        <div class='disease-card' style='border-color: {pheno['risk_color']}; margin: 15px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h3 style='margin:0; color: #f8fafc;'>🛡️ Timed IPM Protection Advisory</h3>
                    <p style='color: #cbd5e1; margin: 4px 0 0 0;'>Biological intervention protocol aligned with current physiological thermal maturity.</p>
                </div>
                <div>
                    <span style='background:{pheno['risk_color']}22; color:{pheno['risk_color']}; padding:6px 14px; border-radius:999px; font-weight:800; border:1px solid {pheno['risk_color']}; text-transform:uppercase;'>{pheno['risk_level']}</span>
                </div>
            </div>
            <p style='margin: 16px 0 0 0; font-size: 1.05rem; color: #f8fafc; font-weight: 500;'>{pheno['action_advisory']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Milestone Timeline Cards
        st.markdown("#### 🧬 Pathogen Thermal Milestone Sequence")
        for m_idx, m in enumerate(pheno["all_milestones"]):
            is_active = (m["stage"] == pheno["stage"])
            is_past = (active_gdd > m["gdd_max"])
            
            card_border = pheno['risk_color'] if is_active else ("#334155" if is_past else "rgba(255,255,255,0.1)")
            bg_style = "rgba(16, 185, 129, 0.05)" if is_past else ("rgba(239, 68, 68, 0.08)" if is_active else "transparent")
            badge_icon = "✅ Completed" if is_past else ("📍 ACTIVE CURRENT STAGE" if is_active else "⏳ Upcoming")
            
            with st.container():
                st.markdown(f"""
                <div style='border: 1px solid {card_border}; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; background: {bg_style};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='font-size: 0.9rem; font-weight: 700; color: {m["risk_color"]};'>{badge_icon} • {m["gdd_min"]} to {m["gdd_max"]} GDD</span>
                            <h4 style='margin: 4px 0; color: #f8fafc;'>{m["stage"]}</h4>
                            <span style='color: #94a3b8; font-size: 0.9rem;'>Inoculum Status: <b>{m["ascospore_maturation_pct"]}</b></span>
                        </div>
                        <div style='text-align: right;'>
                            <span style='color: {m["risk_color"]}; font-weight: 700;'>{m["risk_level"]}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if is_active:
                    st.progress(pheno["stage_progress_pct"] / 100.0)
                    st.caption(f"Stage Completion: {pheno['stage_progress_pct']}% • {pheno['gdd_to_next_stage']} GDD until transition to next biological phase.")

        # Projected Target Dates Table & 30-Day Curve
        st.markdown("<br>", unsafe_allow_html=True)
        col_proj1, col_proj2 = st.columns([1.2, 1], gap="medium")
        
        with col_proj1:
            st.markdown("#### 📅 Projected Developmental Threshold Crossing Dates")
            upcoming = estimate_upcoming_milestones(selected_model, active_gdd, projected_daily_gdd=daily_rate)
            if upcoming:
                proj_df = pd.DataFrame(upcoming)[["stage", "target_gdd", "days_remaining", "projected_date", "risk_level"]].rename(
                    columns={
                        "stage": "Developmental Stage",
                        "target_gdd": "Target GDD",
                        "days_remaining": "Days Remaining",
                        "projected_date": "Projected Date",
                        "risk_level": "Threat Level"
                    }
                )
                st.dataframe(proj_df, use_container_width=True)
            else:
                st.success("✅ Pathogen has completed primary developmental stages. Currently in mature secondary cycling.")
                
        with col_proj2:
            st.markdown("#### 📈 Thermal Accumulation Trajectory")
            # Generate simulated GDD series
            days_span = 30
            records = []
            cur_date = datetime.date.today() - datetime.timedelta(days=days_span)
            for d_idx in range(days_span):
                d_date = cur_date + datetime.timedelta(days=d_idx)
                # Seasonal curve
                t_max = 16.0 + (d_idx * 0.3)
                t_min = 6.0 + (d_idx * 0.2)
                records.append({"date": str(d_date), "t_max": t_max, "t_min": t_min})
            series_res = calculate_cumulative_gdd_series(records, base_temp=model_info["base_temp_c"], cap_temp=model_info["cap_temp_c"])
            df_gdd = pd.DataFrame(series_res["series"])
            st.line_chart(df_gdd.set_index("date")[["cumulative_gdd", "daily_gdd"]])
            st.caption(f"30-day cumulative degree-day trajectory (Average: {series_res['average_daily_gdd']} GDD/day).")
            
        return

    # Spray & Dosage Calculator View
    if page == "⚖️ Spray & Dosage Calculator":
        st.title("⚖️ Precision Spray & Fertilizer Dosage Calculator")
        st.markdown("### Compute exact carrier water volume, chemical/organic concentrate, and tank mixing recipes tailored to your farm.")
        
        tab_dose, tab_tank, tab_drift = st.tabs([
            "⚖️ Field Dosage & Economic ROI Estimator", 
            "🧪 Tank-Mix Incompatibility & Jar Test Simulator",
            "🎯 Nozzle Selection & Spray Drift Simulator"
        ])

        with tab_dose:

            col_left, col_right = st.columns([1.1, 1.3], gap="large")

            with col_left:
                st.markdown("#### 1️⃣ Plot & Plant Scale")
                area_unit = st.selectbox(
                    "📐 Select Scaling Method:",
                    ["Garden Bed (Sq. Feet)", "Plant Count (Individual Plants)", "Greenhouse (Sq. Meters)", "Farm Acreage (Acres)"]
                )

                # Default scale values
                if area_unit == "Garden Bed (Sq. Feet)":
                    area_val = st.number_input("Enter Total Canopy Area (Sq. Ft):", min_value=10.0, max_value=50000.0, value=250.0, step=25.0)
                elif area_unit == "Plant Count (Individual Plants)":
                    area_val = st.number_input("Enter Number of Crop Plants:", min_value=1.0, max_value=5000.0, value=25.0, step=5.0)
                elif area_unit == "Greenhouse (Sq. Meters)":
                    area_val = st.number_input("Enter Greenhouse Floor Area (m²):", min_value=5.0, max_value=10000.0, value=100.0, step=10.0)
                else:
                    area_val = st.number_input("Enter Total Farm Acreage (Acres):", min_value=0.1, max_value=500.0, value=1.0, step=0.25)

                canopy_stage = st.selectbox(
                    "🌿 Canopy Density & Growth Stage:",
                    ["Medium Canopy", "Early Seedlings / Low Canopy", "Dense Fruiting Canopy / Orchard Trees"]
                )

                st.markdown("---")
                st.markdown("#### 2️⃣ Treatment Formulation")

                type_filter = st.radio(
                    "Category:",
                    ["All Formulations", "Organic Botanicals & Bio-Pesticides", "Conventional / Chemical Protectants", "Foliar Fertilizers & Tonics"],
                    horizontal=True
                )

                filtered_prods = []
                for p_name in all_prods:
                    details = get_product_details(p_name)
                    cat = details.get("type", "")
                    if type_filter == "Organic Botanicals & Bio-Pesticides" and "Organic" not in cat and "Biological" not in cat:
                        continue
                    if type_filter == "Conventional / Chemical Protectants" and "Chemical" not in cat:
                        continue
                    if type_filter == "Foliar Fertilizers & Tonics" and "Fertilizer" not in cat and "Mineral" not in cat and "Micronutrient" not in cat:
                        continue
                    filtered_prods.append(p_name)

                selected_product = st.selectbox("🧪 Select Product Formulation:", filtered_prods)
                prod_meta = get_product_details(selected_product)

                st.info(f"**Target Pathogens & Pests:** {', '.join(prod_meta.get('recommended_for', []))}\n\n**Approved Crops:** {', '.join(prod_meta.get('crops', []))}")

            with col_right:
                st.markdown("#### 3️⃣ Tailored Spray Prescription")

                dosage_res = compute_spray_dosage(
                    product_name=selected_product,
                    area_type=area_unit,
                    area_val=float(area_val),
                    plant_stage=canopy_stage
                )

                # Display Quick Metrics
                col_qm1, col_qm2 = st.columns(2)
                with col_qm1:
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 20px 15px;'>
                        <div class='metric-title'>Total Water Carrier</div>
                        <div class='metric-value' style='font-size: 2.2rem;'>{dosage_res['water_liters']} L</div>
                        <div style='color: #94a3b8; font-size: 0.9rem; margin-top: 5px;'>approx. <strong>{dosage_res['water_gallons']} US Gallons</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_qm2:
                    st.markdown(f"""
                    <div class='metric-card' style='padding: 20px 15px;'>
                        <div class='metric-title'>Concentrate Amount</div>
                        <div class='metric-value' style='font-size: 2.2rem;'>{dosage_res['active_amount']} {dosage_res['active_unit']}</div>
                        <div style='color: #10b981; font-size: 0.95rem; margin-top: 5px; font-weight:600;'>{dosage_res['household_measure']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Tank Mix Steps
                with st.container():
                    st.markdown("##### 🧪 WALES Tank Mixing Protocol:")
                    for step in dosage_res["mixing_steps"]:
                        st.markdown(step)

                st.markdown("---")

                # Safety & Field Application Guidance
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f"**⏰ Best Application Window:**\n{dosage_res['best_time']}")
                    st.markdown(f"**🦺 Required PPE:**\n{dosage_res['ppe']}")
                with col_s2:
                    st.markdown(f"**⏳ Pre-Harvest Interval (PHI):** `{dosage_res['phi_days']} Days`")
                    st.markdown(f"**🚫 Re-Entry Interval (REI):** `{dosage_res['rei_hours']} Hours`")

                # Commercial Treatment Economics & ROI Expander
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("💰 Commercial Pack Purchasing & ROI Estimator", expanded=True):
                    st.markdown("##### Estimate commercial container purchases, application labor, and financial ROI.")

                    col_ec1, col_ec2 = st.columns(2)
                    with col_ec1:
                        bench = DEFAULT_PRODUCT_PRICING.get(selected_product, {"pack_size": 1000, "unit": dosage_res['active_unit'], "price": 28.0})
                        custom_price = st.number_input(
                            f"Commercial Pack Price ($ USD per {bench['pack_size']}{bench['unit']}):",
                            min_value=1.0, value=float(bench["price"]), step=2.0
                        )
                        labor_rate = st.number_input("Field Sprayer Labor Rate ($ / hour):", min_value=5.0, value=18.0, step=1.0)
                    with col_ec2:
                        loss_risk = st.slider("Estimated Potential Crop Loss Without Treatment (%):", min_value=5, max_value=85, value=35, step=5)
                        crop_val_bench = BENCHMARK_CROP_VALUES.get(area_unit, 1.0) * float(area_val)
                        crop_rev = st.number_input("Estimated Total Harvest Revenue ($ USD):", min_value=0.0, value=float(round(crop_val_bench, 2)), step=50.0)

                    econ = estimate_treatment_economics(
                        product_name=selected_product,
                        active_amount=dosage_res['active_amount'],
                        active_unit=dosage_res['active_unit'],
                        area_type=area_unit,
                        area_val=float(area_val),
                        custom_price_per_pack=custom_price,
                        custom_labor_rate=labor_rate,
                        crop_value_estimate=crop_rev if crop_rev > 0 else None,
                        untreated_loss_pct=float(loss_risk)
                    )

                    col_ei1, col_ei2 = st.columns(2)
                    with col_ei1:
                        st.markdown(f"""
                        <div class='metric-card' style='padding: 18px 12px; margin-bottom: 12px;'>
                            <div class='metric-title'>Containers Needed</div>
                            <div class='metric-value' style='font-size: 2rem;'>{econ['packs_needed']} <span style='font-size:1.1rem; color:#94a3b8;'>Packs</span></div>
                            <div style='color: #cbd5e1; font-size: 0.85rem;'>{econ['pack_size']}{econ['pack_unit']} per pack (${econ['total_product_cost']})</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_ei2:
                        st.markdown(f"""
                        <div class='metric-card' style='padding: 18px 12px; margin-bottom: 12px;'>
                            <div class='metric-title'>Labor & Spraying</div>
                            <div class='metric-value' style='font-size: 2rem;'>${econ['labor_cost']}</div>
                            <div style='color: #cbd5e1; font-size: 0.85rem;'>approx. {econ['labor_hours']} application hours</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 18px; margin: 10px 0;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <div style='color: #94a3b8; font-size: 0.9rem; text-transform:uppercase;'>Protected Harvest Value</div>
                                <div style='color: #34d399; font-size: 1.8rem; font-weight:800;'>${econ['potential_crop_loss']} Saved</div>
                            </div>
                            <div style='text-align:right;'>
                                <div style='color: #94a3b8; font-size: 0.9rem; text-transform:uppercase;'>Treatment ROI</div>
                                <div style='color: #10b981; font-size: 1.8rem; font-weight:800;'>{econ['roi_ratio']}x Return</div>
                            </div>
                        </div>
                        <div style='margin-top: 10px; color: #cbd5e1; font-size: 0.9rem;'>
                            Total investment of <strong>${econ['total_investment']}</strong> (product + labor) safeguards <strong>${econ['potential_crop_loss']}</strong> in yield loss, generating <strong>${econ['net_economic_savings']}</strong> in net economic savings.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # EPA Chemical Registration & FRAC Code Lookup
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🏷️ EPA Chemical Registration & FRAC Resistance Lookup", expanded=False):
                    st.markdown("##### Verify EPA Registration Number, active ingredient purity, and resistance rotation.")
                    chem_query = st.text_input("Enter EPA Reg. No (e.g., 100-1098, 70051-107, 4-61, 264-1152) or Trade Name:", value="100-1098")
                    chem_res = lookup_chemical_by_epa_or_name(chem_query)
                    if chem_res["match_found"]:
                        rec = chem_res["record"]
                        st.success(f"**{rec['trade_name']}** — EPA Reg. No: `{rec['epa_reg_no']}`")
                        col_ch1, col_ch2 = st.columns(2)
                        with col_ch1:
                            st.markdown(f"**Active Ingredient:** {rec['active_ingredient']}")
                            st.markdown(f"**FRAC Group:** `{rec['frac_group']}`")
                            st.markdown(f"**Resistance Risk:** `{rec['resistance_risk']}`")
                            st.markdown(f"**Signal Word:** `{rec['signal_word']}`")
                        with col_ch2:
                            st.markdown(f"**Max Applications/Year:** `{rec['max_annual_apps']}`")
                            st.markdown(f"**Rainfastness:** `{rec['rainfast_hours']} Hours`")
                            st.markdown(f"**Pollinators / Bees:** {rec['pollinator_safety']}")
                            st.markdown(f"**Tank Mix Restrictions:** {', '.join(rec['restricted_mixes'])}")
                        st.info(f"🔄 **Anti-Resistance FRAC Rotation:** Rotate with {', '.join(rec['recommended_rotation'])}")
                    else:
                        st.warning(chem_res["message"])

                    st.markdown("---")
                    st.markdown("##### 🔄 Multi-Season FRAC Anti-Resistance Rotation Planner")
                    st.caption("Enter the sequence of fungicides applied this season to audit your resistance risk and prevent pathogen immunity.")
                    all_epa_names = [item['trade_name'] for item in get_all_chemical_catalog()]
                    rotation_seq = st.multiselect(
                        "Season Application Sequence (in order):",
                        all_epa_names,
                        default=["Heritage 50 WDG", "Luna Tranquility Fungicide"],
                        key="seasonal_frac_rotation_select"
                    )
                    if rotation_seq:
                        rot_audit = evaluate_seasonal_rotation_plan(rotation_seq)
                        st.markdown(f"""
                        <div class='disease-card' style='border-color: {rot_audit["badge_color"]}; margin: 12px 0;'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <strong style='color:{rot_audit["badge_color"]}; font-size:1.1rem;'>{rot_audit["status"]}</strong>
                                <span style='background:{rot_audit["badge_color"]}; color:#000; font-weight:800; padding:3px 10px; border-radius:10px;'>Resistance Risk: {rot_audit["score"]}/100</span>
                            </div>
                            <p style='color:#cbd5e1; margin-top:8px;'>{rot_audit["recommendation"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if rot_audit["violations"]:
                            for viol in rot_audit["violations"]:
                                st.error(f"Violation: {viol['description']}")

                    st.markdown("---")
                    st.markdown("##### 🔍 EPA Registered Chemical Catalog Multi-Criteria Filter")
                    st.caption("Search across legal EPA fungicides by FRAC mode of action, worker safety intervals (REI), and pollinator toxicity.")
                    col_cf1, col_cf2, col_cf3 = st.columns([1.2, 1, 1])
                    with col_cf1:
                        frac_filter_query = st.text_input("Filter by FRAC Group (e.g. Group 11, Group 7, Group M, Group 3):", "", key="frac_cat_filter_input")
                    with col_cf2:
                        max_rei_choice = st.selectbox("Max Field Re-entry (REI):", ["Any REI", "4 Hours", "12 Hours", "24 Hours"], key="max_rei_cat_select")
                        max_rei_val = int(max_rei_choice.split()[0]) if "Hour" in max_rei_choice else None
                    with col_cf3:
                        pollinator_only = st.checkbox("🐝 Pollinator Safe Only", value=False, key="poll_safe_cat_check")
                    
                    filtered_chems = filter_chemicals(
                        frac_group=frac_filter_query,
                        max_rei_hours=max_rei_val,
                        pollinator_safe_only=pollinator_only
                    )
                    st.markdown(f"**Found {len(filtered_chems)} EPA registered products:**")
                    if filtered_chems:
                        cf_data = []
                        for c in filtered_chems:
                            cf_data.append({
                                "Trade Name": c["trade_name"],
                                "EPA Reg #": c["epa_reg_no"],
                                "FRAC Group": c["frac_group"],
                                "REI (Hrs)": f"{c.get('rei_hours', 12)}h",
                                "PHI (Days)": f"{c.get('phi_days', 0)}d",
                                "Resistance Risk": c["resistance_risk"],
                                "Pollinator Safety": c["pollinator_safety"]
                            })
                        st.dataframe(pd.DataFrame(cf_data), use_container_width=True)

                # Printable Prescription Text Download
                recipe_text = f"""PLANTGUARD AI - SPRAY & FERTILIZER PRESCRIPTION SHEET
    =============================================================
    Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    Target Crop Scale: {area_val} ({area_unit})
    Canopy Stage: {canopy_stage}

    PRODUCT PRESCRIPTION:
    - Product Formulation: {dosage_res['product_name']} ({dosage_res['category']})
    - Carrier Water Volume: {dosage_res['water_liters']} Liters ({dosage_res['water_gallons']} US Gallons)
    - Active Product Needed: {dosage_res['active_amount']} {dosage_res['active_unit']} ({dosage_res['household_measure']})

    STEP-BY-STEP TANK MIX PROTOCOL:
    {chr(10).join(dosage_res['mixing_steps'])}

    SAFETY & COMPLIANCE:
    - Pre-Harvest Interval (PHI): {dosage_res['phi_days']} Days
    - Field Re-Entry Interval (REI): {dosage_res['rei_hours']} Hours
    - Required PPE: {dosage_res['ppe']}
    - Recommended Window: {dosage_res['best_time']}

    ECONOMIC & TREATMENT BUDGET ESTIMATE:
    - Commercial Packs Required: {econ['packs_needed']} x {econ['pack_size']}{econ['pack_unit']} (${econ['total_product_cost']})
    - Application Labor: ~{econ['labor_hours']} hrs (${econ['labor_cost']})
    - Total Investment: ${econ['total_investment']} (Estimated Saved Crop ROI: {econ['roi_ratio']}x)
    =============================================================
    Generated by PlantGuard AI
    """
                st.download_button(
                    label="📄 Download Printable Spray Recipe (.txt)",
                    data=recipe_text,
                    file_name=f"PlantGuard_Spray_Recipe_{selected_product.split()[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        with tab_tank:
            st.markdown("#### 🧪 Tank-Mix Incompatibility Checker & 1-Pint Jar Test Simulator")
            st.markdown("Physical precipitation, chemical antagonism, and phytotoxicity can destroy spray rigs and burn entire crops. Verify chemical compatibility and simulate a proportional benchtop jar test before mixing in the field.")
            
            col_tm1, col_tm2 = st.columns([1.1, 1.2], gap="large")
            with col_tm1:
                st.markdown("##### 1️⃣ Formulation Tank-Mix Selection")
                
                # Presets
                st.markdown("<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:6px;'>Quick Test Scenarios:</div>", unsafe_allow_html=True)
                col_pr1, col_pr2 = st.columns(2)
                preset_pick = None
                with col_pr1:
                    if st.button("⚠️ Calcium + Epsom Salt", use_container_width=True):
                        preset_pick = ["Calcium Nitrate", "Epsom Salt (Magnesium Sulfate)"]
                    if st.button("🚫 Sulfur + Neem Oil", use_container_width=True):
                        preset_pick = ["Wettable Sulfur (Micronized 80%)", "Neem Oil (Cold-Pressed 70%)"]
                with col_pr2:
                    if st.button("🚫 Copper + Bacillus", use_container_width=True):
                        preset_pick = ["Liquid Copper Fungicide", "Bacillus subtilis Bio-Fungicide"]
                    if st.button("🟢 Safe: Neem + Bicarb", use_container_width=True):
                        preset_pick = ["Neem Oil (Cold-Pressed 70%)", "Potassium Bicarbonate"]
                        
                if preset_pick:
                    st.session_state["tank_mix_selection"] = preset_pick
                    
                current_selection = st.session_state.get("tank_mix_selection", ["Calcium Nitrate", "Epsom Salt (Magnesium Sulfate)"])
                current_selection = [p for p in current_selection if p in all_prods]
                if len(current_selection) < 2:
                    current_selection = [all_prods[0], all_prods[1]]
                    
                selected_mix = st.multiselect(
                    "Select 2 or more products to test together in tank:",
                    all_prods,
                    default=current_selection,
                    key="tank_mix_multiselect"
                )
                
                if len(selected_mix) >= 2:
                    compat_res = check_tank_mix_compatibility(selected_mix)
                    
                    st.markdown(f"""
                    <div class='disease-card' style='border-color: {compat_res["badge_color"]}; margin: 15px 0;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h3 style='margin:0; color:{compat_res["badge_color"]};'>{compat_res["status"]}</h3>
                            <span style='background:{compat_res["badge_color"]}; color:#000; font-weight:800; padding:4px 12px; border-radius:12px; font-size:0.8rem;'>{compat_res["overall_safety"]}</span>
                        </div>
                        <p style='color:#cbd5e1; margin-top: 8px; line-height:1.5;'>{compat_res["summary"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if compat_res["conflicts"]:
                        st.markdown("##### ⚠️ Specific Incompatibility Conflicts:")
                        for c in compat_res["conflicts"]:
                            with st.expander(f"🚫 {c['title']} ({c['severity']} Severity)", expanded=True):
                                st.markdown(f"**Hazard Reaction:** {c['hazard']}")
                                st.warning(f"💡 **Remedial Action:** {c['action']}")
                                if c["withholding_days"] > 0:
                                    st.error(f"⏳ **Mandatory Safe Interval:** Wait minimum `{c['withholding_days']} Days` between applications.")
                    else:
                        st.success("✅ **Chemical & Physical Harmony:** No direct chemical precipitation or biological neutralizations known for this combination.")
                        
                    # WALES Order for this mix
                    st.markdown("##### 🔄 Optimal WALES Tank Loading Sequence:")
                    st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>Always add products in this precise order with continuous mechanical agitation:</p>", unsafe_allow_html=True)
                    for idx, prod in enumerate(compat_res["wales_order"]):
                        st.markdown(f"**{idx+1}. {prod}**")
                else:
                    st.info("👆 Please select at least 2 products above to analyze tank compatibility.")

            with col_tm2:
                st.markdown("##### 2️⃣ 1-Pint (500 ml) Benchtop Mason Jar Test Simulator")
                st.markdown("Never risk a whole spray tank. Conduct this scaled-down benchtop test to verify physical compatibility with your specific field water.")
                
                if len(selected_mix) >= 2:
                    jar_proto = get_jar_test_protocol(selected_mix)
                    
                    st.markdown("###### ⚖️ Proportional Jar Dosages (Scaled to 500 ml Water):")
                    for d in jar_proto["jar_dosages"]:
                        st.markdown(f"- **{d['product']}:** `{d['jar_amount']} {d['unit']}` ({d['household_measure']})")
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("###### 🧪 Jar Test Execution Protocol:")
                    for s in jar_proto["protocol_steps"]:
                        st.markdown(s)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_jp, col_jf = st.columns(2)
                    with col_jp:
                        st.markdown(f"""
                        <div class='metric-card' style='border-left: 4px solid #10b981; padding: 12px;'>
                            <div style='color: #10b981; font-weight:700; margin-bottom:5px;'>🟢 PASS CRITERIA</div>
                            <div style='font-size:0.83rem; color:#cbd5e1;'>{jar_proto["evaluation"]["pass_criteria"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_jf:
                        st.markdown(f"""
                        <div class='metric-card' style='border-left: 4px solid #ef4444; padding: 12px;'>
                            <div style='color: #ef4444; font-weight:700; margin-bottom:5px;'>🔴 FAIL CRITERIA</div>
                            <div style='font-size:0.83rem; color:#cbd5e1;'>{jar_proto["evaluation"]["fail_criteria"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Select 2 or more products on the left to generate the tailored 1-pint jar test formulation.")

        with tab_drift:
            st.markdown("#### 🎯 Nozzle Orifice Selection & Spray Drift Hazard Simulator (ASABE S572.1 / ISO)")
            st.markdown("Off-target chemical drift causes expensive legal liability, water contamination, and reduced canopy deposition. Calibrate nozzle droplet size, boom height, and wind threshold before pulling into the field.")
            
            col_ns1, col_ns2 = st.columns([1.2, 1.3], gap="large")
            with col_ns1:
                st.markdown("##### 1️⃣ Sprayer Rig Calibration")
                nozzle_choices = get_available_nozzle_types()
                selected_nozzle = st.selectbox("Select Nozzle Body Type:", nozzle_choices, index=0)
                
                nozzle_info = AVAILABLE_NOZZLE_TYPES[selected_nozzle]
                st.caption(f"ℹ️ {nozzle_info['description']}")
                
                boom_press = st.slider("Boom Operating Pressure (PSI):", min_value=15.0, max_value=90.0, value=40.0, step=5.0)
                wind_speed = st.slider("Crosswind Speed (mph):", min_value=0.0, max_value=18.0, value=6.0, step=0.5)
                boom_height = st.slider("Boom Height Above Crop Canopy (inches):", min_value=15.0, max_value=42.0, value=20.0, step=1.0)
                ground_speed = st.slider("Tractor Ground Speed (mph):", min_value=2.0, max_value=12.0, value=5.0, step=0.5)
                
            with col_ns2:
                st.markdown("##### 2️⃣ Droplet Spectrum & Drift Physics")
                drift_calc = calculate_spray_drift_and_nozzle_specs(
                    nozzle_type=selected_nozzle,
                    boom_pressure_psi=boom_press,
                    wind_speed_mph=wind_speed,
                    boom_height_inches=boom_height,
                    ground_speed_mph=ground_speed
                )
                
                col_dm1, col_dm2, col_dm3 = st.columns(3)
                with col_dm1:
                    st.markdown(f"""
                    <div class='metric-card' style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#94a3b8;'>VMD Droplet Size</div>
                        <div style='font-size:1.8rem; font-weight:800; color:{drift_calc["color_code"]};'>{drift_calc["vmd_microns"]} µm</div>
                        <div style='font-size:0.75rem; color:{drift_calc["color_code"]}; font-weight:700;'>{drift_calc["asabe_category"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_dm2:
                    d_color = "#ef4444" if drift_calc["drift_level"] == "CRITICAL" else ("#f59e0b" if drift_calc["drift_level"] == "MODERATE" else "#10b981")
                    st.markdown(f"""
                    <div class='metric-card' style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#94a3b8;'>Drift Hazard Score</div>
                        <div style='font-size:1.8rem; font-weight:800; color:{d_color};'>{drift_calc["drift_hazard_score"]}/100</div>
                        <div style='font-size:0.75rem; color:{d_color}; font-weight:700;'>{drift_calc["drift_level"]} HAZARD</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_dm3:
                    st.markdown(f"""
                    <div class='metric-card' style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#94a3b8;'>Buffer Setback</div>
                        <div style='font-size:1.8rem; font-weight:800; color:#3b82f6;'>{drift_calc["setback_buffer_ft"]} ft</div>
                        <div style='font-size:0.75rem; color:#94a3b8;'>({drift_calc["setback_buffer_m"]} meters)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"""
                <div class='disease-card' style='border-color:{drift_calc["color_code"]}; margin-top:15px;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <strong style='color:{drift_calc["color_code"]}; font-size:1.05rem;'>Droplet Spectrum: {drift_calc["asabe_category"]}</strong>
                        <span style='color:#cbd5e1; font-size:0.85rem;'>Fines (&lt;150 µm): <strong>{drift_calc["fine_droplets_pct"]}%</strong></span>
                    </div>
                    <p style='color:#cbd5e1; font-size:0.9rem; margin:8px 0;'>{drift_calc["description"]}</p>
                    <div style='font-size:0.85rem; color:#94a3b8;'>
                        <strong>Canopy Penetration:</strong> {drift_calc["canopy_penetration"]} ({drift_calc["penetration_pct"]}%)<br>
                        <strong>Recommended Agronomic Target:</strong> {drift_calc["best_for"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Meteorological Wind Status:** {drift_calc['wind_status']}")
                
                if drift_calc["warnings"]:
                    for w in drift_calc["warnings"]:
                        st.warning(f"⚠️ {w}")
        return

    # Scanner View Content
    st.markdown('<div class="hero-title">🌿 PlantGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Next-Generation Plant Disease Detection powered by Artificial Intelligence.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    predictor, error_msg = load_predictor()

    if error_msg:
        st.error(f"**Model Error:** {error_msg}")
        st.info("💡 **Tip:** If you haven't trained the model yet, please run `python train.py` first with a valid dataset.")
        st.stop()

    st.markdown("#### 📷 Input Leaf Image")
    
    images_to_process = []
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Image(s)", "📸 Take a Photo", "📂 Local Folder", "🧪 Demo Specimens (1-Click)"])
    
    with tab1:
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            if len(uploaded_files) > 1:
                st.info("💡 **Canopy Survey Tip:** You uploaded multiple leaves. Try the **📦 Multi-Leaf Canopy Batch Scanner** in the navigation menu for aggregate Canopy Health Index (CHI) scoring and bulk triage export!")
            if len(uploaded_files) > 20:
                st.warning("⚠️ Large number of files selected! If Chrome freezes, please use the 'Local Folder' tab instead.")
            images_to_process.extend(uploaded_files)
            
    with tab2:
        camera_img = st.camera_input("Take a picture of the plant leaf")
        if camera_img:
            camera_img.name = "camera_capture.jpg"
            images_to_process.append(camera_img)
            
    with tab3:
        st.info("💡 Processing a large dataset? Enter the local folder path here to bypass the browser upload entirely and prevent Chrome from freezing.")
        folder_path = st.text_input("Enter absolute folder path (e.g., C:\\Users\\...\\Dataset)")
        if folder_path and os.path.isdir(folder_path):
            valid_extensions = ('.jpg', '.jpeg', '.png')
            local_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
            if local_files:
                st.success(f"Found {len(local_files)} images in folder.")
                        
                if st.button(f"Queue Images for Analysis"):
                    # Limit to a reasonable number to avoid locking up the processing forever
                    limit = min(len(local_files), 100)
                    for f in local_files[:limit]:
                        images_to_process.append(LocalImageFile(f))
                    if len(local_files) > 100:
                        st.warning(f"Only the first 100 images were queued to prevent excessive processing time.")
            else:
                st.warning("No valid images found in the specified folder.")
        elif folder_path:
            st.error("Directory does not exist. Please check the path.")

    with tab4:
        st.markdown("##### 🔬 Instant-Test Verified Specimen Library")
        st.caption("Instantly test the AI neural network and computer vision lesion segmenter without needing to upload local files.")
        
        sample_cols = st.columns(3)
        sample_specs = [
            ("Tomato Early Blight", "assets/samples/tomato_early_blight.jpg", "Fungal concentric target spots with chlorotic halo."),
            ("Apple Scab", "assets/samples/apple_scab.jpg", "Olive-green to dark brown velvety foliar lesions."),
            ("Healthy Green Leaf", "assets/samples/healthy_leaf.jpg", "Pristine photosynthetic surface with intact chlorophyll.")
        ]
        
        for idx, (title, img_path, desc) in enumerate(sample_specs):
            with sample_cols[idx]:
                if os.path.exists(img_path):
                    st.image(img_path, caption=title, use_column_width=True)
                st.markdown(f"**{title}**")
                st.caption(desc)
                if st.button(f"⚡ Load Specimen", key=f"demo_btn_{idx}", use_container_width=True):
                    st.session_state["queued_demo_leaf"] = img_path

        if st.session_state.get("queued_demo_leaf") and os.path.exists(st.session_state["queued_demo_leaf"]):
            st.success(f"Loaded demo specimen: `{os.path.basename(st.session_state['queued_demo_leaf'])}`")
            images_to_process.append(LocalImageFile(st.session_state["queued_demo_leaf"]))
            
    if images_to_process:
        st.markdown("---")
        st.markdown("#### Analysis Results")
        
        if st.button("Analyze Image(s)", use_container_width=True):
            with st.spinner("Analyzing images... Please wait."):
                time.sleep(1) # Simulate slight processing delay for UX
                
                for uploaded_file in images_to_process:
                    st.markdown(f"### Results for: `{uploaded_file.name}`")
                    col1, col2 = st.columns([1, 2])
                    
                    try:
                        if hasattr(uploaded_file, 'path'):
                            image = Image.open(uploaded_file.path)
                        else:
                            image = Image.open(uploaded_file)
                        
                        with col1:
                            st.image(image, caption="Uploaded Image", use_column_width=True)
                            
                        with col2:
                            results = predictor.predict(image)
                            
                            confidence = results['confidence']
                            disease_class = results['predicted_class']
                            is_healthy = results['is_healthy']
                            
                            # Real Computer Vision Leaf Canopy & Lesion Segmentation
                            cv_analysis = analyze_leaf_damage(image)
                            damage_pct = cv_analysis["damage_percentage"]
                            stage_label = cv_analysis["stage_label"]
                            lesion_count = cv_analysis["lesion_count"]
                            severity_color = cv_analysis["severity_color"]
                            
                            # Confidence Threshold Check
                            if confidence < 0.20:
                                st.warning("⚠️ The model has low confidence. Please upload a clearer leaf image for better accuracy.")
                            
                            # Log to Database with real CV metrics
                            add_prediction(uploaded_file.name, disease_class.replace('_', ' '), confidence, is_healthy, damage_pct=damage_pct, severity_grade=stage_label)
                            
                            # Display Results Card
                            status_color = "healthy-text" if is_healthy else "disease-text"
                            status_icon = "✅" if is_healthy else "🦠"
                            
                            html_content = f"""
                            <div class='disease-card'>
                                <h3 style='margin-bottom: 15px;'>{status_icon} Status: <span class='{status_color}'>{disease_class.replace('_', ' ')}</span></h3>
                                <div style='margin-bottom: 8px; color: #cbd5e1;'><strong>AI Model Confidence: {confidence * 100:.2f}%</strong></div>
                                <div style='width: 100%; background-color: #334155; border-radius: 999px; height: 8px; margin-bottom: 20px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);'>
                                    <div style='background-color: #10b981; width: {confidence*100}%; height: 8px; border-radius: 999px; box-shadow: 0 0 10px rgba(16,185,129,0.5);'></div>
                                </div>
                            """
                            
                            # Display Real Computer Vision Phytopathology Severity
                            if not is_healthy:
                                html_content += f"""
                                <div style='margin-bottom: 8px; color: #cbd5e1;'><strong>🔬 Foliar Lesion Surface Area: <span style='color:{severity_color};'>{damage_pct}%</span> ({stage_label})</strong></div>
                                <div style='width: 100%; background-color: #334155; border-radius: 999px; height: 8px; margin-bottom: 12px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);'>
                                    <div style='background-color: {severity_color}; width: {min(100.0, max(3.0, damage_pct))}%; height: 8px; border-radius: 999px; box-shadow: 0 0 10px {severity_color};'></div>
                                </div>
                                <div style='font-size: 0.9rem; color: #94a3b8;'>Detected <strong>{lesion_count}</strong> active necrotic/chlorotic lesion clusters across the leaf blade.</div>
                                """
                            else:
                                html_content += f"""
                                <div style='margin-bottom: 8px; color: #34d399;'><strong>🌿 Photosynthetic Health: {100.0 - damage_pct:.1f}% Vigorous Blade</strong></div>
                                <div style='font-size: 0.9rem; color: #94a3b8;'>Clean cuticle structure with negligible lesion formation.</div>
                                """
                                
                            html_content += "</div>"
                            st.markdown(html_content, unsafe_allow_html=True)
                            
                            # Hands-free Audio Summary Briefing for Field Farmers
                            speech_summary = f"PlantGuard Diagnostic Report for {uploaded_file.name}. Primary diagnosis: {disease_class.replace('_', ' ')}. AI confidence: {int(confidence * 100)} percent. Quantitative foliar damage is {damage_pct} percent, classified as {stage_label}."
                            if not is_healthy:
                                speech_summary += f" Immediate action recommended: {cv_analysis['field_advice']}"
                            
                            clean_speech = speech_summary.replace('"', '').replace("'", "").replace('\n', ' ')
                            safe_key = uploaded_file.name.replace('.', '_').replace(' ', '_')
                            html_audio = f"""
                            <div style="margin: 8px 0 16px 0;">
                                <button id="btn_{safe_key}" style="background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 10px 20px; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; font-size: 0.95rem;">
                                    🔊 <span>Listen to Audio Diagnostic Summary (Hands-Free)</span>
                                </button>
                                <script>
                                    document.getElementById('btn_{safe_key}').addEventListener('click', function() {{
                                        if ('speechSynthesis' in window) {{
                                            window.speechSynthesis.cancel();
                                            var utter = new SpeechSynthesisUtterance("{clean_speech}");
                                            utter.rate = 0.95;
                                            window.speechSynthesis.speak(utter);
                                        }} else {{
                                            alert('Speech synthesis not supported in this browser.');
                                        }}
                                    }});
                                </script>
                            </div>
                            """
                            components.html(html_audio, height=65)
                            
                            info = get_disease_info(disease_class)
                            
                            # Generate Heatmap image array
                            heatmap_img = None
                            if results.get('heatmap') is not None:
                                heatmap = results['heatmap']
                                heatmap = np.uint8(255 * heatmap)
                                jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                                
                                original_img_arr = np.array(image.convert("RGB"))
                                jet = cv2.resize(jet, (original_img_arr.shape[1], original_img_arr.shape[0]))
                                superimposed_img = jet * 0.45 + original_img_arr * 0.55
                                heatmap_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                            
                            # Compute Spectral, Physical Dimension & GDD Phenology Metrics
                            spectral_res = calculate_canopy_spectral_indices(image)
                            phys_dim = estimate_physical_leaf_dimensions(image)
                            pheno_res = evaluate_pathogen_phenology(results['predicted_class'], cumulative_gdd=st.session_state.get("farm_cumulative_gdd", 195.0))

                            # Generate Enhanced PDF with CV, Spectral and Phenology Analysis
                            epi_data = get_disease_epidemiology_profile(results['predicted_class'])
                            pdf_path = generate_pdf_report(
                                image, heatmap_img, results, info, 
                                cv_analysis=cv_analysis,
                                epidemiology_data=epi_data,
                                spectral_analysis=spectral_res,
                                phenology_data=pheno_res
                            )
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="📄 Download Diagnostic PDF Report",
                                    data=f,
                                    file_name=f"{uploaded_file.name}_diagnostic_report.pdf",
                                    mime="application/pdf",
                                    key=f"dl_{uploaded_file.name}",
                                    use_container_width=True
                                )
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Visual Tabs for Inspection
                            st.markdown("#### 🔬 Detailed Visual & Pathological Inspection")
                            vtab1, vtab2, vtab3, vtab4, vtab5 = st.tabs([
                                "🧠 AI Focus (Grad-CAM)",
                                "🔬 CV Lesion Segmentation",
                                "🌿 Spectral Chlorophyll & Biometrics",
                                "🌡️ Pathogen GDD Phenology",
                                "📊 Probabilities"
                            ])
                            
                            with vtab1:
                                if heatmap_img is not None:
                                    st.image(heatmap_img, caption="Grad-CAM Convolutional Attention Heatmap (Highlights leaf areas driving model prediction)", use_column_width=True)
                                else:
                                    st.info("Grad-CAM visualization not available for this input.")
                                    
                            with vtab2:
                                st.image(cv_analysis["overlay_image"], caption=f"Computer Vision Segmentation: Green leaf perimeter with highlighted necrotic lesions ({damage_pct}% affected)", use_column_width=True)
                                st.info(f"**Biological Severity Assessment:** {cv_analysis['field_advice']}")
                                
                            with vtab3:
                                col_vm1, col_vm2 = st.columns([1, 1.2], gap="medium")
                                with col_vm1:
                                    st.image(spectral_res["spectral_map"], caption="False-Color Viridis Chlorophyll Density Map (Yellow: Active chloroplasts | Purple/Blue: Senescent/chlorotic tissue)", use_column_width=True)
                                with col_vm2:
                                    st.markdown(f"##### 🌿 Multispectral Chlorophyll & Physical Dimensions")
                                    st.markdown(f"**SPAD Chlorophyll Proxy:** <span style='color:{spectral_res['status_color']}; font-weight:800; font-size:1.2rem;'>{spectral_res['spad_chlorophyll_proxy']} / 80</span> — `{spectral_res['chlorophyll_status']}`", unsafe_allow_html=True)
                                    st.caption(spectral_res["diagnostic_desc"])
                                    
                                    col_sm1, col_sm2 = st.columns(2)
                                    with col_sm1:
                                        st.markdown(f"**Green Leaf Index (GLI):** `{spectral_res['mean_gli']:+.3f}`")
                                        st.markdown(f"**Vegetative Index (VARI):** `{spectral_res['mean_vari']:+.3f}`")
                                        st.markdown(f"**Triangular Greenness (TGI):** `{spectral_res['mean_tgi']}`")
                                    with col_sm2:
                                        st.markdown(f"**Photosynthetic Lamina:** `{spectral_res['photosynthetic_efficiency_pct']}%`")
                                        st.markdown(f"**Leaf Surface Area:** `~{phys_dim['leaf_area_cm2']} cm²`")
                                        st.markdown(f"**Absolute Lesion Area:** `~{phys_dim['lesion_area_cm2']} cm² ({phys_dim['lesion_area_mm2']} mm²)`")
                                        
                                    st.caption(f"Estimated Dimensions: {phys_dim['leaf_length_cm']} cm length × {phys_dim['leaf_width_cm']} cm width • Aspect Ratio: {phys_dim['aspect_ratio']} • Circularity: {phys_dim['circularity_form_factor']}")
                                
                            with vtab4:
                                st.markdown(f"##### 🌡️ Thermal Phenology & Biofix Stage ({pheno_res['model_key']})")
                                col_ph1, col_ph2 = st.columns([1.2, 1], gap="medium")
                                with col_ph1:
                                    st.markdown(f"**Current Life Stage:** <span style='color:{pheno_res['risk_color']}; font-weight:800; font-size:1.15rem;'>{pheno_res['stage']}</span>", unsafe_allow_html=True)
                                    st.markdown(f"**Inoculum Ripening / Discharge:** `{pheno_res['ascospore_maturation_pct']}`")
                                    st.markdown(f"**Biofix Event:** `{pheno_res['biofix_event']}` (Base {pheno_res['base_temp_c']}°C)")
                                    st.markdown(f"**Thermal Accumulation:** `{pheno_res['cumulative_gdd']} GDD`")
                                    st.markdown(f"**Progress to Next Stage:** `{pheno_res['stage_progress_pct']}%` ({pheno_res['gdd_to_next_stage']} GDD remaining)")
                                    st.progress(pheno_res["stage_progress_pct"] / 100.0)
                                with col_ph2:
                                    st.markdown(f"**Biological Threat:** <span style='background:{pheno_res['risk_color']}22; color:{pheno_res['risk_color']}; padding:4px 10px; border-radius:8px; font-weight:700; border:1px solid {pheno_res['risk_color']}66;'>{pheno_res['risk_level']}</span>", unsafe_allow_html=True)
                                    st.info(f"**Timed IPM Spray Advisory:**\n{pheno_res['action_advisory']}")

                            with vtab5:
                                st.markdown("##### 🔬 Top Differential Diagnoses:")
                                if results.get("top_3"):
                                    for t_rank, t_item in enumerate(results["top_3"], 1):
                                        c_pct = t_item["confidence"] * 100
                                        t_badge = "🟢 Primary Match" if t_rank == 1 else f"#{t_rank} Differential"
                                        st.markdown(f"**{t_badge}:** `{t_item.get('crop')} — {t_item.get('disease')}` (**{c_pct:.2f}%**)")
                                        st.progress(min(1.0, max(0.01, t_item["confidence"])))
                                st.markdown("<br>", unsafe_allow_html=True)
                                probs = results['all_probabilities']
                                df_probs = pd.DataFrame(list(probs.items()), columns=['Disease', 'Probability']).sort_values(by='Probability', ascending=False).head(5)
                                st.bar_chart(df_probs.set_index('Disease'))
                            
                            # Display Disease Information and Treatment Care Prescription
                            st.markdown("### 🤖 AI Agronomist Prescription & Care Regimen")
                            
                            with st.chat_message("assistant", avatar="🌿"):
                                if not is_healthy:
                                    st.write(f"Hello! I detected **{disease_class.replace('_', ' ')}** on your plant. Here is your tailored agronomic prescription:")
                                    
                                    # Urgency and Category Banner
                                    col_inf1, col_inf2, col_inf3 = st.columns(3)
                                    with col_inf1:
                                        st.markdown(f"**Classification:** `{info.get('category', 'Pathology')}`")
                                    with col_inf2:
                                        st.markdown(f"**Severity Rating:** `{info.get('severity_level', 'Moderate')}`")
                                    with col_inf3:
                                        st.markdown(f"**Crop:** `{info.get('crop', 'Crop')}`")
                                        
                                    st.warning(f"**Action Urgency:** {info.get('urgency', 'Immediate attention recommended')}")
                                    
                                    t1, t2, t3, t4, t5 = st.tabs([
                                        "⚡ First-Aid & Checklist",
                                        "🌿 Organic Remedies",
                                        "🧪 Chemical Treatments",
                                        "🛡️ Cultural Prevention",
                                        "💧 Nutrition & Soil"
                                    ])
                                    
                                    with t1:
                                        st.markdown("#### ⚡ Immediate First-Aid Steps")
                                        immed = info.get("immediate_actions", [])
                                        if isinstance(immed, list):
                                            for step in immed:
                                                st.markdown(f"- {step}")
                                        else:
                                            st.write(immed)
                                            
                                        st.markdown("---")
                                        st.markdown("##### 📋 Field Action Checklist (Track Your Progress):")
                                        st.checkbox(f"Isolate / bottom prune affected foliage ({uploaded_file.name})", key=f"chk1_{uploaded_file.name}")
                                        st.checkbox("Sterilize pruning shears and tools with 70% alcohol", key=f"chk2_{uploaded_file.name}")
                                        st.checkbox("Switch to ground-level drip watering (halt overhead sprinklers)", key=f"chk3_{uploaded_file.name}")
                                        st.checkbox("Apply prescribed organic or chemical treatment spray", key=f"chk4_{uploaded_file.name}")
                                        
                                    with t2:
                                        st.markdown("#### 🌿 Organic & Eco-Friendly Controls")
                                        org_list = info.get("organic_treatment", [])
                                        if isinstance(org_list, list):
                                            for item in org_list:
                                                st.markdown(f"- {item}")
                                        else:
                                            st.write(org_list)
                                            
                                    with t3:
                                        st.markdown("#### 🧪 Chemical Treatments & Fungicides")
                                        chem_list = info.get("chemical_treatment", [])
                                        if isinstance(chem_list, list):
                                            for item in chem_list:
                                                st.markdown(f"- {item}")
                                        else:
                                            st.write(chem_list)
                                        st.caption("⚠️ Always adhere strictly to label instructions and personal protective equipment (PPE) guidelines.")
                                        
                                    with t4:
                                        st.markdown("#### 🛡️ Long-term Cultural Prevention")
                                        st.markdown(f"**Pathology Overview:** {info.get('description', '')}")
                                        st.markdown(f"**Root Causes:** {info.get('causes', '')}")
                                        prev_list = info.get("preventive_measures", [])
                                        if isinstance(prev_list, list):
                                            for p in prev_list:
                                                st.markdown(f"- {p}")
                                        else:
                                            st.write(prev_list)
                                            
                                    with t5:
                                        st.markdown("#### 💧 Nutrition, Watering & Soil Care")
                                        st.write(info.get("nutrition_and_care", "Maintain balanced organic fertility."))
                                        
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    col_act1, col_act2, col_act3 = st.columns(3)
                                    with col_act1:
                                        st.info("⚖️ **Prepare Spray Tank:** Head to **⚖️ Spray & Dosage Calculator** to compute exact carrier volume & dilution!")
                                    with col_act2:
                                        st.warning("🌦️ **Check Climate Risk:** Visit **🌦️ Disease Risk Forecaster** to check if weather triggers a secondary outbreak!")
                                    with col_act3:
                                        st.success("📅 **Growth Calendar:** Check **📅 Seasonal Spray Calendar** for proactive phenological protection milestones!")
                                else:
                                    st.write(f"Great news! Your **{disease_class.replace('_', ' ')}** specimen is in excellent health! Here is your optimal care guide to keep it thriving:")
                                    
                                    col_h1, col_h2 = st.columns(2)
                                    with col_h1:
                                        st.success("🟢 Status: Optimal Health & Photosynthetic Vigor")
                                    with col_h2:
                                        st.info("🌿 Recommended Regimen: Routine Maintenance & Scouting")
                                        
                                    ht1, ht2, ht3 = st.tabs(["🌱 Growth & Maintenance", "💧 Watering & Nutrition", "🛡️ Proactive Protection"])
                                    
                                    with ht1:
                                        st.markdown("#### 🌱 Optimal Growth Conditions")
                                        st.write(info.get("description", "Vigorous foliage with intact natural defenses."))
                                        st.markdown(f"**Key Health Indicators:** {info.get('symptoms', 'Crisp green leaf blade with healthy venation.')}")
                                        
                                    with ht2:
                                        st.markdown("#### 💧 Nutrient & Hydration Regimen")
                                        st.write(info.get("nutrition_and_care", "Ensure balanced feeding."))
                                        
                                    with ht3:
                                        st.markdown("#### 🛡️ Proactive Scouting & Prevention")
                                        prevs = info.get("preventive_measures", [])
                                        if isinstance(prevs, list):
                                            for pr in prevs:
                                                st.markdown(f"- {pr}")
                                        else:
                                            st.write(prevs)
                                        orgs = info.get("organic_treatment", [])
                                        if isinstance(orgs, list):
                                            for og in orgs:
                                                st.markdown(f"- {og}")
                                        
                            st.markdown("---")
                            st.write("**Was this prediction helpful?**")
                            fb_col1, fb_col2 = st.columns(2)
                            with fb_col1:
                                if st.button("👍 Yes, spot on!", key=f"yes_{uploaded_file.name}", use_container_width=True):
                                    st.toast("Thank you for your feedback! 🌟")
                            with fb_col2:
                                if st.button("👎 No, incorrect", key=f"no_{uploaded_file.name}", use_container_width=True):
                                    st.toast("Thanks, we will use this to improve our model! 🛠️")
                                    
                    except Exception as e:
                        st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
                    st.markdown("---")
    else:
        st.info("Please upload one or more images to see the analysis results.")

if __name__ == "__main__":
    main()
