import streamlit as st
from PIL import Image
import google.generativeai as genai
import time
import os
import cv2
import numpy as np
from predict import PlantDiseasePredictor
from utils.db import add_prediction, get_recent_predictions, get_summary_stats, get_disease_distribution, get_daily_trends, get_all_predictions_df
from utils.pdf_generator import generate_pdf_report
from utils.disease_expert import get_disease_expert_info, get_all_disease_guides
from utils.weather_risk import fetch_live_weather, calculate_disease_outbreak_risk, WEATHER_PRESETS
from utils.dosage_calculator import compute_spray_dosage, get_all_products, get_product_details, TREATMENT_DATABASE
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
                "🌦️ Disease Risk Forecaster", 
                "⚖️ Spray & Dosage Calculator",
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
                st.caption(f"📅 {item['timestamp']} - {item['filename']}")
        
        st.markdown("---")
        st.warning("⚠️ **Disclaimer:** This is an educational tool and should not replace professional agricultural advice.")

    # Analytics Dashboard View
    if page == "📊 Analytics Dashboard":
        st.title("📊 Analytics Dashboard")
        st.markdown("### Monitor crop health trends and historical data.")
        
        # Interactive filters for UI enhancement
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            date_range = st.selectbox("📅 Select Time Range", ["Last 7 Days", "Last 30 Days", "All Time"], index=2)
        with col_f2:
            crop_filter = st.selectbox("🌱 Filter by Crop", ["All Crops", "Apple", "Potato", "Tomato", "Grape", "Corn"])
            
        st.markdown(f"**Showing data for:** {date_range} | {crop_filter}")
        st.markdown("---")
        
        stats = get_summary_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Scans</div><div class="metric-value">{stats["total"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Diseased Plants</div><div class="metric-value" style="color:#ef4444">{stats["diseased"]}</div></div>', unsafe_allow_html=True)
        with col3:
            health_ratio = (stats["healthy"] / stats["total"] * 100) if stats["total"] > 0 else 0
            st.markdown(f'<div class="metric-card"><div class="metric-title">Overall Health Ratio</div><div class="metric-value">{health_ratio:.1f}%</div></div>', unsafe_allow_html=True)
        
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
        st.markdown("#### 🗄️ Raw Data Export")
        df_all = get_all_predictions_df()
        
        st.dataframe(df_all, use_container_width=True)
        
        if not df_all.empty:
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f'plantguard_history_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
        
        return # Stop execution of the scanner

    # AI Assistant View
    if page == "💬 AI Plant Assistant":
        st.title("💬 AI Plant Assistant")
        st.markdown("### Ask any questions about plant care, diseases, and gardening!")
        
        api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get your API key from Google AI Studio")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your PlantGuard AI assistant powered by Gemini. Please enter your API key in the sidebar and ask me anything about your plants!"}
            ]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "🧑‍🌾"):
                st.write(msg["content"])
                
        if prompt := st.chat_input("Ask about fertilizers, watering, specific diseases..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍🌾"):
                st.write(prompt)
                
            with st.chat_message("assistant", avatar="🌿"):
                if not api_key:
                    response = "Please provide your Gemini API key in the sidebar to get real responses from the AI."
                    st.warning(response)
                else:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        history = []
                        for msg in st.session_state.messages[:-1]:
                            if msg["role"] == "user":
                                history.append({"role": "user", "parts": [msg["content"]]})
                            elif msg["role"] == "assistant" and "Gemini API key" not in msg["content"]:
                                history.append({"role": "model", "parts": [msg["content"]]})
                                
                        chat = model.start_chat(history=history)
                        
                        with st.spinner("Thinking..."):
                            resp = chat.send_message(prompt)
                            response = resp.text
                            st.write(response)
                    except Exception as e:
                        response = f"An error occurred: {str(e)}"
                        st.error(response)
                        
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
                
                tab_g1, tab_g2, tab_g3, tab_g4, tab_g5 = st.tabs([
                    "🔍 Symptoms & Causes",
                    "⚡ Immediate Actions",
                    "🌿 Organic Remedies",
                    "🧪 Chemical Treatments",
                    "🛡️ Prevention & Nutrition"
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
            
        # Target Crop Filter
        st.markdown("---")
        col_cf, _ = st.columns([2, 2])
        with col_cf:
            crop_choice = st.selectbox("🌱 Focus on Host Crop:", ["All Crops", "Apple", "Corn (Maize)", "Bell Pepper", "Potato", "Tomato"])

        # Calculate Outbreak Risk
        analysis = calculate_disease_outbreak_risk(
            temp=w["temp"],
            humidity=w["humidity"],
            rainfall=w["rainfall"],
            leaf_wetness=w["leaf_wetness"],
            wind_speed=w["wind_speed"],
            crop_filter=crop_choice
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
                    
                st.markdown(f"**🛡️ Pre-emptive Countermeasure:**\n{d['advice']}")
                
        return

    # Spray & Dosage Calculator View
    if page == "⚖️ Spray & Dosage Calculator":
        st.title("⚖️ Precision Spray & Fertilizer Dosage Calculator")
        st.markdown("### Compute exact carrier water volume, chemical/organic concentrate, and tank mixing recipes tailored to your farm.")
        
        all_prods = get_all_products()
        
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
            
        return

    # Scanner View Content
    st.markdown('<div class="hero-title">🌿 PlantGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Next-Generation Plant Disease Detection powered by Artificial Intelligence.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Model Initialization
    @st.cache_resource
    def load_predictor():
        predictor = PlantDiseasePredictor()
        try:
            predictor.load_model()
            return predictor, None
        except Exception as e:
            return None, str(e)
            
    predictor, error_msg = load_predictor()

    if error_msg:
        st.error(f"**Model Error:** {error_msg}")
        st.info("💡 **Tip:** If you haven't trained the model yet, please run `python train.py` first with a valid dataset.")
        st.stop()

    st.markdown("#### 📷 Input Leaf Image")
    
    images_to_process = []
    
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image(s)", "📸 Take a Photo", "📂 Local Folder"])
    
    with tab1:
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
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
                
                class LocalImageFile:
                    def __init__(self, path):
                        self.path = path
                        self.name = os.path.basename(path)
                        
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
                            
                            # Confidence Threshold Check
                            if confidence < 0.20:
                                st.warning("⚠️ The model has low confidence. Please upload a clearer leaf image for better accuracy.")
                            
                            # Log to Database
                            add_prediction(uploaded_file.name, disease_class.replace('_', ' '), confidence, is_healthy)
                            
                            # Display Results Card
                            status_color = "healthy-text" if is_healthy else "disease-text"
                            status_icon = "✅" if is_healthy else "🦠"
                            
                            html_content = f"""
                            <div class='disease-card'>
                                <h3 style='margin-bottom: 15px;'>{status_icon} Status: <span class='{status_color}'>{disease_class.replace('_', ' ')}</span></h3>
                                <div style='margin-bottom: 8px; color: #cbd5e1;'><strong>AI Confidence: {confidence * 100:.2f}%</strong></div>
                                <div style='width: 100%; background-color: #334155; border-radius: 999px; height: 8px; margin-bottom: 20px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);'>
                                    <div style='background-color: #10b981; width: {confidence*100}%; height: 8px; border-radius: 999px; box-shadow: 0 0 10px rgba(16,185,129,0.5);'></div>
                                </div>
                            """
                            
                            # Calculate and display Severity Index
                            if not is_healthy:
                                severity_index = float(confidence) # Mock severity based on confidence
                                severity_label = "High" if severity_index > 0.8 else "Moderate"
                                severity_icon = "🔥" if severity_index > 0.8 else "⚠️"
                                severity_color = "#ef4444" if severity_index > 0.8 else "#f59e0b"
                                
                                html_content += f"""
                                <div style='margin-bottom: 8px; color: #cbd5e1;'><strong>{severity_icon} Severity Index: {severity_label} ({severity_index * 100:.1f}%)</strong></div>
                                <div style='width: 100%; background-color: #334155; border-radius: 999px; height: 8px; margin-bottom: 5px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);'>
                                    <div style='background-color: {severity_color}; width: {severity_index*100}%; height: 8px; border-radius: 999px; box-shadow: 0 0 10px {severity_color};'></div>
                                </div>
                                """
                                
                            html_content += "</div>"
                            st.markdown(html_content, unsafe_allow_html=True)
                            
                            with st.expander("📊 View Probability Distribution"):
                                probs = results['all_probabilities']
                                df_probs = pd.DataFrame(list(probs.items()), columns=['Disease', 'Probability']).sort_values(by='Probability', ascending=False).head(5)
                                st.bar_chart(df_probs.set_index('Disease'))
                            
                            info = get_disease_info(disease_class)
                            
                            # Generate PDF and provide download button
                            heatmap_img = None
                            if results.get('heatmap') is not None:
                                heatmap = results['heatmap']
                                heatmap = np.uint8(255 * heatmap)
                                jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                                
                                original_img_arr = np.array(image.convert("RGB"))
                                jet = cv2.resize(jet, (original_img_arr.shape[1], original_img_arr.shape[0]))
                                superimposed_img = jet * 0.4 + original_img_arr * 0.6
                                heatmap_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                            
                            pdf_path = generate_pdf_report(image, heatmap_img, results, info)
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="📄 Download Diagnostic PDF Report",
                                    data=f,
                                    file_name=f"{uploaded_file.name}_report.pdf",
                                    mime="application/pdf",
                                    key=f"dl_{uploaded_file.name}"
                                )
                            
                            # Display Heatmap Explainability in UI
                            if heatmap_img is not None:
                                st.markdown("#### 🧠 AI Focus Area (Grad-CAM)")
                                st.image(heatmap_img, caption="Grad-CAM Heatmap", use_column_width=True)
                            
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
                                    col_act1, col_act2 = st.columns(2)
                                    with col_act1:
                                        st.info("⚖️ **Prepare Spray Tank:** Head to **⚖️ Spray & Dosage Calculator** in the navigation menu to compute exact dilution and water volume for this crop!")
                                    with col_act2:
                                        st.warning("🌦️ **Check Climate Risk:** Visit **🌦️ Disease Risk Forecaster** to see if current microclimate speeds up this pathogen!")
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
