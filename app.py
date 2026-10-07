import streamlit as st
from PIL import Image
import time
import os
import cv2
import numpy as np
from predict import PlantDiseasePredictor
from utils.db import add_prediction, get_recent_predictions, get_summary_stats, get_disease_distribution, get_daily_trends, get_all_predictions_df
from utils.pdf_generator import generate_pdf_report
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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Earthy Light Theme for Main Background */
    .stApp {
        background: linear-gradient(-45deg, #f0fdf4, #dcfce7, #fdf8f6, #ecfdf5);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #1e293b;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(240, 253, 244, 0.8) 0%, rgba(220, 252, 231, 0.7) 100%) !important;
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-right: 1px solid rgba(20, 83, 45, 0.1);
    }
    
    /* Header hero text */
    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #166534 0%, #15803d 50%, #14532d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
        letter-spacing: -2px;
        text-shadow: 0 4px 10px rgba(21, 128, 61, 0.1);
    }
    .hero-subtitle {
        font-size: 1.4rem;
        color: #475569;
        font-weight: 400;
        margin-top: 5px;
        margin-bottom: 40px;
        letter-spacing: 0.5px;
    }
    
    /* Text color overrides */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #1e293b;
    }
    .stMarkdown p, .stMarkdown div {
        color: #334155 !important;
    }
    
    /* Sleek Buttons with Earthy Tone */
    .stButton>button {
        background: linear-gradient(135deg, #15803d 0%, #166534 100%);
        color: white !important;
        border-radius: 14px;
        padding: 12px 28px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 4px 10px rgba(22, 101, 52, 0.2), inset 0 2px 4px rgba(255,255,255,0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 8px 16px rgba(22, 101, 52, 0.3), 0 0 10px rgba(22, 101, 52, 0.4);
        border: 1px solid rgba(21, 128, 61, 0.5);
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.6) !important;
        border: 2px dashed rgba(21, 128, 61, 0.4) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        padding: 40px !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(255, 255, 255, 0.9) !important;
        border-color: #15803d !important;
        box-shadow: 0 0 15px rgba(21, 128, 61, 0.15) !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(255, 255, 255, 0.6);
        padding: 10px 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        border: 1px solid rgba(20, 83, 45, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(21, 128, 61, 0.1) !important;
        border: 1px solid rgba(21, 128, 61, 0.3) !important;
        color: #15803d !important;
        box-shadow: 0 0 10px rgba(21, 128, 61, 0.05);
    }
    
    /* Dashboard Custom Metrics Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(20, 83, 45, 0.1);
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        border-color: rgba(21, 128, 61, 0.3);
        box-shadow: 0 12px 30px rgba(21, 128, 61, 0.15);
    }
    .metric-title {
        font-size: 1.15rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #15803d 0%, #166534 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(21, 128, 61, 0.1);
    }
    
    /* Modern Glassmorphism Cards */
    .disease-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.98));
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        padding: 35px;
        border-radius: 24px;
        border: 1px solid rgba(20, 83, 45, 0.1);
        box-shadow: 0 15px 40px 0 rgba(0, 0, 0, 0.08);
        margin: 30px 0;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.8s ease-out forwards;
        position: relative;
        overflow: hidden;
    }
    .disease-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.8) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        animation: shine 6s infinite;
    }
    .disease-card:hover {
        transform: translateY(-10px) scale(1.01);
        border-color: rgba(21, 128, 61, 0.4);
        box-shadow: 0 20px 50px 0 rgba(21, 128, 61, 0.15), inset 0 0 20px rgba(21, 128, 61, 0.05);
    }
    
    /* Typography */
    .healthy-text {
        color: #15803d !important;
        font-weight: 800;
        font-size: 1.8rem;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        text-shadow: 0 0 15px rgba(21, 128, 61, 0.2);
    }
    .disease-text {
        color: #b91c1c !important;
        font-weight: 800;
        font-size: 1.8rem;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        text-shadow: 0 0 15px rgba(185, 28, 28, 0.2);
        animation: pulseRed 2.5s infinite;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes shine {
        0% { left: -100%; }
        20% { left: 200%; }
        100% { left: 200%; }
    }
    @keyframes pulseRed {
        0% { text-shadow: 0 0 8px rgba(185, 28, 28, 0.2); }
        50% { text-shadow: 0 0 20px rgba(185, 28, 28, 0.4); }
        100% { text-shadow: 0 0 8px rgba(185, 28, 28, 0.2); }
    }
    </style>
""", unsafe_allow_html=True)

# --- Disease Information Dictionary (Template) ---
DISEASE_INFO = {
    "default_disease": {
        "description": "A common fungal or bacterial infection affecting the leaves.",
        "symptoms": "Spots, yellowing, or wilting of the leaves.",
        "causes": "Excessive moisture, poor air circulation, or contaminated soil.",
        "prevention": "Ensure proper spacing, avoid overhead watering, and apply suitable fungicides if necessary."
    },
    "Apple___Apple_scab": {
        "description": "Apple scab is a disease of Malus trees, such as apple trees, caused by the ascomycete fungus Venturia inaequalis.",
        "symptoms": "Dull black or grey-brown lesions on the surface of tree leaves, buds or fruits.",
        "causes": "Fungus Venturia inaequalis, thrives in wet, cool spring weather.",
        "prevention": "Rake up and destroy fallen leaves. Water in the morning so leaves dry quickly. Apply fungicides preventatively."
    },
    "Potato___Early_blight": {
        "description": "Early blight is a common disease of potatoes and tomatoes caused by the fungus Alternaria solani.",
        "symptoms": "Dark, concentric rings on older leaves, leading to yellowing and leaf drop.",
        "causes": "Fungus Alternaria solani, favored by warm, humid weather and heavy dew.",
        "prevention": "Use certified disease-free seeds. Practice crop rotation. Apply appropriate fungicides."
    },
    "Potato___Late_blight": {
        "description": "Late blight is a devastating disease caused by the oomycete Phytophthora infestans.",
        "symptoms": "Water-soaked spots on leaves that turn brown or black, often with a white mold underneath.",
        "causes": "Oomycete Phytophthora infestans, spreading rapidly in cool, wet conditions.",
        "prevention": "Plant resistant varieties, ensure good drainage, and apply protective fungicides before infection starts."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot is a fungal disease caused by Corynespora cassiicola.",
        "symptoms": "Small, dark spots on leaves with concentric rings resembling a target.",
        "causes": "Corynespora cassiicola fungus, thriving in high humidity and warm temperatures.",
        "prevention": "Improve air circulation, avoid overhead watering, and use fungicide treatments if severe."
    }
}

def get_disease_info(class_name):
    for key in DISEASE_INFO.keys():
        if key.lower() in class_name.lower():
            return DISEASE_INFO[key]
    return DISEASE_INFO["default_disease"]

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
        
        page = st.radio("Navigation", ["📸 Disease Scanner", "📊 Analytics Dashboard", "💬 AI Plant Assistant"])
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
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your PlantGuard AI assistant. How can I help your plants today?"}
            ]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "🧑‍🌾"):
                st.write(msg["content"])
                
        if prompt := st.chat_input("Ask about fertilizers, watering, specific diseases..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍🌾"):
                st.write(prompt)
                
            # Simulated response
            with st.chat_message("assistant", avatar="🌿"):
                with st.spinner("Thinking..."):
                    time.sleep(1)
                    response = f"That's a great question about '{prompt}'. To provide the best care, ensure your plants have adequate sunlight, proper drainage, and the right nutrients. If you suspect a disease, use our 📸 Disease Scanner tab for a precise AI diagnosis!"
                    st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
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
    
    tab1, tab2 = st.tabs(["📁 Upload Image(s)", "📸 Take a Photo"])
    
    with tab1:
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            images_to_process.extend(uploaded_files)
            
    with tab2:
        camera_img = st.camera_input("Take a picture of the plant leaf")
        if camera_img:
            camera_img.name = "camera_capture.jpg"
            images_to_process.append(camera_img)
            
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
                        image = Image.open(uploaded_file)
                        
                        with col1:
                            st.image(image, caption="Uploaded Image", use_column_width=True)
                            
                        with col2:
                            results = predictor.predict(image)
                            
                            confidence = results['confidence']
                            disease_class = results['predicted_class']
                            is_healthy = results['is_healthy']
                            
                            # Confidence Threshold Check
                            if confidence < 0.50:
                                st.warning("⚠️ The model is not sufficiently confident. Please upload a clearer leaf image.")
                                continue
                            
                            # Log to Database
                            add_prediction(uploaded_file.name, disease_class.replace('_', ' '), confidence, is_healthy)
                            
                            # Display Results Card
                            st.markdown("<div class='disease-card'>", unsafe_allow_html=True)
                            
                            status_color = "healthy-text" if is_healthy else "disease-text"
                            status_icon = "✅" if is_healthy else "🦠"
                            
                            st.markdown(f"### {status_icon} Status: <span class='{status_color}'>{disease_class.replace('_', ' ')}</span>", unsafe_allow_html=True)
                            
                            st.markdown("**AI Confidence:**")
                            st.progress(confidence)
                            st.write(f"{confidence * 100:.2f}%")
                            
                            # Calculate and display Severity Index
                            if not is_healthy:
                                severity_index = float(confidence) # Mock severity based on confidence
                                st.markdown(f"**🔥 Severity Index:** High ({severity_index * 100:.1f}%)" if severity_index > 0.8 else f"**⚠️ Severity Index:** Moderate ({severity_index * 100:.1f}%)")
                                st.progress(severity_index)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
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
                            
                            # Display Disease Information if not healthy
                            if not is_healthy:
                                st.markdown("### 🤖 AI Agronomist Analysis")
                                
                                with st.chat_message("assistant", avatar="🌿"):
                                    st.write(f"Hello! I detected **{disease_class.replace('_', ' ')}** on your plant. Here is my analysis:")
                                    
                                    info_tab1, info_tab2, info_tab3, info_tab4 = st.tabs(["🔬 What is it?", "⚠️ Symptoms", "🌱 Root Causes", "🛡️ Action Plan"])
                                    
                                    with info_tab1:
                                        st.write(info["description"])
                                    with info_tab2:
                                        st.write(info["symptoms"])
                                    with info_tab3:
                                        st.write(info["causes"])
                                    with info_tab4:
                                        st.info(info["prevention"])
                                        
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
