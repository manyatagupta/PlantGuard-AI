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
