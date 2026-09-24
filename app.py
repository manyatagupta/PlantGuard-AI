import streamlit as st
from PIL import Image
import time
import os
import cv2
import numpy as np
from predict import PlantDiseasePredictor
from utils.db import add_prediction, get_recent_predictions, get_summary_stats
from utils.pdf_generator import generate_pdf_report
import pandas as pd

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Dark-Nature Theme for Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #064e3b 0%, #020617 100%);
        color: #f1f5f9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(2, 6, 23, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Text color overrides for dark theme */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #f8fafc;
    }
    .stMarkdown p, .stMarkdown div {
        color: #cbd5e1 !important;
    }
    
    /* Sleek Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white !important;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
    }
    
    /* Modern Glassmorphism Cards */
    .disease-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin: 20px 0;
        transition: transform 0.3s ease, border-color 0.3s ease;
        animation: fadeInUp 0.6s ease-out forwards;
    }
    .disease-card:hover {
        transform: translateY(-5px);
        border-color: rgba(16, 185, 129, 0.4);
    }
    
    /* Typography */
    .healthy-text {
        color: #34d399 !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.3);
    }
    .disease-text {
        color: #f87171 !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 0 10px rgba(248, 113, 113, 0.3);
        animation: pulse 2s infinite;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { text-shadow: 0 0 5px rgba(248, 113, 113, 0.2); }
        50% { text-shadow: 0 0 20px rgba(248, 113, 113, 0.6); }
        100% { text-shadow: 0 0 5px rgba(248, 113, 113, 0.2); }
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
        
        # Display Prediction History from Database
        st.markdown("### 📊 Dashboard Metrics")
        stats = get_summary_stats()
        col1, col2 = st.columns(2)
        col1.metric("Total Scans", stats["total"])
        health_ratio = (stats["healthy"] / stats["total"] * 100) if stats["total"] > 0 else 0
        col2.metric("Health Ratio", f"{health_ratio:.1f}%")
        
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

    # Main Content
    st.title("🌿 Plant Disease Detection System")
    st.markdown("### Identify plant diseases instantly using Artificial Intelligence.")

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
                            st.image(image, caption="Uploaded Image", use_container_width=True)
                            
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
                                st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                            
                            # Display Disease Information if not healthy
                            if not is_healthy:
                                st.markdown("### 🤖 AI Agronomist Analysis")
                                
                                with st.chat_message("assistant", avatar="🌿"):
                                    st.write(f"Hello! I detected **{disease_class.replace('_', ' ')}** on your plant. Here is my analysis:")
                                    
                                    st.markdown("#### 🔬 What is it?")
                                    st.write(info["description"])
                                    
                                    st.markdown("#### ⚠️ Symptoms to look for:")
                                    st.write(info["symptoms"])
                                    
                                    st.markdown("#### 🌱 Root Causes:")
                                    st.write(info["causes"])
                                    
                                    st.markdown("#### 🛡️ Action Plan (Prevention & Care):")
                                    st.info(info["prevention"])
                                    
                    except Exception as e:
                        st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
                    st.markdown("---")
    else:
        st.info("Please upload one or more images to see the analysis results.")

if __name__ == "__main__":
    main()
