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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
    }
    
    /* Sleek Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
    
    /* Modern Glassmorphism Cards */
    .disease-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin: 20px 0;
        transition: transform 0.3s ease;
    }
    .disease-card:hover {
        transform: scale(1.02);
    }
    
    /* Typography */
    .healthy-text {
        color: #10b981;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .disease-text {
        color: #ef4444;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
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

    st.markdown("#### Input Leaf Image")
    
    input_source = st.radio("Select Input Method:", ["Upload File(s)", "Take a Picture"], horizontal=True)
    
    images_to_process = []
    
    if input_source == "Upload File(s)":
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            images_to_process.extend(uploaded_files)
    else:
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
                            
                            st.markdown("**Confidence Score:**")
                            st.progress(confidence)
                            st.write(f"{confidence * 100:.2f}%")
                            
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
                                st.markdown("### 📖 Disease Information")
                                info_tabs = st.tabs(["Description", "Symptoms", "Causes", "Prevention/Care"])
                                
                                with info_tabs[0]:
                                    st.write(info["description"])
                                with info_tabs[1]:
                                    st.write(info["symptoms"])
                                with info_tabs[2]:
                                    st.write(info["causes"])
                                with info_tabs[3]:
                                    st.write(info["prevention"])
                                    
                    except Exception as e:
                        st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
                    st.markdown("---")
    else:
        st.info("Please upload one or more images to see the analysis results.")

if __name__ == "__main__":
    main()
