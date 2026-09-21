import streamlit as st
from PIL import Image
import time
import os
import cv2
import numpy as np
from predict import PlantDiseasePredictor
from utils.db import add_prediction, get_recent_predictions
from utils.pdf_generator import generate_pdf_report

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
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        border-color: #1b5e20;
        color: white;
    }
    .disease-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .healthy-text {
        color: #2e7d32;
        font-weight: bold;
    }
    .disease-text {
        color: #c62828;
        font-weight: bold;
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
        st.markdown("### 🕒 Recent History (Database)")
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

    st.markdown("#### Upload Leaf Image(s)")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.markdown("---")
        st.markdown("#### Batch Analysis Results")
        
        if st.button("Analyze All Leaves", use_container_width=True):
            with st.spinner("Analyzing images... Please wait."):
                time.sleep(1) # Simulate slight processing delay for UX
                
                for uploaded_file in uploaded_files:
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
