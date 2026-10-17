# PlantGuard AI - Plant Disease Detection System 🌿

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

![PlantGuard AI Header](https://via.placeholder.com/1200x300.png?text=PlantGuard+AI+-+Plant+Disease+Detection+System)

## 📌 Project Overview
PlantGuard AI is an intelligent, full-stack web application designed to help farmers, gardeners, and agriculturists identify plant diseases instantly. By simply uploading an image of a plant leaf, the system utilizes a fine-tuned **EfficientNetV2B0** Deep Learning model to determine if the plant is healthy or suffering from a specific disease, while also providing actionable care and prevention suggestions via an AI Agronomist interface.

**⚠️ Disclaimer:** This system is built for educational and assistive purposes and should not replace professional agricultural advice.

## ✨ Key Features
- **🔬 Quantitative Computer Vision Leaf Damage Segmentation**: Measures exact percentage of damaged/necrotic foliar surface area (% Damaged Area) using advanced color space segmentation and assigns phytopathological severity grades (Grade 0 to Grade 4).
- **🌱 Dual-Scan Post-Treatment Recovery & Healing Tracker**: Compares baseline pre-treatment vs follow-up post-treatment leaf scans to compute lesion reduction rate, foliar reclamation percentage, and clinical healing status.
- **🗺️ 3D Regional Outbreak Radar (PyDeck Geospatial)**: Interactive 3D geospatial radar mapping surrounding farm weather stations, telemetry, and regional pathogen outbreak threats.
- **💰 Commercial Treatment Pack & ROI Budget Estimator**: Automatically calculates required commercial containers/packs, application labor costs, potential crop loss avoided, and Return on Investment (ROI).
- **🧪 1-Click Fast-Test Specimen Library**: Pre-loaded high-resolution demonstration leaves (Tomato Early Blight, Apple Scab, Healthy Leaf) for instant zero-upload evaluations.
- **🧠 Explainable AI (Grad-CAM Heatmaps)**: Deep-learning feature activation localization using Keras 3 EfficientNetV2B0 top convolutional activations to show growers the exact leaf regions driving classification.
- **🔊 Hands-Free Audio Diagnostic Briefing**: Integrated browser speech synthesis allowing field growers and farmers wearing work gloves to listen to diagnostic summaries hands-free.
- **📅 Seasonal Disease & Spray Calendar**: Stage-by-stage phenological protection timelines (from dormant to harvest) for Apple, Tomato, Potato, Corn, and Bell Pepper.
- **🌦️ Meteorological Disease Outbreak Forecaster**: Live Open-Meteo satellite weather queries and epidemiological pathogen vulnerability scoring based on temperature, continuous leaf wetness, and humidity.
- **⚖️ Precision Spray & Dosage Calculator**: Exact active concentrate amount, carrier water volume, WALES tank mixing protocols, PHI withholding periods, and REI safety intervals.
- **💬 Dual-Mode AI Agronomist**: Works completely offline out-of-the-box with the embedded localized Agronomy Brain, or seamlessly connects with Google Gemini AI Studio for conversational dialogues.
- **📄 Enhanced 3-Panel PDF Diagnostic Reports**: Beautifully styled multi-page export featuring the original specimen, Grad-CAM attention map, and CV lesion segmentation overlay alongside tailored prescription regimens.
- **📊 Analytics Dashboard**: Monitor historical crop health trends, average foliar lesion surface area, and export data to CSV.
- **Persistent Prediction History**: A local SQLite database tracks all your past predictions securely with one-click database management.

## 📸 Demo Screenshots
*(Add your screenshots here once you run the application)*
- **Home & Upload Screen**: `assets/demo_home.png`
- **Analytics Dashboard**: `assets/demo_analytics.png`
- **Prediction Result & PDF**: `assets/demo_result.png`

## 🗂 Dataset
You can use the provided script to automatically download a dataset sample:
```bash
python download_dataset.py
```

Alternatively, you can use **[PlantVillage](https://data.mendeley.com/datasets/tywbtsjrjv/1)**. 
To train the model yourself, organize the dataset such that each disease class has its own folder inside the `dataset/` directory.

```
plantguard-ai/
└── dataset/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Apple___healthy/
    └── ...
```

## 🧠 Model Architecture & Training
- **Base Model**: EfficientNetV2B0 (Pre-trained on ImageNet)
- **Top Layers**: Global Average Pooling -> Dropout (0.3) -> Dense (Softmax)
- **Input Size**: 224 x 224 x 3
- **Optimizer**: Adam with `CosineDecayRestarts` Learning Rate Schedule
- **Data Augmentation**: RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast.
- **Loss Function**: Sparse Categorical Crossentropy

**Training Process:**
1. **Phase 1 (Feature Extraction):** We freeze the base model and train only the custom top layers using balanced class weights and heavy data augmentation to prevent overfitting.
2. **Phase 2 (Fine-Tuning):** We unfreeze the top 50 layers of EfficientNetV2B0 and train the entire stack with a lower learning rate to achieve maximum accuracy.

## 💻 Technologies Used
- **Python 3.x**
- **TensorFlow / Keras** (Deep Learning)
- **Streamlit** (Web Application Framework)
- **OpenCV & Pillow** (Image Processing)
- **SQLite3 & Pandas** (Database & Analytics)
- **Matplotlib & fpdf** (PDF Generation & Charting)

## 🛠️ Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manyatagupta/PlantGuard-AI.git
   cd PlantGuard-AI
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏋️ How to Train the Model
If you want to train the model from scratch:
1. Ensure your dataset is placed in the `dataset/` directory.
2. Run the training script:
   ```bash
   python train.py
   ```
3. The script will apply data augmentation, train the model, save the best weights to `models/plant_disease_model.keras`, and save `models/class_names.txt`.
4. Training history and confusion matrix plots will be saved in the `assets/` directory.

## 🚀 How to Run the Web Application
Once the model is trained (or if you already have the `.keras` file in the `models/` directory):

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`.

## 🔮 Future Improvements
- **Multi-leaf detection**: Implement object detection (e.g., YOLO) to identify multiple diseased spots on a single plant.
- **Wider Dataset**: Include more plant species and diseases.
- **Mobile App**: Port the model using TensorFlow Lite for an offline mobile application.
<!-- formatting 1 -->
<!-- formatting 2 -->
<!-- formatting 3 -->
<!-- formatting 4 -->
<!-- formatting 5 -->
<!-- formatting 6 -->
<!-- formatting 7 -->
<!-- ui polish spacing 1 -->
<!-- ui polish spacing 2 -->
<!-- ui polish spacing 3 -->
<!-- ui polish spacing 4 -->
<!-- ui polish spacing 5 -->
<!-- ui polish spacing 6 -->
