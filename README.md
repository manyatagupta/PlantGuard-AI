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
- **Explainable AI (Grad-CAM)**: The model doesn't just predict; it generates a heatmap showing exactly which part of the leaf it considers diseased!
- **📊 Analytics Dashboard**: A built-in dashboard to monitor crop health trends, view disease distribution charts, and export historical scanning data to CSV.
- **Premium UI/UX**: A sleek, responsive "Dark-Nature" theme with glassmorphism effects and micro-animations for an ultra-modern feel.
- **🤖 AI Agronomist**: A simulated chat interface that breaks down the disease description, symptoms, and prevention strategies in a conversational format.
- **📄 Professional PDF Reports**: Generate and download beautifully styled PDF diagnostic reports featuring the original image, Grad-CAM heatmap, and confidence distribution charts.
- **Batch Image Upload**: Upload multiple `.jpg`, `.jpeg`, or `.png` images of plant leaves simultaneously.
- **Persistent Prediction History**: A local SQLite database tracks all your past predictions securely so you never lose them across sessions.

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
