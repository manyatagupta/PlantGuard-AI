# PlantGuard AI - Plant Disease Detection System 🌿

![PlantGuard AI Header](https://via.placeholder.com/1200x300.png?text=PlantGuard+AI+-+Plant+Disease+Detection+System)

## 📌 Project Overview
PlantGuard AI is an intelligent web application designed to help farmers, gardeners, and agriculturists identify plant diseases instantly. By simply uploading an image of a plant leaf, the system utilizes a trained Deep Learning model to determine if the plant is healthy or suffering from a specific disease, while also providing actionable care and prevention suggestions.

**⚠️ Disclaimer:** This system is built for educational and assistive purposes and should not replace professional agricultural advice.

## ✨ Features
- **Explainable AI (Grad-CAM)**: The model doesn't just predict; it generates a heatmap showing exactly which part of the leaf it considers diseased!
- **Batch Image Upload**: Upload multiple `.jpg`, `.jpeg`, or `.png` images of plant leaves simultaneously via a drag-and-drop interface.
- **Accurate Disease Prediction**: Uses a fine-tuned Convolutional Neural Network (CNN) powered by Transfer Learning (MobileNetV2) for high accuracy.
- **Downloadable PDF Reports**: Generate and download a professional PDF diagnostic report containing the original image, heatmap, and care suggestions.
- **Persistent Prediction History**: A local SQLite database tracks all your past predictions securely so you never lose them across sessions.
- **Confidence Visualization**: View the model's confidence score via an intuitive progress bar.
- **Detailed Disease Information**: Get comprehensive insights into the detected disease, including symptoms, causes, and prevention methods.
- **Modern UI**: A responsive, clean, and user-friendly interface built with Streamlit.

## 📸 Demo Screenshots
*(Add your screenshots here once you run the application)*
- **Home & Upload Screen**: `assets/demo_home.png`
- **Prediction Result**: `assets/demo_result.png`
- **Disease Info Tabs**: `assets/demo_info.png`

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
- **Base Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Top Layers**: Global Average Pooling -> Dropout (0.2) -> Dense (Softmax)
- **Input Size**: 224 x 224 x 3
- **Optimizer**: Adam (Learning Rate dynamically reduced via ReduceLROnPlateau)
- **Loss Function**: Sparse Categorical Crossentropy

**Training Process:**
1. **Phase 1 (Feature Extraction):** We freeze the base model and train only the custom top layers using balanced class weights.
2. **Phase 2 (Fine-Tuning):** We unfreeze the top 100 layers of MobileNetV2 and train the entire stack with a lower learning rate to achieve maximum accuracy.

## 💻 Technologies Used
- **Python 3.x**
- **TensorFlow / Keras** (Deep Learning)
- **Streamlit** (Web Application Framework)
- **OpenCV & Pillow** (Image Processing)
- **Scikit-learn** (Evaluation Metrics)
- **Matplotlib & Seaborn** (Data Visualization)

## 🛠️ Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/plantguard-ai.git
   cd plantguard-ai
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

## 📊 Model Evaluation & TensorBoard
During training, the model evaluates itself on a 20% validation split. The `train.py` script automatically generates:
- Accuracy, Precision, Recall, and F1-Score reports.
- A Confusion Matrix (`assets/confusion_matrix.png`).
- Training vs. Validation Loss & Accuracy graphs (`assets/training_history.png`).
- **TensorBoard Logs**: View live training metrics by running:
  ```bash
  tensorboard --logdir logs/fit
  ```

## 🔮 Future Improvements
- **Multi-leaf detection**: Implement object detection (e.g., YOLO) to identify multiple diseased spots on a single plant.
- **Wider Dataset**: Include more plant species and diseases.
- **Mobile App**: Port the model using TensorFlow Lite for an offline mobile application.
- **User Feedback Loop**: Allow users to confirm or correct predictions to continuously improve the model.
