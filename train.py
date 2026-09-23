import os
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import datetime
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
import matplotlib.pyplot as plt
# Configuration
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "models/plant_disease_model.keras"
ASSETS_DIR = "assets"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 15
LEARNING_RATE = 1e-4

def create_directories():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: '{DATASET_DIR}' not found. Please place your image data in this directory.")

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(ASSETS_DIR, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'confusion_matrix.png'))
    plt.close()

def main():
    create_directories()
    
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print(f"Error: Dataset directory '{DATASET_DIR}' is empty or does not exist.")
        print("Please download a dataset (e.g., PlantVillage), extract it, and place class folders directly inside 'dataset/'.")
        return

    # Data Loading and Augmentation
    print("Loading dataset...")
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Save class names for inference
    with open('models/class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # Data Augmentation Layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.2),
    ])

    # Preprocessing is built into EfficientNetV2B0 when include_preprocessing=True
    # Build Model Architecture
    print("Building model...")
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    
    # Load EfficientNetV2B0 without the top classification layer
    base_model = EfficientNetV2B0(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet', include_preprocessing=True)
    base_model.trainable = False # Freeze base model initially
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)

    # Calculate class weights for imbalanced datasets
    print("Calculating class weights...")
    train_labels = []
    for images, labels in train_dataset.unbatch():
        train_labels.append(labels.numpy())
    
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Learning Rate Schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )

    # Compile Model
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # TensorBoard setup
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Callbacks
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tensorboard_callback
    ]

    # Training - Phase 1: Feature Extraction
    print("Starting Training Phase 1: Feature Extraction (Top layers only)...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # Training - Phase 2: Fine Tuning
    print("Starting Training Phase 2: Fine-tuning...")
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a lower learning rate schedule
    lr_schedule_fine = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE / 10,
        first_decay_steps=500,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )
    
    model.compile(optimizer=Adam(learning_rate=lr_schedule_fine),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_fine = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS // 2, # Fewer epochs for fine-tuning
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    print("Training complete. Saving history plot...")
    # For plot, we can just use the first history, or combine them (for simplicity we use the first here)
    plot_training_history(history)

    # Evaluation
    print("Evaluating model...")
    
    # Get true labels and predictions
    y_true = []
    y_pred = []
    
    for images, labels in val_dataset:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("Generating Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Plots saved to {ASSETS_DIR}/")

if __name__ == "__main__":
    main()
