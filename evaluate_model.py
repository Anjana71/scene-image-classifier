import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIG ---
IMG_SIZE = (150,150)
BATCH_SIZE = 32
model_path = "model/cnn_intel_model.h5"  
val_dir = "seg_test/seg_test"
output_path = "outputs/confusion_matrix.png"
# -------------

# 1. Load the model
model = load_model(model_path)

# 2. Load the validation dataset
val_gen = ImageDataGenerator(rescale=1./255)
val_ds = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # IMPORTANT: keeps labels in order
)

# 3. Get predictions
y_true = val_ds.classes
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
class_names = list(val_ds.class_indices.keys())

# 4. Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 5. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig(output_path)
plt.show()
