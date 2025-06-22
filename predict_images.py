import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

model = load_model("model/cnn_intel_model.h5")
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
pred_dir = "seg_pred/seg_pred"

output_dir = "outputs/predictions"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(pred_dir):
    img_path = os.path.join(pred_dir, fname)
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Save or display result
    print(f"{fname}: {predicted_class}")

    # Optional: save result image with label
    img.save(f"{output_dir}/{predicted_class}_{fname}")
