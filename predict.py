import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("model/osteonet_best.keras")

# Ask user for image path
img_path = input("Enter image path: ")

# Check if file exists
if not os.path.exists(img_path):
    print("Image not found!")
    exit()

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

grades = {
    0: "Grade 0 - Healthy",
    1: "Grade 1 - Doubtful",
    2: "Grade 2 - Mild",
    3: "Grade 3 - Moderate",
    4: "Grade 4 - Severe"
}

print("\nPrediction:")
print(grades[predicted_class])
print("Confidence:", round(np.max(prediction) * 100, 2), "%")