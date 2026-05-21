import os
import json

# 📁 Path to your dataset (train folder)
DATASET_PATH = "dataset/train"   # change if needed

disease_info = {}

for folder in os.listdir(DATASET_PATH):
    label = folder.strip()

    # Convert label to readable name
    display_name = label.replace("___", " ").replace("_", " ")

    # Auto generate description + solution
    disease_info[label] = {
        "description": f"{display_name} is a plant disease affecting crop health.",
        "solution": "Remove infected leaves and apply appropriate fungicide or pesticide. Maintain proper irrigation and hygiene."
    }

# Save to file
with open("disease_info.json", "w") as f:
    json.dump(disease_info, f, indent=4)

print("✅ disease_info.json generated successfully")