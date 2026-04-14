# preprocess_dataset.py
import os
import json
import cv2
import numpy as np

dataset_dir = "dataset"
processed_dir = "processed_dataset"
os.makedirs(processed_dir, exist_ok=True)

with open(f"{dataset_dir}/metadata.json", "r") as f:
    metadata = json.load(f)

processed_metadata = []

for entry in metadata:
    top_img = cv2.imread(os.path.join(dataset_dir, entry["top_image"]))
    wrist_img = cv2.imread(os.path.join(dataset_dir, entry["wrist_image"]))

    # Resize to 224x224
    top_img = cv2.resize(top_img, (224, 224))
    wrist_img = cv2.resize(wrist_img, (224, 224))

    # Normalize
    top_img = top_img.astype(np.float32) / 255.0
    wrist_img = wrist_img.astype(np.float32) / 255.0

    step_id = entry["step_id"]

    np.save(f"{processed_dir}/top_{step_id}.npy", top_img)
    np.save(f"{processed_dir}/wrist_{step_id}.npy", wrist_img)

    processed_metadata.append(entry)

with open(f"{processed_dir}/metadata.json", "w") as f:
    json.dump(processed_metadata, f, indent=4)

print("Dataset preprocessing completed.")