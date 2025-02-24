import os

import cv2
import numpy as np
import torch

from pre_process import val_transform


def make_prediction_yolo(model, image_path):
    results_new = model.predict(source=image_path)

    for result in results_new:
        boxes = result.boxes  # Boxes object for detected objects
        masks = result.masks  # Masks object for segmentation masks
        probs = result.probs

    for box in boxes:
        xyxy = box.xyxy[0].tolist()  # Bounding box coordinates (x1, y1, x2, y2)
        conf = box.conf[0].item()  # Confidence score
        cls = box.cls[0].item()  # Class ID

        print(f"Box: {xyxy}, Confidence: {conf}, Class: {cls}")

    return boxes


def make_prediction_unet(model, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(image_path.split('/')[2])
    with torch.no_grad():
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        aug = val_transform()(image=image)['image'].unsqueeze(0).to(device, dtype=torch.float32)
        outputs = model(aug)
        preds = torch.argmax(outputs, dim=1)
        # cv2.imwrite("./../test.jpg", preds.squeeze(0))
        preds = np.where(preds == 2, 255, preds)
        preds = np.where((preds > 0) & (preds <= 1), 30, preds)

    return preds.squeeze(0)
