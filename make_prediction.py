import cv2
import numpy as np
import torch

from pre_process import val_transform


def make_prediction_yolo(model, image_path):
    results_new = model.predict(source=image_path, iou=0.4)

    for result in results_new:
        boxes = result.boxes
    print(boxes)

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()

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
        preds = np.where(preds == 2, 255, preds)
        preds = np.where((preds > 0) & (preds <= 1), 255, preds)

    return preds.squeeze(0)
