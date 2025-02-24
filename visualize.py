import cv2

def plot_yolo_boxes(image_path, boxes, classes):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for box in boxes:
        xyxy = box.xyxy[0]
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{classes[cls]} {conf:.2f}"
        xyxy_int = xyxy.int().tolist()
        x1, y1, x2, y2 = xyxy_int
        print(label, x1, y1, x2, y2)
        color = (0, 255, 0) if 'Crop' in label else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Draw rectangle
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)  # Draw label

    img = cv2.resize(img, (512, 512))
    return img