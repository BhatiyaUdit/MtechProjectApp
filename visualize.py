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
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)  # Draw label
    return img


def plot_updated_boxes(image_path, boxes, classes, original_size = [1088, 1920] ):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_size = [512, 512]

    scale_x = original_size[1] / mask_size[1]  # width scaling factor
    scale_y = original_size[0] / mask_size[0]  # height scaling factor

    for box in boxes:
        (x1, y1, x2, y2, confidence, class_id) = box

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        label = f"{classes[class_id]} {confidence:.2f}"

        color = (0, 255, 0) if 'Crop' in label else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)  # Draw label

    return image