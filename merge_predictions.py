import cv2
import numpy as np


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area) if box1_area + box2_area > 0 else 0
    return iou


def adjust_yolo_confidence(yolo_predictions, segmentation_mask, iou_threshold=0.3, confidence_boost=0.2, confidence_threshold=0.4, original_size=[1088, 1920]):
    updated_predictions = []
    mask_size = [512, 512]

    scale_x = original_size[1] / mask_size[1]  # width scaling factor
    scale_y = original_size[0] / mask_size[0]  # height scaling factor

    # print(scale_x, scale_y)

    for prediction in yolo_predictions:
        # print("Predicted: ", prediction)
        xyxy = prediction.xyxy[0]
        confidence = prediction.conf[0].item()
        class_id = int(prediction.cls[0].item())

        xyxy_int = xyxy.int().tolist()
        x1, y1, x2, y2 = xyxy_int

        x1 = int(x1 / scale_x)
        y1 = int(y1 / scale_y)
        x2 = int(x2 / scale_x)
        y2 = int(y2 / scale_y)

        # print(x1, y1, x2, y2, confidence, class_id)

        box_mask = np.zeros(segmentation_mask.shape, dtype=np.int64)
        box_mask[y1:y2, x1:x2] = 255

        intersection = cv2.bitwise_and(box_mask, segmentation_mask)
        intersection_area = np.sum(intersection) / 255

        bbox_area = (x2 - x1) * (y2 - y1)

        iou = intersection_area / float(bbox_area) if bbox_area > 0 else 0

        print("IOU: ", iou, " class_id: ", class_id, "confidence" , confidence)
        # confidence_threshold = 0.3
        # iou_threshold = 0.3

        if class_id == 1:
            if confidence >= confidence_threshold and iou >= iou_threshold:
                confidence += confidence_boost
                confidence = min(confidence, 1.0)
            elif confidence < confidence_threshold and iou >= iou_threshold:
                class_id = 0
                print("Changing class to CROP")
            elif confidence > confidence_threshold and iou <= iou_threshold:
                # class_id = 0
                print("KEEP class to WEED")
            else:
                continue

        # Adjust confidence for CROP class predictions
        elif class_id == 0:
            # Increase confidence if there is overlap with the segmentation mask
            if intersection_area > 0:  # There is some plant detected in the bounding box
                confidence += confidence_boost  # Boost confidence for CROP
                confidence = min(confidence, 1.0)  # Ensure confidence does not exceed 1.0

        updated_predictions.append((x1, y1, x2, y2, confidence, class_id))

    # refined_predictions = []
    # final_confidence_threshold = 0.55
    # for x1, y1, x2, y2, confidence, class_id in updated_predictions:
    #     if confidence >= final_confidence_threshold:
    #         refined_predictions.append((x1, y1, x2, y2, confidence, class_id))
    #
    # return refined_predictions

    return updated_predictions


# if __name__ == '__main__':
#     yolo_model_path = "./models/best_after 44 epoch in middle.pt"
#     unet_model_path = "./models/checkpoint_epoch_20.pth"
#
#     yolo_model = YOLO(yolo_model_path)
#     unet_checkpoint = torch.load(unet_model_path, map_location=torch.device('cpu'))
#     print(unet_checkpoint.keys())
#     unet_model = UNet()
#     unet_model.load_state_dict(unet_checkpoint['model_state_dict'])
#
#     file_name = 'ave-0125-0020.jpg'
#     temp_image_path = "./uploads/"+file_name
#
#     unet_predictions = make_prediction_unet(unet_model, temp_image_path)
#
#     # print("Unet predictions:" , unet_predictions)
#
#     boxes = make_prediction_yolo(yolo_model, temp_image_path)
#
#     # print("YOLO predictions:" , boxes)
#
#     updated_preds = adjust_yolo_confidence(boxes, unet_predictions)
#
#     print(updated_preds)
#     classes = yolo_model.names
#     predictions_dir = "./predictions/Test"
#
#     image = cv2.imread(temp_image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (512, 512))
#
#     for box in updated_preds:
#         (x1, y1, x2, y2, confidence, class_id) = box
#         label = f"{classes[class_id]} {confidence:.2f}"
#
#         color = (0, 255, 0) if 'Crop' in label else (255, 0, 0)
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Draw rectangle
#         cv2.putText(image, label, (x1, y1 - int(0.3 * 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)  # Draw label
#
#     output_path = os.path.join(predictions_dir, "yolo" + file_name)
#     cv2.imwrite(output_path, image)