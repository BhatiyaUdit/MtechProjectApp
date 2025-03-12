import os

import cv2
import streamlit as st
import torch
from ultralytics import YOLO

from make_prediction import make_prediction_yolo, make_prediction_unet
from unet_model import UNet
from visualize import plot_yolo_boxes, plot_updated_boxes
from merge_predictions import adjust_yolo_confidence

yolo_model_path = "./models/best after 73 epochs.pt"
unet_model_path = "./models/checkpoint_epoch_20.pth"
predictions_dir = "./predictions"
uploads_dir = "./uploads"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title="Weed Detection App")

if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)


yolo_model = YOLO(yolo_model_path)
unet_checkpoint = torch.load(unet_model_path, map_location=torch.device('cpu'))
print(unet_checkpoint.keys())
unet_model = UNet().to(device)
unet_model.load_state_dict(unet_checkpoint['model_state_dict'])

print(yolo_model.info())
print(unet_model)

if __name__ == "__main__":
    st.title("Crop Or Weed Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    print(uploaded_file)
    if uploaded_file is not None:
        temp_image_path = "./uploads/" + uploaded_file.name
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = cv2.imread(temp_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        unet_predictions = make_prediction_unet(unet_model, temp_image_path)
        output_path = os.path.join(predictions_dir, "mask" + uploaded_file.name)
        cv2.imwrite(output_path, unet_predictions)

        boxes = make_prediction_yolo(yolo_model, temp_image_path)
        predicted_image = plot_yolo_boxes(temp_image_path, boxes, yolo_model.names)
        output_path = os.path.join(predictions_dir, "yolo" + uploaded_file.name)
        cv2.imwrite(output_path, predicted_image)

        with st.expander("Show/Hide Intermediate results"):
            st.image(unet_predictions, caption='Predicted Mask', use_container_width=False, channels='GRAY')
            st.image(predicted_image, caption='Predicted Boxes', use_container_width=True)

        updated_preds = adjust_yolo_confidence(boxes, unet_predictions, original_size=[image.shape[1], image.shape[0]])
        merged_image = plot_updated_boxes(temp_image_path, updated_preds, yolo_model.names, original_size=[image.shape[1], image.shape[0]])
        st.image(merged_image, caption='Boosted Prediction', use_container_width=True)
