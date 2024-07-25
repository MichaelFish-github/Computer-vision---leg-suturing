import os
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO


def read_classes(classes_list):
    return {i: name for i, name in enumerate(classes_list)}


def predict(model, image_path, output_path, save=False):

    fig, axes = plt.subplots(1, figsize=(15, 5))
    axes = [axes]

    image = cv2.imread(image_path)
    results = model.predict(image_path, conf=0.55)
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        # Draw bounding boxes and labels on the frame
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    axes[0].set_title(os.path.basename(image_path))

    plt.tight_layout()
    plt.show()

    if save:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        raise NotImplementedError


def inference_on_video():
    model_path = './best_model.pt'
    # Local paths
    # video_path = './data/id_video_data/test2.mp4'
    # output_path = './data/id_video_data'

    # Server paths
    image_path = '/datashare/HW1/labeled_image_data/images/val/ff8c22da-output_0182.png'
    output_path = './'
    model = YOLO(model_path)
    predict(model, image_path, output_path)


if __name__ == '__main__':
    inference_on_video()
