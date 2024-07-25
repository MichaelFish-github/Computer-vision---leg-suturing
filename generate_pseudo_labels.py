import os
import random
import cv2
import numpy as np


def gamma_correction(img, gamma=0.6):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def generate_pseudo_labels(model, video_path, output_labels_dir, output_images_dir, processed_frames,
                           confidence_threshold=0.6, frame_frequency=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    train_images_dir = os.path.join(output_images_dir, 'train')
    val_images_dir = os.path.join(output_images_dir, 'val')
    train_labels_dir = os.path.join(output_labels_dir, 'train')
    val_labels_dir = os.path.join(output_labels_dir, 'val')

    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    frame_indices = list(range(frame_count))

    for i in frame_indices:
        if i in processed_frames or i % frame_frequency != 0:
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # frame = gamma_correction(frame)
        results = model.predict(frame, iou=0.6)
        frame_pseudo_labels = []

        height, width = frame.shape[:2]
        for result in results:
            xywh = result.boxes.xywh
            confs = result.boxes.conf
            classes = result.boxes.cls
            for box, conf, cls in zip(xywh, confs, classes):
                if conf >= confidence_threshold:
                    x_center, y_center, box_width, box_height = box
                    # Normalize the coordinates
                    x_center /= width
                    y_center /= height
                    box_width /= width
                    box_height /= height
                    frame_pseudo_labels.append((int(cls.item()), x_center, y_center, box_width, box_height))

        if frame_pseudo_labels:
            if random.random() < 0.95:
                image_dir = train_images_dir
                label_dir = train_labels_dir
            else:
                image_dir = val_images_dir
                label_dir = val_labels_dir

            video_name = os.path.basename(video_path).split('.')[0]
            frame_filename = f"{video_name}_{i:06d}.jpg"
            frame_path = os.path.join(image_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_filename} with {conf} pseudo labels.")

            labels_filename = f"{video_name}_{i:06d}.txt"
            labels_path = os.path.join(label_dir, labels_filename)
            with open(labels_path, 'w') as file:
                for label in frame_pseudo_labels:
                    cls, x_center, y_center, box_width, box_height = label
                    file.write(f"{cls} {x_center} {y_center} {box_width} {box_height}\n")

            processed_frames.add(i)

    cap.release()
    return processed_frames
