import os
import random
import cv2


def rotate_yolo_labels_90(lines, img_width, img_height):
    rotated_lines = []

    for line in lines:
        class_idx, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.split())
        class_idx = int(class_idx)

        # Convert normalized center coordinates to pixel coordinates
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height

        # Compute new pixel coordinates for rotated bounding box
        x_new = y_center
        y_new = img_width - x_center
        width_new = height_norm * img_height
        height_new = width_norm * img_width

        # Convert new pixel coordinates back to normalized coordinates
        x_new_norm = x_new / img_height
        y_new_norm = y_new / img_width
        width_new_norm = width_new / img_height
        height_new_norm = height_new / img_width

        # Append rotated annotation to list of strings
        rotated_line = f"{class_idx} {x_new_norm:.6f} {y_new_norm:.6f} {width_new_norm:.6f} {height_new_norm:.6f}\n"
        rotated_lines.append(rotated_line)

    return rotated_lines


def augment_data():
    print('Augmenting Data...')
    count = 0
    img_dir = '/home/student/HW1/pseudo/images/train'
    labels_dir = '/home/student/HW1/pseudo/labels/train'

    img_files = sorted(os.listdir(img_dir))
    label_files = sorted(os.listdir(labels_dir))

    for img_file, label_file in zip(img_files, label_files):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        rotation_num = random.choice([1, 2, 3])
        for i in range(rotation_num):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            lines = rotate_yolo_labels_90(lines, img.shape[1], img.shape[0])

        cv2.imwrite(os.path.join(img_dir, f'rotated_frame_{count}.jpg'), img)
        with open(os.path.join(labels_dir, f'rotated_frame_{count}.txt'), 'w') as f:
            f.writelines(lines)

        print(f"Processed: {img_file} with {rotation_num * 90} degree rotation")
        count += 1

