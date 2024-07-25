import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def read_yolo_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_idx, x_center, y_center, width, height = map(float, line.split())
            labels.append((class_idx, x_center, y_center, width, height))
    return labels


def plot_image_with_bboxes(image, labels, ax, img_width, img_height):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for label in labels:
        class_idx, x_center, y_center, width, height = label
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


def visualize_random_pairs(base_dir, num_pairs=5):
    images_list = []
    labels_list = []

    for subset in ['train', 'val']:
        images_dir = os.path.join(base_dir, 'images', subset)
        labels_dir = os.path.join(base_dir, 'labels', subset)

        for image_filename in os.listdir(images_dir):
            if image_filename.endswith('.jpg'):
                label_filename = image_filename.replace('.jpg', '.txt')
                label_path = os.path.join(labels_dir, label_filename)

                if os.path.exists(label_path):
                    images_list.append(os.path.join(images_dir, image_filename))
                    labels_list.append(label_path)

    sampled_indices = random.sample(range(len(images_list)), num_pairs)

    fig, axes = plt.subplots(num_pairs, figsize=(15, num_pairs * 5))
    if num_pairs == 1:
        axes = [axes]

    for idx, ax in zip(sampled_indices, axes):
        image_path = images_list[idx]
        label_path = labels_list[idx]

        image = cv2.imread(image_path)
        labels = read_yolo_labels(label_path)
        img_height, img_width = image.shape[:2]

        plot_image_with_bboxes(image, labels, ax, img_width, img_height)
        ax.set_title(os.path.basename(image_path))

    plt.tight_layout()
    plt.show()


def main():
    # Usage
    base_dir = './data/labeled_and_pseudo'
    visualize_random_pairs(base_dir)

if __name__ == "__main__":
    main()