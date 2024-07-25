import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


def load_yolo_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            label = int(parts[0])
            bbox = list(map(float, parts[1:]))
            labels.append((label, bbox))
    return labels


def analyze_label_distribution(labels_dir):
    label_counts = {}
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        labels = load_yolo_labels(label_path)
        for label, _ in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    return label_counts


def analyze_image_size_distribution(images_dir):
    image_sizes = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        image_sizes.append((width, height))
    return image_sizes


def analyze_bounding_box_distribution(labels_dir, images_dir):
    bbox_dimensions = []
    for label_file in os.listdir(labels_dir):
        try:
            label_path = os.path.join(labels_dir, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
        except AttributeError as e:
            label_path = os.path.join(labels_dir, label_file)
            image_file = label_file.replace('.txt', '.png')
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
        labels = load_yolo_labels(label_path)
        for _, bbox in labels:
            bbox_width = bbox[2] * width
            bbox_height = bbox[3] * height
            bbox_dimensions.append((bbox_width, bbox_height))
    return bbox_dimensions


def plot_distribution(data, title, xlabel, ylabel, kde=True):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=kde)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_label_distribution(label_df, title):
    label_names = ["2 - Needle driver", "1 - Tweezers", "0 - Empty"]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_df['Label'], y=label_df['Count'])
    plt.xticks(ticks=label_df['Label'], labels=label_names)
    plt.title(title)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.show()


def main():
    base_dir = '/datashare/HW1/labeled_image_data'
    dirs = ['train', 'val']

    for dir in dirs:
        print(f"Analyzing {dir} data...")

        images_dir = os.path.join(base_dir, 'images', dir)
        labels_dir = os.path.join(base_dir, 'labels', dir)

        # Label distribution
        label_distribution = analyze_label_distribution(labels_dir)
        label_df = pd.DataFrame(list(label_distribution.items()), columns=['Label', 'Count'])
        print(f"Label distribution for {dir}:\n", label_df)

        # Image size distribution
        image_sizes = analyze_image_size_distribution(images_dir)
        image_sizes_df = pd.DataFrame(image_sizes, columns=['Width', 'Height'])
        print(f"Image size distribution for {dir}:\n", image_sizes_df.describe())

        # Bounding box distribution
        bbox_dimensions = analyze_bounding_box_distribution(labels_dir, images_dir)
        bbox_df = pd.DataFrame(bbox_dimensions, columns=['BBox_Width', 'BBox_Height'])
        print(f"Bounding box size distribution for {dir}:\n", bbox_df.describe())

        # Plot distributions
        plot_label_distribution(label_df, f'Label Distribution ({dir})')
        plot_distribution(bbox_df['BBox_Width'], f'Bounding Box Width Distribution ({dir})', 'Width', 'Frequency')
        plot_distribution(bbox_df['BBox_Height'], f'Bounding Box Height Distribution ({dir})', 'Height', 'Frequency')


if __name__ == "__main__":
    main()
