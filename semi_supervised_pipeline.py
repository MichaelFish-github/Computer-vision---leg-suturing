import os
import shutil
import yaml
from ultralytics import YOLO
from generate_pseudo_labels import generate_pseudo_labels
import torch
from video import inference_on_video
import rotation_augmentation
print("Finished importing")


def read_classes(classes_list):
    return {i: name for i, name in enumerate(classes_list)}


def load_model(workingDir, last_model_name, max_attempts=10):
    base_path_detect = os.path.join(workingDir, 'runs/detect', last_model_name, 'weights', 'best.pt')

    for i in range(max_attempts, 0, -1):
        print(f"Trying to load model {last_model_name}{i}")
        base_path_train = os.path.join(workingDir, f'runs/detect/{last_model_name}{i}/weights/best.pt')
        if os.path.exists(base_path_train):
            return YOLO(base_path_train)

    if os.path.exists(base_path_detect):
        return YOLO(base_path_detect)

    raise FileNotFoundError(f"Model not found in any of the checked paths up to {max_attempts} attempts")


def clear_pseudo_dirs(pseudo_labels_dir, pseudo_images_dir):
    os.remove(os.path.join(pseudo_images_dir, "val.cache"))
    os.remove(os.path.join(pseudo_images_dir, "train.cache"))
    os.remove(os.path.join(pseudo_labels_dir, "val.cache"))
    os.remove(os.path.join(pseudo_labels_dir, "train.cache"))
    os.remove("demofile.txt")
    for file in os.listdir(pseudo_labels_dir):
        if os.path.exists(os.path.join(pseudo_labels_dir, file)) and os.path.isdir(
                os.path.join(pseudo_labels_dir, file)):
            shutil.rmtree(os.path.join(pseudo_labels_dir, file))
            print(
                f"Directory '{os.path.join(pseudo_images_dir, file)}' and all its contents have been removed.")
    # Create the train and val directories in pseudo labels dir
    if not os.path.exists(os.path.join(pseudo_labels_dir, 'train')):
        os.makedirs(os.path.join(pseudo_labels_dir, 'train'))
    if not os.path.exists(os.path.join(pseudo_labels_dir, 'val')):
        os.makedirs(os.path.join(pseudo_labels_dir, 'val'))

    for file in os.listdir(pseudo_images_dir):
        if os.path.exists(os.path.join(pseudo_images_dir, file)) and os.path.isdir(
                os.path.join(pseudo_images_dir, file)):
            shutil.rmtree(os.path.join(pseudo_images_dir, file))
            print(
                f"Directory '{os.path.join(pseudo_images_dir, file)}' and all its contents have been removed.")
    # Create the train and val directories in pseudo images dir
    if not os.path.exists(os.path.join(pseudo_images_dir, 'train')):
        os.makedirs(os.path.join(pseudo_images_dir, 'train'))
    if not os.path.exists(os.path.join(pseudo_images_dir, 'val')):
        os.makedirs(os.path.join(pseudo_images_dir, 'val'))


def train_yolo(model, data_dir, working_dir, num_classes=3, epochs=5, model_name='initial_training'):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    if not os.access(data_dir, os.R_OK):
        raise PermissionError(f"Read permission denied for directory {data_dir}.")

    # Initialize the YOLOv8 model
    classes = read_classes(['Empty', 'Tweezers', 'Needle_driver'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_config = {
        'path': data_dir,
        'train': os.path.join(data_dir, 'images/train'),
        'val': os.path.join(data_dir, 'images/val'),
        'nc': num_classes,
        'names': classes
    }

    # Save the data configuration to a YAML file
    data_config_path = os.path.join(working_dir, 'data_config.yaml')
    with open(data_config_path, 'w+') as f:
        yaml.dump(data_config, f)

    # Train the model
    model.train(data=data_config_path, epochs=epochs, imgsz=640, name=model_name, device=device, task='detect', hsv_h=0, hsv_s=0, hsv_v=0, fliplr=0)


def main():
    # Hyperparameters
    initial_epochs = 40
    iterations_per_video = 1
    pseudo_epochs = 25
    confidence_threshold = 0.5
    frame_frequency = 30

    # Local paths
    # Define the directories and classes
    # dataDir = 'C:/Users/micah/OneDrive - Technion/Technion/8th Semester/Computer Vision in Operation room/HW1/data/labeled_image_data'
    # workingDir = 'C:/Users/micah/OneDrive - Technion/Technion/8th Semester/Computer Vision in Operation room/HW1'
    # pseudoDir = 'C:/Users/micah/OneDrive - Technion/Technion/8th Semester/Computer Vision in Operation room/HW1/data/pseudo'
    # id_videos_dir = 'C:/Users/micah/OneDrive - Technion/Technion/8th Semester/Computer Vision in Operation room/HW1/data/id_video_data'

    # Server paths
    dataDir = '/home/student/HW1/labeled_image_data'
    workingDir = '/home/student/HW1'
    pseudoDir = '/home/student/HW1/labeled_and_pseudo'
    id_videos_dir = '/datashare/HW1/id_video_data'
    ood_videos_dir = '/datashare/HW1/ood_video_data'

    pseudo_labels_dir = os.path.join(pseudoDir, 'labels')
    pseudo_images_dir = os.path.join(pseudoDir, 'images')
    if not os.path.exists(pseudoDir):
        os.makedirs(pseudoDir)

    if not os.path.exists(pseudo_labels_dir):
        os.makedirs(pseudo_labels_dir)

    if not os.path.exists(pseudo_images_dir):
        os.makedirs(pseudo_images_dir)


    # Initial training
    model = YOLO('yolov8m.pt')
    last_model_name = 'attempt18'
    print("Starting initial training")
    train_yolo(model, dataDir, workingDir, epochs=initial_epochs, model_name=last_model_name)


    # Pseudo label generation and training
    model = load_model(workingDir, last_model_name)
    for vid_idx, video_file in enumerate(os.listdir(id_videos_dir)):
        if video_file != '4_2_24_B_2.mp4':
            continue
        if video_file.endswith('.mp4'):
            processed_frames = set()
            video_path = os.path.join(id_videos_dir, video_file)
            for i in range(iterations_per_video):
                current_model_name = f'attempt17_vid_{vid_idx + 1}_iter_{i + 1}'
                if last_model_name != current_model_name:
                    model = load_model(workingDir, last_model_name)
                processed_frames = generate_pseudo_labels(model, video_path, pseudo_labels_dir, pseudo_images_dir,
                                                          processed_frames, confidence_threshold=confidence_threshold, frame_frequency=frame_frequency)
                # augment.augment_data()
                if os.listdir(os.path.join(pseudo_labels_dir, 'val')):
                    train_yolo(model, pseudoDir, workingDir, epochs=pseudo_epochs, model_name=current_model_name)
                    last_model_name = current_model_name
                    # clear_pseudo_dirs(pseudo_labels_dir, pseudo_images_dir)
                else:
                    print(f"No pseudo labels generated for video {video_file}")
                    break

    # inference_on_video(model, os.path.join(ood_videos_dir, 'surg_1.mp4'), os.path.join(workingDir, 'annotated_videos'),
    #                    'surg_1_annotated')
    # inference_on_video(model, os.path.join(ood_videos_dir, '4_2_24_A_1.mp4'),
    #                    os.path.join(workingDir, 'annotated_videos'), '4_2_24_A_1_annotated')


if __name__ == '__main__':
    main()
