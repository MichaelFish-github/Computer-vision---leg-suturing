import os
import cv2
from ultralytics import YOLO


def read_classes(classes_list):
    return {i: name for i, name in enumerate(classes_list)}


def annotate_video(model, video_path, output_path, video_name):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    classes = read_classes(['Empty', 'Tweezers', 'Needle_driver'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_video_path = os.path.join(output_path, f'{video_name}.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Shuffle frame indices
    frame_indices = list(range(frame_count))

    for i in frame_indices:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.55)
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            # Draw bounding boxes and labels on the frame
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f'Annotated video saved to {output_video_path}')


def inference_on_video():
    model_path = './best_model.pt'
    # Local paths
    # video_path = './data/id_video_data/test2.mp4'
    # output_path = './data/id_video_data'

    # Server paths
    video_path = '/datashare/HW1/ood_video_data/surg_1.mp4'
    output_path = '/home/student/HW1/annotated_videos'
    video_name = 'surg_1_annotated'
    model = YOLO(model_path)
    annotate_video(model, video_path, output_path, video_name)


if __name__ == '__main__':
    inference_on_video()
