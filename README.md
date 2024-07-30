In this repo we solve surgical tool (either tweezers, needle driver or empty hand) recognition in a leg suturing surgery.
All the paths in the repository are set to be relative paths. Just as it was when the model was trained, evaluated and used for inference

Running predict.py allows inference in a single image/frame. The annotated image will be shown. Note that the part responsible for saving the annotated image is not implemented.
Running video.py allows inference on a video. It will annotate/label all the frames of a given video. The annotated video will be saved in the given path

The final model is available here:
https://drive.google.com/file/d/1JREYLxUsEQOYf6AjV6lbhqynBK0Px6mI/view?usp=sharing

