# import os
# os.environ["PAFY_BACKEND"] = "internal"

import torch
import numpy as np
import cv2
import pafy
from time import time


class ObjectDetection:
    """
    Class implements YOLOv5 model to make inferences on a YouTube video using OpenCV.
    """

    def __init__(self, url, out_file):
        """
        Initializes the class with a YouTube URL and output file.

        Parameters:
            - url: YouTube URL on which predictions are made.
            - out_file: A valid output file name.
        """
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def get_video_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame for predictions.

        Returns:
            - OpenCV2 video capture object with the lowest quality frame available for video.
        """
        play = pafy.new(self._URL).streams[-1]
        assert play is not None
        return cv2.VideoCapture(play.url)

    def load_model(self):
        """
        Loads YOLOv5 model from PyTorch Hub.

        Returns:
            - Trained PyTorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input and scores the frame using YOLOv5 model.

        Parameters:
            - frame: Input frame in numpy/list/tuple format.

        Returns:
            - Labels and coordinates of objects detected by the model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.

        Parameters:
            - x: Numeric label

        Returns:
            - Corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input and plots bounding boxes and labels on the frame.

        Parameters:
            - results: Contains labels and coordinates predicted by the model on the given frame.
            - frame: Frame which has been scored.

        Returns:
            - Frame with bounding boxes and labels plotted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = map(int, (row[0] * x_shape, row[1] * y_shape, row[2] * x_shape, row[3] * y_shape))
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when the class is executed. It runs the loop to read the video frame by frame
        and writes the output into a new file.

        Returns:
            - None
        """
        player = self.get_video_from_url()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))

        while True:
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            print(f"Frames Per Second: {fps}")
            out.write(frame)

        player.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create a new object and execute.
    detection = ObjectDetection("https://www.youtube.com/watch?v=EXUQnLyc3yE", "video2.avi")
    detection()
