import torch
import numpy as np
import cv2
from time import time

class ObjectDetection:
    def __init__(self, capture_index):
        """
        Initializes the ObjectDetection class.

        Parameters:
            - capture_index: Index of the camera to capture video from.
        """
        self.capture_index = capture_index
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)

    def load_model(self):
        """
        Loads the YOLOv5 model from PyTorch Hub.

        Returns:
            - Trained PyTorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def get_video_capture(self):
        """
        Creates a new video capture object to extract video frame by frame.

        Returns:
            - OpenCV video capture object.
        """
        return cv2.VideoCapture(self.capture_index)

    def score_frame(self, frame):
        """
        Takes a single frame as input and scores the frame using YOLOv5 model.

        Parameters:
            - frame: Input frame in numpy/list/tuple format.

        Returns:
            - Labels and Coordinates of objects detected by the model in the frame.
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

    def process_video(self):
        """
        Runs a loop to read the video frame by frame and displays the result.

        Returns:
            - None
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            start_time = time()

            ret, frame = cap.read()
            if not ret:
                break

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create a new object and execute.
    detector = ObjectDetection(capture_index=0)
    detector.process_video()
