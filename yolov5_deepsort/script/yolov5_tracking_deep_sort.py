import cv2
import numpy as np
import time
import torch
# pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort

class YoloDetector:
    def __init__(self, model_name=None, feature="person"):
        self.feature = feature
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def detect(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        results = self.model(frame)
        label, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return label, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                if self.class_to_label(labels[i]) == self.feature:
                    x_center = x1 + (x2 - x1)
                    y_center = y1 + ((y2 - y1) / 2)
                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), self.feature))
            else:
                print("No detection")
                continue
        return frame, detections

class ObjectTracker:
    def __init__(self):
        self.object_tracker = DeepSort(
            max_age=10,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )

    def update_tracks(self, detections, frame):
        return self.object_tracker.update_tracks(detections, frame=frame)

class VideoProcessor:
    def __init__(self, yolo_detector, object_tracker):
        self.yolo_detector = yolo_detector
        self.object_tracker = object_tracker

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, img = cap.read()

            if not success:
                break

            start = time.perf_counter()

            results = self.yolo_detector.detect(img)
            img, detections = self.yolo_detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)

            tracks = self.object_tracker.update_tracks(detections, frame=img)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = ltrb
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.putText(img, "ID: "+str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            end = time.perf_counter()
            total_time = end - start
            fps = 1 / total_time

            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.imshow("Object Tracker", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo_detector = YoloDetector(model_name=None, feature="person")
    object_tracker = ObjectTracker()
    video_processor = VideoProcessor(yolo_detector, object_tracker)
    video_processor.process_video()
