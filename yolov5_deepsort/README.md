<H1 align="center"> Object Detection using YOLOv5 and DeepSORT Tracking Pytorch </H1>

### How to setup

- Clone the repository

- Goto the cloned folder.
```
cd yolov5_deepsort
```
- Clone yolov5 repository
```
git clone https://github.com/ultralytics/yolov5
```
- Install dependencies
```
pip install -r yolov5/requirements.txt
```

### Usage

#### for detection and tracking
```
python track.py --yolo-weights yolov5n.pt --source 'your video.mp4' --device 0 --show-vid
```
#### for WebCam
```
python track.py --yolo-weight yolov5n.pt --source 0 --device 0
```
#### for specific class (person)
```
python track.py --yolo-weights yolov5n.pt --source 'your video.mp4' --device 0 --show-vid --classes 0
```

- Output file will be created in the ```working-dir/runs/track/exp``` with original filename