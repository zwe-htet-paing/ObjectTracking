<H1 align="center"> Object Detection using YOLOv9 and DeepSORT Tracking Pytorch </H1>

### How to setup

- Clone the repository

- Goto the cloned folder.
```
cd YOLOv9-DeepSORT
```
- Clone yolov9 repository
```
git clone https://github.com/WongKinYiu/yolov9
```
- Install dependencies
```
pip install -r yolov9/requirements.txt
```
- Download the pre-trained YOLOv9 model weights
[yolov9](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt)

- Downloading the DeepSORT Files From The Google Drive 
```
gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
```
- After downloading the DeepSORT Zip file from the drive, unzip it. 

### Usage

#### for detection only
```
python detect_dual.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0
```
#### for detection and tracking
```
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0
```
#### for WebCam
```
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 0 --device 0
```
#### for specific class (person)
```
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --classes 0
```

- Output file will be created in the ```working-dir/runs/detect/exp``` with original filename