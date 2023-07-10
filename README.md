# Exercise Counting with YOLOv8



## Introduction
The application is a pose estimation demo for exercise counting with YOLOv8. 
It relies on YOLOv8-Pose.

## Installation

```
# (1) Creating a conda virtual environment
conda create -n yolov8 python=3.9
conda activate yolov8
# (2) Clone and install the ultralytics repository
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
# (3) Clone exercise counting demo
git clone https://github.com/yuyoujiang/exercise-counting-with-YOLOv8.git
```

## Getting Started

- Download favorite model weights [here](https://docs.ultralytics.com/tasks/pose/#models).
- Prepare a video to be tested, such as [here](https://github.com/yuyoujiang/test_video).
- Then run [demo.py](./demo.py) to view the counting results.


## References

[https://github.com/ultralytics/](https://github.com/ultralytics/)