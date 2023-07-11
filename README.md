# Deploy Exercise Counting with YOLOv8 on Jetson

![result 00_00_00-00_00_30](https://github.com/yuyoujiang/exercise-counting-with-YOLOv8/assets/76863444/d5657aa5-5a61-4451-9adb-7f9fbd395a13)




The application is a pose estimation demo for exercise counting with YOLOv8. 
It relies on YOLOv8-Pose.
This document also describes how to deploy the algorithm to edge computing device 
like [Jetson Orin NX](https://www.seeedstudio.com/reComputer-J4011-p-5585.html?queryID=7e0c2522ee08fd79748dfc07645fdd96&objectID=5585&indexName=bazaar_retailer_products).


## Introduction

The YOLOv8-Pose model can detect 17 key points in the human body, 
then select discriminative key-points based on the characteristics of the exercise. 
Calculate the angle between key-point lines, 
when the angle reaches a certain threshold, the target can be considered to have completed a certain action.
By utilizing the aforementioned mechanism, 
it is possible to achieve remarkably fascinating *Exercise Counting* Application.

## Installation

- For Windows/Ubuntu Desktop
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

- For Jetson
- - (1) Flash JetPack OS to edge device [(Refer to here)](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/).
- - (2) Install Ultralytics.
- - - Firstly, download the source code and unregister torch and torchvision from requirements.txt. [(Refer to here)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/#install-necessary-packages)
- - - Then, install Ultralytics by this command: `pip install -e .`
- - (3) Install PyTorch and Torchvision [(Refer to here)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/#install-pytorch-and-torchvision).
- - - `pip3 install ultralytics` may install torch and torchvision, but they don't work properly on jetson. So we need to uninstall Torch and Torchvision first, and then refer to the above link to reinstall.
- - - `yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'` Please run the command here to ensure that yolo can work properly.
- - (4) Clone exercise counting demo: `git clone https://github.com/yuyoujiang/exercise-counting-with-YOLOv8.git`. 


## Getting Started

- Download favorite model weights [(Refer to here)](https://docs.ultralytics.com/tasks/pose/#models).
- Prepare a video to be tested, such as [(Refer to here)](https://github.com/yuyoujiang/test_video).
- Then run [demo.py](./demo.py) to view the counting results `python demo.py --model yolov8s-pose.pt`.


## References

[https://github.com/ultralytics/](https://github.com/ultralytics/)  
[https://wiki.seeedstudio.com](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)
