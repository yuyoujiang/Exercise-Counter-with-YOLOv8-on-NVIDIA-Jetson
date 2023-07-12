# Exercise Counter with YOLOv8 on NVIDIA Jetson
![ezgif com-optimize (1)](https://github.com/yuyoujiang/exercise-counting-with-YOLOv8/assets/76863444/d592ff9b-6bc2-4017-8731-cf408052f0dd)


This is a pose estimation demo application for exercise counting with YOLOv8 using [YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose) model. 

This has been tested and deployed on a [reComputer Jetson J4011](https://www.seeedstudio.com/reComputer-J4011-p-5585.html?queryID=7e0c2522ee08fd79748dfc07645fdd96&objectID=5585&indexName=bazaar_retailer_products). However, you can use any NVIDIA Jetson device to deploy this demo.

Current only 3 different exercise types can be counted:

- Squats
- Pushups
- Situps

However, I will keep updating this repo to add more exercises and also add the function of detecting the exercise type.

## Introduction

The YOLOv8-Pose model can detect 17 key points in the human body, then select discriminative key-points based on the characteristics of the exercise. 
Calculate the angle between key-point lines, when the angle reaches a certain threshold, the target can be considered to have completed a certain action.
By utilizing the above-mentioned mechanism, it is possible to achieve an interesting *Exercise Counter* Application.

## Installation

- **Step 1:** Flash JetPack OS to reComputer Jetson device [(Refer to here)](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/).

- **Step 2:** Access the terminal of Jetson device, install pip and upgrade it

```sh
sudo apt update
sudo apt install -y python3-pip
pip3 install --upgrade pip
```

- **Step 3:** Clone the following repo

```sh
git clone https://github.com/ultralytics/ultralytics.git
```

- **Step 4:** Open requirements.txt

```sh
cd ultralytics
vi requirements.txt
```

- **Step 5:** Edit the following lines. Here you need to press i first to enter editing mode. Press ESC, then type :wq to save and quit

```sh
# torch>=1.7.0
# torchvision>=0.8.1
```

**Note:** torch and torchvision are excluded for now because they will be installed later.

- **Step 6:** Install the necessary packages

```sh
pip3 install -e .
```

- **Step 7:** If there is an error in numpy version, install the required version of numpy

```sh
pip3 install numpy==1.20.3
```

- **Step 8:** Install PyTorch and Torchvision [(Refer to here)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/#install-pytorch-and-torchvision).

- **Step 9:** Run the following command to make sure yolo is installed properly

```sh
yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' 
```

- **Step 10:** Clone exercise counter demo

```sh
git clone https://github.com/yuyoujiang/exercise-counting-with-YOLOv8.git
```

## Prepare The Model File

YOLOv8-pose pretrained pose models are PyTorch models and you can directly use them for inferencing on the Jetson device. However, to have a better speed, you can convert the PyTorch models to TensorRT optimized models by following below instructions.

- **Step 1:** Download model weights in PyTorch format [(Refer to here)](https://docs.ultralytics.com/tasks/pose/#models).

- **Step 2:** Execute the following command to convert this PyTorch model into a TensorRT model 

```sh
# TensorRT FP32 export
yolo export model=yolov8s-pose.pt format=engine device=0

# TensorRT FP16 export
yolo export model=yolov8s-pose.pt format=engine half=True device=0
```

**Tip:** [Click here](https://docs.ultralytics.com/modes/export) to learn more about yolo export 

- **Step 3:** Prepare a video to be tested. [Here]() we have included sample videos for you to test

## Let's Run It!

To run the exercise counter, enter the following commands with the `exercise_type` as:

- sit-up
- pushup
- squat

### For video 

```sh
python3 demo.py --sport <exercise_type> --model yolov8s-pose.pt --show True --input <path_to_your_video>
```

### For webcam

```sh
python3 demo.py --sport <exercise_type> --model yolov8s-pose.pt --show True --input 0
```
![result 00_00_00-00_00_30](https://github.com/yuyoujiang/exercise-counting-with-YOLOv8/assets/76863444/414e1cd1-ab7d-4ca6-91e4-c8a948fe55ae)

## References

[https://github.com/ultralytics/](https://github.com/ultralytics/)  
[https://wiki.seeedstudio.com](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)
