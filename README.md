<!-- ABOUT THE CAPSULE -->
## About The Capsule
<div align= center><img src="resources/fall_people.jpg"/ height="500"></div>


Capsule is a structure trained with a custom YOLOv8 dataset. The capsule consists of 5 configurations: 4 main configurations and 1 configuration under the device setting.

The Device configuration offers a selection between GPU and CPU. If GPU is selected, options to enable or disable half-precision are presented.

For Draw BBox, there is an option to activate or deactivate. The IoU Threshold and Confidence Threshold configurations accept float values between 0 and 1. The capsule only accepts images as input and returns Detections and the image as output.



### Built With

The containers that work together with the capsule are as follows:

* Pytorch
* OpenCV
* Wsl
* Redis

> [!WARNING]
> Do not forget Pytorch is main image.
## Configs
* **Device** (dependentDropdownlist)
  * **half-precision**       (dropdownlist)
* **Draw Bbox**              (dropdownlist)
* **Confidence Threshold**   (textInput-float number)
* **IoU Threshold**          (textInput-float number)



### Installation

Clone the repo to under your image.
   ```sh
   git submodule add -f -b develop https://github.com/novavision-ai/cap-fall-detection.git .\capsules\FallDetection
   ```


<!-- USAGE EXAMPLES -->
## Train
You can use this [Notebook](https://colab.research.google.com/drive/1UCSDuAaQYVvIUMP7pDvu0Kc6rX94XAFz?usp=sharing) to train model.
First of all we need to dataset to train model in my model I used this [Roboflow Dataset](https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8/dataset/4)
In dataset 9444 Train, 899 valid, 450 test image available and you can download it any way format.
In the Notebook in first section here
```python
from roboflow import Roboflow
rf = Roboflow(api_key=[YOUR_ACCOUNT_API_KEY])
project = rf.workspace("roboflow-universe-projects").project("fall-detection-ca3o8")
version = project.version(4)
dataset = version.download("yolov8")
```
You need to enter your account api key



<!-- Resources -->
## Resources

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Yolov8 Training with Custom Dataset](https://www.youtube.com/watch?v=LNwODJXcvt4)
* [Train Notebook](https://colab.research.google.com/drive/1UCSDuAaQYVvIUMP7pDvu0Kc6rX94XAFz?usp=sharing)
* [Roboflow Dataset](https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8/dataset/4)
* [My Faster-RCNN model](https://colab.research.google.com/drive/1g3HQewFUte63QPHRU1J6owZ_Xj_jQzGD?usp=sharing)
