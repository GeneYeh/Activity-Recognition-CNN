# Deep Learning-Based Real-Time Activity Recognition with Multiple Inertial Sensors

## Abstract
This thesis proposes a real-time activity recognition system based on data from several [wearable inertial sensors](https://epl.tw/ecomini/). They are worn at the user’s right wrist, waist, and right ankle to collect acceleration and angular velocity data, which are then transmitted via Bluetooth to a computer. The data are used to train a convolutional neural network (CNN) model to recognize 13 types of activities, including sitting, standing, walking, going upstairs, going downstairs, drinking water, brushing teeth, cleaning, jogging, opening a door, stretching, lying down, and walking while using a mobile phone. The trained model has been ported to Tensorflow Lite running on [Raspberry Pi](https://www.raspberrypi.org/products/raspberry-pi-3-model-b-plus/) to enable edge processing. The latency for recognizing the first motion takes 2.6 seconds, and subsequent ones are 1.325 seconds. Experimental results show that among the six human subjects, our model achieves high accuracy of 98.62% and 99.84% for leave-one-out cross-validation and 10-fold cross-validation, respectively.

### Daily activity data
This repository contains **3060 daily activity data** are collected from **six volunteers** equipped with several [wearable inertial sensors](https://epl.tw/ecomini/). The inertial sensor's sample rate is 125.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

###Prerequisites

- install [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

- install [bluepy](https://github.com/IanHarvey/bluepy)

- install the following packages:

```javascript
pip install numpy scipy matplotlib pandas keras
pip install --upgrade tensorflow-gpu==1.5.0
```


## System overview
There are five steps to implement the system:

1. Collected raw data from sensors and stored with pickle type.
    - ./Collectdata/Collectdata.py
    
2. Preprocessing (low-pass filter and data alignment), normalization, segmentation, and save the data to txt format.
    - Preprocessing.py
    
3. Get data from the txt file, train CNN model by leave-one-out cross-validation and 10-fold cross-validation, and then save the highest accuracy's Keras model.
    - Loocv_CNN.py
    - K-Fold_CNN.py
    
4. Convert the Keras model to TensorFlow lite model and store the TensorFlow lite model to Raspberry Pi for real-time recognition.
    - Keras2tflite.py
    
5.  The result of recognition is shown in Raspberry Pi.
    - ./Demo/Demo.py


## Demo [link1](https://www.youtube.com/watch?v=coPhCzglX8w) [link2](https://www.youtube.com/watch?v=xjqU5sxhCuw)


## More Information
For more details about the methods and the performance, please see the attached thesis.pdf.
