# DeepStab

Deepstab is a tool that is able to detect an specific object on a video and stabilize it by tracking the object and centering the image. The tool is compatible for models trained in TensorFlow Object Detection API, and Caffee models. There is a full explanation of how this works in this article [this article](https://medium.com/hci-wvu/face-stabilization-in-videos-using-deep-learning-features-dcfd4be365).

## Installation

Run

> pip install -r requirements.txt

## Quickstart

Example how to run it:

> python app.py

In the built in example, the model is a face detection model (tf_ssd_mobilenet_openimages.pb) and you can tested it with your webcam by placing your face in front of it and moving around.

Models can be found here
https://drive.google.com/drive/folders/1Wak-kieXQtpo8UtqL30WDFk1kfQVYCxu?usp=sharing
