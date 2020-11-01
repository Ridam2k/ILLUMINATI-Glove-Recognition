

# Team ILLUMINATI - Glove Detection

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

For our complete submisisson, headover to - [Illuminati](https://github.com/swatig23/myntra_ILLUMINATI)  
This is the second component of our project - [**Real Time Glove Detection**](https://github.com/Ridam2k/ILLUMINATI-Glove-Recognition)

Before going further please see the demo - [Demo](https://photos.app.goo.gl/HNz73R7qVCbS2LVp9)

# Features

-   Easy 5 step setup on your local PC.
-   No external peripherals required, your in-built webcam is all it needs!
-   Can be easily accessed through our interactive website
-   Boasts of an accuracy of 90-93% and identifies gloves of varying colors and shapes

You can also:

-   Use this as an API and integrate with various other projects

# Tech

Our RT-Glove Detection uses a number of open source projects and libraries to work properly:

-   Tensorflow - For training our DL Model.
-   Keras - For helping us evaluate our DL Model
-   Open CV - For Hand Recognition and all our computer vision needs.
-   Caffe - A deep learning framework made with expression, speed, and modularity in mind. Also used for Hand Detection.
-   Streamlit - Open Source Project that helps us serve our models as a web app.

# How Does it Work?

## Dataset

-   We use a dataset that consists of **1135 images** belonging to two classes:
    -   **yesreal: 635 images**
    -   **noreal: 500 images**
-   Link to dataset - [Download](https://www.kaggle.com/adityagupta008/medical-gloves-recognition-dataset)

## Training

**Step 1:** Prepare Data and Augment Data - First we start by augmenting data to increase training data size using the following methods:

-   Horizontal Flip
-   Zoom
-   Width and Height Shift Range

**Step2:** The overall model developed is quite light-weight as it uses a fairly simple CNN architecture for classifying the real-time detected image into “Glove” and “No Glove”. For the purpose of detecting the hand from an image, pre-trained Haar Feature-based Cascade Classifier is used. This particular classifier is selected because:

-   It is an effective object detection method with pre-trained methods easily accessible on it’s [GitHub](https://github.com/opencv/opencv/tree/master/data) repository.
-   It is specially focused on detecting human body parts from images and is thus quite reliable.
-   It is one of the fewest algorithms to run in real-time. The key to its high performance is the use of integral image, which only performs basic operations in a very low processing time.
-   The propsed model can therefore be used in real-time applications which require glove-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

**The CNN Network architecture**  
![](https://raw.githubusercontent.com/mk-gurucharan/Face-Mask-Detection/master/cnn_facemask.png)

**Step 3:** Evaluate Performance and save model to disk. Our model performs very well on real life data. Below is an example:

# Installation

We have tried to keep the installation process as simple as possible. Please keep note of the following points:

-   Please follow the exact steps.The App is tested and verified only if each step is followed.
-   First time deployment of app might take 1 -2 minutes.
-   Note that all commands are to be entered with Admin Permissions in the command line.

Step 1 - Clone the Repository.

```sh
$ git clone https://github.com/Ridam2k/ILLUMINATI-Glove-Recognition
$ cd ILLUMINATI-Glove-Recognition

```

Step 2 - Create Virtual Env and Download Reqs.

```sh
$ mkvirtualenv tester (optional)
$ pip3 install -r requirements.txt
OR
$ mkvirtualenv tester (optional)
$ pip install -r requirements.txt

```

And that’s it! You are done. Let’s now get this working.

# Usage

Step 1: CD into the folder  
Step 2: Enter the following code

> Webcam Mode

```sh
$ python glove_detection.py 
OR
$ python3 glove_detection .py 

```

# That’s it !

Team Illuminati  
NSUT, Delhi
