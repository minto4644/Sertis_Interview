# Sertis Assignment for Machine Learning Engineer Role

## Problem statement:
Imagine we have a model deployed on the cloud which performs face recognition on images sent to it. This model works great on well-oriented images, i.e. images which are the right way up. However, when badly-oriented images are sent, e.g. upside-down images, the model performs poorly. Since we have no  control over how the images are sent and have no guarantee that the images will come with orientation-metadata, we would like a pre-processing step which fixes the orientation of the images before being sent to the main model. This is where you come in.

          


## Please attempt the following:

- Design a solution which classifies the orientation of any image of a face into one of four categories. The four orientations considered are shown in the image. 

- Create an API which takes in images of faces and returns them with their orientation fixed. This could be a RESTful API or just a class which contains your model.

- [Optional] Write some unit tests for your API.

- [Optional] Deploy your API somewhere, e.g., heroku

You can use any framework, dataset, stack overflow answer or github repo you desire , just make sure you can explain your code.

**We do not expect you to build or train your own model, but if you want to you can!**

If you have questions, feel free to ask.

Time: please mark your start time and finish it within 48 hours

Submission: please submit all the related code, and instructions on running the code, to the submission link at the bottom of the email.

A little Tip: we're not interested in just seeing you complete the problem in full, we want to see the quality of your work also. (its better to do the first two tasks well, than all 4 rushed!)



## Solution Approach

The basic problem is to detect and correct orientation of face in given image before it can be sent to image recognition model for inference. Since the model is not able to detect badly-oriented images, means the feature vector generated from bad-oriented image by model isn't good for recognition. Thus the model is not rotation invariant. 


I will list down the possible solution approach that I tried to ponder for this problem.

1. Using existing simple face detection model to detect face bounding boxes
	- Rotate the given image at 90, 180, 270 angles and use face detection model to get face predictions. Select for angle that gives maximun score(face detection models generally gives confidence score with predicted bounding box).

2. Traning orientation classification model from scratch
	- Take a human face dataset. Rotate images at 90, 180, 270 as data augumentation step. Label for each image would be roatation angle. Train on this dataset with angle as classification label. 


I went ahead with first approach. Let me explain the approach in detail.
Used dlib frontal face detector for detecting 


