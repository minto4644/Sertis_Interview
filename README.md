# Sertis Assignment for Machine Learning Engineer Role

### Problem statement:
Imagine we have a model deployed on the cloud which performs face recognition on images sent to it. This model works great on well-oriented images, i.e. images which are the right way up. However, when badly-oriented images are sent, e.g. upside-down images, the model performs poorly. Since we have no  control over how the images are sent and have no guarantee that the images will come with orientation-metadata, we would like a pre-processing step which fixes the orientation of the images before being sent to the main model. This is where you come in.

          


### Please attempt the following:

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



### Solution Approach

The basic problem is to detect and correct orientation of face in given image before it can be sent to image recognition model for inference. Since the model is not able to detect badly-oriented images, means the feature vector generated from bad-oriented image by model isn't good for recognition. Thus the model is not rotation invariant. 


I will list down the possible solution approach that I tried to ponder for this problem.

1. Using existing simple face detection model to detect face bounding boxes
	- Rotate the given image at 90, 180, 270 angles and use face detection model to get face predictions. 
	- Select for angle that gives maximun score(face detection models generally gives confidence score with predicted bounding box).

2. Traning orientation classification model from scratch
	- Take a human face dataset. 
	- Rotate images at 90, 180, 270 as data augumentation step. 
	- Label for each image would be roatation angle. 
	- Train on this dataset with angle as classification label. 

I went ahead with first approach. Let me explain the approach in detail.

- Solution Initution

&nbsp;&nbsp;&nbsp;&nbsp;Following the intuition from facial landmarks detection [blog](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) by pyimagesearch, thought of extracting leftEye , rightEye and nose coordinates. Then calculate angle bewtween midpoint of eyes and nose point for understanding orientation. But, this approach was based on first detecting faces inside images. Dlib hog detector is not rotation invariant and thus was failing on badly-oriented images to detect faces. Egg vs chicken problem.

Let's name 4 orientations that we need to classify.

	L: LEFT
	R: RIGHT
	D: DOWN
	U: UP

Dlib HOG detector is based on Histogram Of Gradients method to extract features and then uses SVM for training. 

A very basic solution is to rotate given image for all orientations and use Dlib detector to get faces bb and their corresponding scores. Select one that gives max scores across all angle configurations(0,90,180,270). 

imutils is a nice utility package that enhances over cv2 functions. Used rotate_bound from imutils to correctly rotate image in clockwise direction. Rotating using cv2.getMatrixRotatation2D and cv2.warpAffine crops out roatated image because the roatated matrix have different height and width as compared to original image. 

Since the rotation function rotate_bound rotates in clockwise direction, orientatation for different angle is as follows:
	
	- 0
	The given image is in correct orientation. i.e "DOWN" orientatation
	- 90
	The given image has to be rotated 90 degrees in clockwise direction for detector to get maximum score . Means given image was in "RIGHT" orientatation
	- 180
	Upside down oriented
	- 270
	Left side oriented

## Steps to run

### Clone and cd into it
```bash
git clone https://github.com/minto4644/orienTFace.git
cd orienTFace
```
### Create Env
```bash
conda create -n orient python=2.7
source activate orient
```
### Install dependencies
Create new conda env for easy speration of projects and their dependendies
```bash
pip install requirements.txt
```
## Download and extract pretrained dlib hog model
```bash
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -zk shape_predictor_68_face_landmarks.dat.bz2
```
## Run
```bash
python run_face_orient.py --shape-predictor shape_predictor_68_face_landmarks.dat --images-dir sample
```

## Brief about code files
- run_face_orient
	- The orient_face takes into two arguments . One is pre-trained dlib model and other is directory inside which images are present. 
	- Reads images inside directory
	- It creates directory names "out" inside sample directory. All the inferenced images will be written into it.
- face_orient
	- Contains FaceOrient class .  Intializes with detector, predictor, and images.
	- Returns:
		- Corrected Images i.e rotated to 'D' orientatation
		- Original orientatations of given images




