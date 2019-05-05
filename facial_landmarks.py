# import important lbrabries for image processing
import numpy as np
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
				help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark detector

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image , resize it and convert to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)


# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region , then
	# Convert the facial landmark (x,y)-coordinates to Numpy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# Convert dlib's rectangle to a OpenCV-style bounding box
	# i.e (x,y,w,h) then draw the boundig box
	(x,y,w,h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x,y)-coordinates for the facial landmarks
	# and draw then on image

	for (x,y) in shape:
		cv2.circle(image, (x,y), 1, (0, 255, 0), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)
