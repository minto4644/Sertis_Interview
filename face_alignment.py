# import the necessary packages
from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
import numpy as np
import cv2

class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
		desiredFaceWidth=256, desiredFaceHeight=None):

		'''
		Args:

			desiredLeftEye: a tuple of percentages. These percentages control how much the face is visible after alignment
		'''

		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight

		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth


	def align(self, image, gray, rect):
		'''
		@Args
			image: RGB input image
			gray: grayscal input image
			rect: bounding box rectangle produced by dlib's HOG face detector
		'''
		# Convert the landmark (x,y)-coordinates to a numpy array
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)

		# extract the left and right eye coordinates
		(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		# compute center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between the eye centriods
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]

		angle = np.degrees(np.arctan2(dY, dX)) -180

		# compute thr desired right-eye x-coordinate based on
		# the desired x-coordinates of the left eye

		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking the ratio
		# of the distance between eyes in the current image to the ratio of
		# distance between eyes in the desired image

		dist = np.sqrt((dX**2) + (dY**2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist/dist

		# Compute center (x,y)-coordinates(median point) of the two eyes 
		# in the input frame

		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0])//2, 
						(leftEyeCenter[1] + rightEyeCenter[1])//2)

		# grab the rotation matrix for rotation and scaling
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth*0.5
		tY = self.desiredFaceHeight*self.desiredLeftEye[1]

		M[0,2] += (tX - eyesCenter[0])
		M[1,2] += (tY - eyesCenter[1])

		(w,h) = (image.shape[1], image.shape[0])

		output = cv2.warpAffine(image, M, (w,h), 
				flags=cv2.INTER_CUBIC)
		return output


