from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
import numpy as np

from imutils import face_utils
from imutils import rotate_bound
import imutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import cv2
import dlib

class FaceOrient:
	def __init__(self, images, shape_predictor):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(shape_predictor)
		self.images = images
		self.corrected_images = []
		self.angles = []


	def inferOrient_68(self, gray, rot_angle, counter):
		# load the input image , resize it and convert to grayscale

		#rects = self.detector(gray, 1)
		dets, scores, idx = self.detector.run(gray, 1, 0)
		if len(dets) >1:
			print("Multiple faces detected in image. Input correct image")
			return 400
		elif len(dets) <1:
			print("No face detected")
			return 400
		else:
			# return the max score face from the detected faces
			#dets, scores, idx = self.detector.run(gray, 1, 0)
			
			for i,d in enumerate(dets):
				print("For image: {} angle: {} Detection {}, score: {}, face_type:{}".format(counter,rot_angle,
					d, scores[i], idx[i]))
			
			ma = max(scores)
			return ma
			'''
			shape = self.predictor(gray, rects[0])
			shape = shape_to_np(shape)

			# Extract lefteye and RightEye points
			(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

			leftEyePts = shape[lStart:lEnd]
			rightEyePts = shape[rStart:rEnd]

			# compute center of mass for each eye
			leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
			rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

			# Calaculate eyes Center by taking mid point of two eyes
			eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0])//2, 
				(leftEyeCenter[1] + rightEyeCenter[1])//2)

			# Extract nose points
			(nStart,nEnd) = FACIAL_LANDMARKS_IDXS["nose"]

			nosePts = shape[nStart:nEnd]

			# Nose center point
			noseCenter = nosePts.mean(axis=0).astype("int")

			# For calculating orientataion we need to calculate angle 
			# The line joining eyesCenter with noseCenter gives us angle 

			dX = noseCenter[0] - eyesCenter[0]
			dY = noseCenter[1] - eyesCenter[1]
			magnitude = np.sqrt((dX**2 + dY**2))

			angle = np.degrees(np.arctan2(dY, dX))
			print("ANGLE: %d" %(int(angle)))
			#(x,y,w,h) = face_utils.rect_to_bb(rects[0])
			#cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)

			# show the face number
			#cv2.putText(image, "Angle #{}".format(angle), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#cv2.line(image, tuple(noseCenter), tuple(eyesCenter),(255, 0, 0), 5)
			#for (x,y) in shape:
			#	cv2.circle(image, (x,y), 1,  (0,255,0), -1)

			#plt.imshow(gray)
			#plt.show()
			return ma
			'''




	def orient_image(self,image, gray,counter):
		max_score = -400.0
		correct_angle = 0
		for angle in range(0,360,90):
			#rot_start = time.time()
			#rotated_image = rotate_bound(image, angle)
			gray_rotated_image = rotate_bound(gray, angle)
			#rot_end = time.time()
			
			#print("For image: {} Time to rotate at angle {} is {} ".format(counter, angle, rot_end-rot_start))
			#infer_start = time.time()
			score = self.inferOrient_68(gray_rotated_image, angle,counter)
			#infer_end = time.time()
			#print("For image: {} Time to infer at angle {} is {} ".format(counter, angle, infer_end-infer_start))
			if score != 400:

				#print(score, max_score)
				if max_score < score:
					max_score = score
					correct_angle = angle

		corrected_image = rotate_bound(image, correct_angle)
		print(correct_angle)
		#return corrected_image
		self.corrected_images.append(corrected_image)
		if correct_angle == 0:
			self.angles.append("D")
		elif correct_angle == 90:
			self.angles.append('R')
		elif correct_angle == 180:
			self.angles.append('U')
		else:
			self.angles.append('L')


	def orient_images(self):
		for i in tqdm(range(len(self.images))):
			if self.images[i] is not None:
				image = imutils.resize(self.images[i], width=300, height=300)
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				self.orient_image(image, gray, i)

		return self.corrected_images, self.angles