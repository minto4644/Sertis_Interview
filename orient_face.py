from face_orient import FaceOrient
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import glob
import os
import time
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--images-dir", required=True,
	help="path to input images folder")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(args["shape_predictor"])
types = ("*.jpg", "*.jpeg", "*.png", "*.JPG")
files = []
file_paths = []
print("OK")
for file_type in types:
	path = os.path.join(args["images_dir"], file_type)
	print(path)
	paths = glob.glob(path)
	file_paths.extend(paths)
	files.extend([cv2.imread(img) for img in paths])
print("OK")
#print(files)
#image = cv2.imread(args["image"])
#image = imutils.resize(image, width=500)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fo = FaceOrient(files, args["shape_predictor"])
start = time.time()
oriented_images, orientations = fo.orient_images()
end = time.time()
for path,orient in zip(file_paths, orientations):
	print(path, orient)
print("Time taken to orient {} images is : {}".format(len(files), end-start))
out_dir = os.path.join(args["images_dir"],"test_out")
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

for i in range(len(oriented_images)):
	cv2.imwrite(out_dir + '/' + str(i) +'.jpg', oriented_images[i])
#print(oriented_images)
#cv2.imshow("Corrected_image", oriented_image)
#cv2.waitKey(5000)
#cv2.destroyAllWindows()