import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from keras.backend import clear_session


import os
import glob
import datetime

import tensorflow as tf

from keras.applications.resnet50 import preprocess_input


clear_session()

import custom
# Root directory of the project
ROOT_DIR = "/Users/myrondza/Library/Mobile Documents/com~apple~CloudDocs/Damage Detection AI/"
ROOT_DIR1 = "/Users/myrondza/Library/Mobile Documents/com~apple~CloudDocs/Damage Detection AI/static"
ROOT_DIR2 = "/Users/myrondza/Library/Mobile Documents/com~apple~CloudDocs/Damage Detection AI/static/uploads"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  

custom_WEIGHTS_PATH = sorted(glob.glob("mask_rcnn_*.h5"))[-1]


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "customImages")


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



# Load validation dataset
import custom 

dataset = custom.CustomDataset()
dataset.load_custom(custom_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)
model.keras_model._make_predict_function()

from importlib import reload 
reload(visualize)



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	import os
	import glob

	files = glob.glob('/Users/myrondza/Library/Mobile Documents/com~apple~CloudDocs/Damage Detection AI/static/uploads/*')
	for f in files:
		os.remove(f)
	if 'files[]' not in request.files:
		flash('No file part')
		return redirect(request.url)
	files = request.files.getlist('files[]')
	file_names = []
	for file in files:
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file_names.append(filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		IMAGE_DIR = os.path.join(ROOT_DIR1, "uploads")
		print(IMAGE_DIR)
		class_names = ['BG','damage']
		file_names = next(os.walk(IMAGE_DIR))[2]
		print(file_names)
		image1 = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[0]),plugin='matplotlib')
		print(image1)
		results = model.detect([image1], verbose=0)
		r = results[0]

		test_labels = os.listdir(ROOT_DIR2)
		test_path = ROOT_DIR2

		# variables to hold features and labels
		features_new = []

		from keras.preprocessing import image
		from keras.models import Model


		res_model = tf.keras.applications.ResNet50(weights="imagenet")
		image_size = (224, 224)
		
		# loop over all the labels in the folder

		count = 1
		for i, label in enumerate(test_labels):
			cur_path = test_path + "/" + label
			count = 1
			for image_path in glob.glob(cur_path):
				img = image.load_img(image_path, target_size=image_size)
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				feature = res_model.predict(x)
				flat = feature.flatten()
				features_new.append(flat)
				print ("Processed - " + str(count))
				count += 1
			print ("Completed label - " + label)


		########## Predict Damage or Not ###########

		print(features_new)
		import pickle
		logmodel = pickle.load(open("car_parts_check/classifier.pickle", 'rb'))
		preds = logmodel.predict(np.array(features_new))
		preds


		########## Predict Damage Parts Location ###########

		clf = pickle.load(open("car_parts_check1/classifier1.pickle", 'rb'))
		preds1 = clf.predict(np.array(features_new))
		preds1


		########## Predict Scartch or Dent ###########

		logmodel1 = pickle.load(open("car_parts_check2/classifier2.pickle", 'rb'))
		preds2 = logmodel1.predict(np.array(features_new))
		print(preds2)  # Scratch / Dent

		label_check1 = ['Damaged', 'Not Damaged']
		print(label_check1[preds[0]])
		label_check2 = ['NULL','Scratch','Dent']
		print(label_check2[preds2[0]])
		print(preds1) # Parts
		parts=""
		price=""
		repair=""

		if (preds1==1):
			parts = "Bonnet"
			price = "₹7,864"
			repair = "Repairing, Denting & Painting of Bonnet - Damaged Vehicle"
		elif (preds1==2):
			parts = "Rear"
			price = "₹7,940"
			repair = "Repairing, Denting & Painting of Rear - Damaged Vehicle"
		elif (preds1==3):
			parts = "Left"
			price = "₹7,480"
			repair = "Repairing, Denting & Painting of LHS Door - Damage Vehicle"
		elif (preds1==4):
			parts = "Right"
			price = "₹7,480"
			repair = "Repairing, Denting & Painting of RHS Door - Damage Vehicle"
			

		import pandas as pd
		final = pd.DataFrame({'Status':[label_check1[preds[0]]],'Type':[label_check2[preds2[0]]],'Location':parts})
		final.set_index('Status',inplace=True)
		final = final.transpose()

		visualize.save_image(image1,image_name="1",boxes=r['rois'],masks=r['masks'],class_ids=r['class_ids'],class_names=class_names,scores=r['scores'],filter_classs_names=None,scores_thresh=0.1, save_dir="/Users/myrondza/Library/Mobile Documents/com~apple~CloudDocs/Damage Detection AI/static/uploads/", mode=0)
		filename1="1.jpg"
		file_names.append(filename1)
		print(file_names)
		return render_template('upload.html', filenames=file_names,scroll='something',tables=[final.to_html(classes='data', index = True)], titles=final.columns.values,value_p=price,value_r=repair)
	else:
		flash('Allowed image types are -> png, jpg, jp')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()