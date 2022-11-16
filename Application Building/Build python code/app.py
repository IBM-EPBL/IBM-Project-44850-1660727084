from __future__ import division, print_function
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)
MODEL_PATH = 'fruit.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()
default_image_size = (128, 128)
labels = ["Apple___Black_rot", "Apple___healthy", "Corn_(maize)___healthy",
"Corn_(maize)___Northern_Leaf_Blight", "Peach___Bacterial_spot","Peach___healthy"]
def convert_image_to_array(image_dir):
 try:
  image = cv2.imread(image_dir)
  if image is not None:
   image = cv2.resize(image, default_image_size)
   return img_to_array(image)
  else:
   return np.array([])

