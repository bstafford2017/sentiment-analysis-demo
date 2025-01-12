import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Rescaling, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

train_path = 'animal-data/animals'
predict_path = 'animal-data/predict_3.jpg'
batch_size = 32
img_height = 180
img_width = 180

# Get full training set
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Get all classification output values
class_names = train_ds.class_names
print(class_names)

# Load predict image
img = tf.keras.utils.load_img(
    predict_path, target_size=(img_height, img_width)
)

# Convert to array
img_array = tf.keras.utils.img_to_array(img)

# Create a batch
img_array = tf.expand_dims(img_array, 0) 

# Load existing model
model = tf.keras.models.load_model('src/animal-recognition/model.keras')

# Show the model architecture
model.summary()

# Perform prediction
predictions = model.predict(img_array)

# Perform distribution
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)