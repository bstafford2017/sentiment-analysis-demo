import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np

# Read data set into memory
data = pd.read_csv('reviews.csv')

# Remove unnecessary columns
x = pd.get_dummies(data.drop(['id', 'profileName', 'rating', 'date', 'images', 'variant:color', 'variant:size'], axis=1))

y = data['helpful'].apply(lambda x: 50 if x=='Yes' else 0)

# Organize the data for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

y_train.head()

# Type conversion to float 32
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)

# Initiate the model
model = Sequential()

# 32 neurons for 1st hidden layer
model.add(Dense(units=32, activation='relu'))

# 64 neurons for 1st hidden layer
model.add(Dense(units=64, activation='relu'))

# One output layer
model.add(Dense(units=1, activation='sigmoid'))

# Optimize with loss function
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train neural net
model.fit(x_train, y_train, epochs=200, batch_size=32)

y_hat = model.predict(x_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

# Analysis
accuracy_score(y_test, y_hat)

# Save
model.save('tfmodel.keras')
