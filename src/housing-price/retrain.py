import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

# Load existing model
model = tf.keras.models.load_model('src/housing-price/housing-model.keras')

# Show the model architecture
model.summary()

# Read data set into memory
df = pd.read_csv('house-data/Bengaluru_House_Data.csv')

# Removing rows with missing values
df_cleaned = df.dropna()

# Remove unnecessary columns
X = pd.get_dummies(df_cleaned.drop(['society', 'price'], axis=1))

# Idenity output column
y = df_cleaned['price']

# Normalize the numerical features
normalizer = Normalizer()
normalized_train_X = normalizer.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(normalized_train_X, y, test_size=0.2)

# Train neural net
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate performance 
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Make predictions on new data
y_pred = model.predict(X_test)

# Show a few predictions alongside actual values
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(predictions.head())

# Save the model
model.save('housing-model.keras')

