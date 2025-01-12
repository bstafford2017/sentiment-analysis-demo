import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

def create_model(): 
    # Initiate the model
    model = Sequential([
        # 32 neurons for 1st hidden layer
        Dense(units=32, activation='relu'),
        # 64 neurons for 1st hidden layer
        Dense(units=64, activation='relu'),
        # One output layer
        Dense(units=1, activation='sigmoid')
    ])

    # Optimize with loss function
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# Read data set into memory
data = pd.read_csv('review-data/reviews.csv')

# Remove unnecessary columns
x = pd.get_dummies(data.drop(['id', 'profileName', 'rating', 'date', 'images', 'variant:color', 'variant:size'], axis=1))

y = data['helpful']

# Organize the data for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# Type conversion to float 32
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)

model = create_model()

# Train neural net
model.fit(x_train, y_train, epochs=200, batch_size=32)

# Evaluate performance 
test_loss = model.evaluate(x_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Make predictions on new data
y_pred = model.predict(x_test)

# Show a few predictions alongside actual values
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(predictions.head())

# Save the model
model.save('src/review-amazon/review-model.keras')
