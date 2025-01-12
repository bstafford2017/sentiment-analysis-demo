import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

def create_model(X_train): 
    # Initiate the model
    model = Sequential([
        # 64 neurons for 1st hidden layer
        Dense(units=64, activation='relu'),
        # 32 neurons for 2nd hidden layer
        Dense(units=32, activation='relu'),
        # One output layer
        Dense(1)
    ])

    # Optimize with loss function
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

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

model = create_model(X_train)

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
model.save('src/housing-price/housing-model.keras')
