from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def splitdataset(balance_data):
  
    # Separating the target variable
    X = balance_data.values[:, 2:-2]
    Y = balance_data.values[:, -2]
  
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
      
    return X, Y, X_train, X_test, y_train, y_test

df = pd.read_csv('Kaggle-data.csv')

df = df.loc[df['Machine'] != "3ab1aa9785d0681434766bb0ffc4a13c"]
for column in df.columns:
  if df[column].isnull().values.any():
    if column == 'MajorLinkerVersion':
      df2 = df.loc[~df[column].isnull()]

X, Y, X_train, X_test, Y_train, y_test = splitdataset(df2)

inputs = keras.Input(shape=(54,))
dense = layers.Dense(128, activation="relu")
x = dense(inputs)

x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(1,activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="MalwareClassification")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=["acc"],
)
model.summary()

X_train = np.asarray(X_train).astype(float)
Y_train = np.array(Y_train).astype(float)
history = model.fit(X_train, Y_train,batch_size=1, epochs=2, validation_split=0.2)


X_test = np.asarray(X_test).astype(float)
y_test = np.array(y_test).astype(float)
test_scores = model.evaluate(X_test, y_test, verbose=2)