import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Sequential, model_from_json
import os
import pickle

# Example: weights_path = './model_weight/best_model1_weights.h5'
def create_model(weights_path):
    # 1. Build model with exact original architecture
    model = tf.keras.Sequential([
        L.Conv1D(512, 5, strides=1, padding='same', activation='relu', input_shape=(2376, 1)),
        L.BatchNormalization(),
        L.MaxPool1D(5, strides=2, padding='same'),
        
        L.Conv1D(512, 5, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(5, strides=2, padding='same'),
        L.Dropout(0.2),
        
        L.Conv1D(256, 5, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(5, strides=2, padding='same'),
        
        L.Conv1D(256, 3, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(5, strides=2, padding='same'),
        L.Dropout(0.2),
        
        L.Conv1D(128, 3, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(3, strides=2, padding='same'),
        L.Dropout(0.2),
        
        L.Flatten(),
        L.Dense(512, activation='relu'),
        L.BatchNormalization(),
        L.Dense(7, activation='softmax')
    ])

    # 2. Build model properly
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    dummy_input = tf.ones((1, 2376, 1))
    _ = model.predict(dummy_input)

    # 3. Verify architecture
    print("Model summary:")
    model.summary()


    
    if not os.path.exists(weights_path):
        print(f" File not found: {weights_path}")
    else:
        try:
            model.load_weights(weights_path)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f" Error loading weights: {e}")
    
    return model
def load_model(model_path, weights_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    if not os.path.exists(weights_path):
        print(f" File not found: {weights_path}")
    else:
        try:
            model.load_weights(weights_path)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f" Error loading weights: {e}")
    return model
export = create_model, load_model
