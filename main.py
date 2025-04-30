from model import create_model, load_model
from extract_features import get_predict_feat
from encoder import create_encoder
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time

emotion_map = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad', 
    4: 'angry',
    5: 'fear',
    6: 'disgust'
}

def predict_emotion(path, model, encoder):
    res = get_predict_feat(path)
    predictions = model.predict(res)
    print(f"pred: {predictions}, max: {np.argmax(predictions)}")
    y_pred = encoder.inverse_transform(predictions)
    print(y_pred[0][0])
    return y_pred[0][0]

def main():
    start_time = time.time()
    # model = create_model('./model_weight/best_model1_weights.h5')
    model = load_model('./CNN_model.json','./model_weight/best_model1_weights.h5')
    audio_path = './test_data/angry-short.wav'
    encoder = create_encoder()
    emotion = predict_emotion(audio_path, model, encoder)
    print(f"Predicted emotion: {emotion}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()