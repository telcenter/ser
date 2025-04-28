from model import create_model, load_model
from extract_features import get_predict_feat
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

def predict_emotion(audio_path, model):
    features = get_predict_feat(audio_path)  
    pred = model.predict(features)
    print(f"pred: {pred}")
    return emotion_map[np.argmax(pred)]

def main():
    print("==========================")
    print("Loading model...")
    start_time = time.time()
    # model = create_model('./model_weight/best_model1_weights.h5')
    model = load_model('./CNN_model.json','./model_weight/best_model1_weights.h5')
    end_time = time.time()
    print(f"Model loaded in {(end_time - start_time):.2f} seconds.")
    print("==========================")
    print("\n" * 6)

    for audio_path in [
        './test_data/03-01-05-02-01-02-06.wav',
        './test_data/03-01-05-02-01-02-06.wav',
        './test_data/angry_1.wav',
        './test_data/happy_1.wav',
    ]:
        print("==========================")
        print(f"Predicting emotion of file: {audio_path}")
        start_time = time.time()
        emotion = predict_emotion(audio_path, model)
        print(f"Predicted emotion: {emotion}")
        end_time = time.time()
        print(f"Prediction completed in {(end_time - start_time):.2f} seconds.")
        print("==========================")

if __name__ == "__main__":
    main()
