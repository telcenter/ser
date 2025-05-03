from ser import SERModel

def main():
    model = SERModel(
        model_path='./model_weight/CNN_model.json',
        weights_path='./model_weight/best_model1_weights.h5'
    )
    audio_path = './test_data/surprise.wav'
    emotion = model.predict_emotion_from_wav_file(audio_path)
    print(f"Predicted emotion: {emotion}")

if __name__ == "__main__":
    main()
