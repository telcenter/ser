from .model import load_model #, create_model
from .extract_features import get_predict_feat
from .encoder import create_encoder
import numpy as np
import time
import logging

# emotion_map = {
#     0: 'neutral',
#     1: 'calm',
#     2: 'happy',
#     3: 'sad', 
#     4: 'angry',
#     5: 'fear',
#     6: 'disgust'
# }

class SERModel:
    def __init__(self, model_path: str, weights_path: str):
        """
        Recommended files:
        model_path: `./model_weight/CNN_model.json`
        weights_path: `./model_weight/best_model1_weights.h5`
        """
        self.model = load_model(model_path, weights_path)
        self.encoder = create_encoder()

    def predict_emotion_from_wav_file(self, wav_file_path: str):
        start_time = time.time()
        try:
            res = get_predict_feat(wav_file_path)
            predictions = self.model.predict(res)
            logging.info(f"SER: pred: {predictions}, max: {np.argmax(predictions)}")
            y_pred = self.encoder.inverse_transform(predictions)
            inference_result = y_pred[0][0]
            logging.info(f"SER: inference result: {inference_result}")
            return inference_result
        finally:
            end_time = time.time()
            logging.info(f"SER: inference time: {end_time - start_time:.2f} seconds")
