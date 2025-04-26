import librosa
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def extract_features(data, sr=22050, n_mfcc=40, frame_length=2048, hop_length=512):
    """Extracts enough features to get 2376 dimensions"""
    features = []
    
    # 1. Basic features (13-20 dims)
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    
    features.extend([
        np.mean(zcr),
        np.mean(rms),
        np.mean(spectral_centroid)
    ])
    
    # 2. MFCCs (40 coeffs × 60 frames = 2400 dims)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    mfccs_processed = mfccs.flatten()[:2373]  # Truncate to get exact 2376
    
    features.extend(mfccs_processed)
    
    return np.array(features)[:2376]  # Ensure exactly 2376 features

def get_predict_feat(path):
    with open('./model_weight/scaler2.pickle', 'rb') as f:
        scaler2 = pickle.load(f)
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6, sr=22050)
    
    # Extract features
    features = extract_features(d, sr=s_rate)
    
    # Reshape for model
    result = np.reshape(features, (1, 2376))
    
    # If you have a scaler

    result = scaler2.transform(result)
    
    return np.expand_dims(result, axis=2)  # Shape: (1, 2376, 1)
