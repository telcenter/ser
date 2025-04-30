from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def create_encoder():
    encoder = OneHotEncoder()
    Y = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    print(encoder.categories_)
    return encoder

