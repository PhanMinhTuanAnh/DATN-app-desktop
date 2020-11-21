import tensorflow as tf
import sys
import cv2
import numpy as np

sys.path.append("..")


def predict_all(img, kanji_model, hiragana_model=None, katakana_model=None):
    if(hiragana_model==None and kanji_model==None and katakana_model==None):
        return None
    img = cv2.imread('img.jpg')
    img = cv2.resize(img,(48,48))
    cv2.imwrite('img2.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('img3.jpg', img)
    img = img.reshape(1,48,48,1)
    img = np.array(img,dtype='float') / 255.0
    # predicted_hiragana = np.argmax(hiragana_model.predict(img))
    # return predicted_hiragana
    predicted_kanji = np.argmax(kanji_model.predict(img))
    return predicted_kanji


# def load_hiragana_model():
#     hiragana = tf.keras.models.load_model("ji-trained-models/hiragana8B(1).h5")
#     return hiragana

def load_kanji_model():
    hiragana = tf.keras.models.load_model("ji-trained-models/kanji9B(2).h5")
    return hiragana
