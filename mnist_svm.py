import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

##########데이터 로드

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

labels = ['짝수', '홀수']

##########데이터 분석

print(x_train.shape) #(60000, 28, 28)
print(y_train.shape) #(60000,)

##########데이터 전처리

(x_train, y_train), (x_test, y_test) = (x_train[:100], y_train[:100]), (x_test[:100], y_test[:100]) #데이터를 100개로 제한

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
x_train = x_train / 255 #최소 최대 정규화 ([0, 1])
x_test = x_test / 255 #최소 최대 정규화 ([0, 1])

y_train = y_train % 2 #짝수는 0, 홀수는 1로 변환
y_test = y_test % 2

##########모델 생성

model = SVC(kernel='rbf', C=1.0, gamma='auto', confidence=True)

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #

print(model.score(x_test, y_test)) #0.5

##########모델 예측

image = Image.open('0.png')
image = image.convert('L') #'L': greyscale, '1': 이진화, 'RGB' , 'RGBA', 'CMYK'
image = image.resize((28, 28))
image = np.array(image) #이미지 타입을 넘파이 타입으로 변환
image = image.reshape((28 * 28,))
x_test = np.array(image)
x_test = 255 - x_test #흰색 반전 
x_test = x_test / 255 #최소 최대 정규화 ([0, 1])

y_predict = model.predict(x_test)
print(labels[y_predict[0]]) #
y_predict = model.predict(x_test)
label = labels[y_predict[0]]
y_predict = model.predict_proba(x_test)
confidence = y_predict[0][y_predict[0].argmax()]

print(label, confidence) #