"""
LeNet 직접 구현하기
LeNet은 CNN을 가장 처음 도입한 적용한 모델입니다.
LeNet의 경우 LeNet-1부터 LeNet-5까지 다양한 버전으로 존재합니다. LeNet-5를 직접 구현하고 이전 실습의 CNN과 어떤 점이 다른지 확인해보세요.

지시사항
아래 그림은 LeNet의 구조 입니다. 13.LeNet_직접_구현하기.png

아래 표는 LeNet의 각 레이어별 커널, 스트라이드 사이즈와 사용되는 활성화 함수가 적인 표 입니다.
13.LeNet_직접_구현하기_표.png 참고
위 두 그림을 보고 LeNet을 구현하세요.

Tips!
이전에 만든 CNN 모델과 비슷하지만 다른 부분이 있으니 모델 구조에 유의해서 작성해 주세요.
"""
import logging, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import tf
logger = tf.get_logger()
logger.setLevel(logging.FATAL)

import keras
from tensorflow.keras import datasets, layers, models, activations, utils

# 모델 변수를 선언합니다.
model = models.Sequential()
# 모델에 첫번째 입력 레이어를 추가합니다.
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32, 32, 1)))
model.add(layers.AveragePooling2D((2,2), strides=(2,2)))

# 아래에 지시상항에 있는 모델 구조가 되도록 나머지 모델 구조를 선언해주세요.
model.add(layers.Conv2D(16, (5,5), strides=(1,1), activation='tanh'))
model.add(layers.AveragePooling2D((2,2), strides=(2,2)))
model.add(layers.Conv2D(120, (5,5), strides=(1,1), activation='tanh'))

model.add(layers.Flatten())
model.add(layers.Dense(84, 'tanh'))
model.add(layers.Dense(10, 'softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD')

# Model 구조 확인
model.summary()