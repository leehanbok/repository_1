"""
배치 정규화를 진행하여 모델 성능 개선하기
배치 정규화는 신경망의 깊이가 깊을수록 원래 의도했던 데이터 분포가 달라지는 것을 막아주는 효과가 있습니다.

네트워크의 깊이가 깊어 질수록 데이터 분포가 치우치는 현상이 더 많이 발생하며, 배치 정규화를 사용했을 때 일반적으로 성능이 좋아집니다.

최근에 개발되는 네트워크에는 대부분 배치 정규화를 사용하고 있습니다.

주어진 모델은 LeNet의 신경망 깊이를 늘린 변형 모델입니다.

이 변형된 LeNet에 배치 정규화를 추가해보고 어떤 효과가 있는지 확인해봅시다.
--------------------------------------
배치 정규화
BN이라고도 불리며 Keras에서 다음과 같이 레이어를 사용할 수 있습니다.

layers.BatchNormalization()

지시사항
앞서 구현한 LeNet에 아래의 구조와 같은 배치 정규화를 추가하세요. 그리고 test_loss와 test_acc를 확인하세요.
image
Tips!
train_cnt와 test_cnt가 동일한 상태에서 배치 정규화를 추가하기 전과 후의 test_loss와 test_acc를 비교해보세요.
train_cnt, test_cnt = 50000, 10000 일 때 정상적으로 정답 처리가 되는 것에 유의하세요.
"""
import os 
import cv2
import numpy

# Fix seed
import tensorflow as tf
tf.random.set_seed(1)
import numpy as np
np.random.seed(1)

from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics
from tensorflow.keras import utils

# mnist 데이터 셋을 로드합니다.
# 각각 학습셋(이미지, 라벨), 테스트 셋(이미지, 라벨)으로 구성이 되어 있습니다.
data_path = os.path.abspath("./mnist.npz")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)

train_cnt, test_cnt = 50000, 10000
train_images, train_labels = train_images[:train_cnt], train_labels[:train_cnt]
test_images, test_labels = test_images[:test_cnt], test_labels[:test_cnt]

# 학습 셋은 60000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
train_images = train_images.reshape((train_cnt, 28, 28, 1))

# 테스트 셋은 10000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
test_images = test_images.reshape((test_cnt, 28, 28, 1))

# LeNet의 입력은 32x32 이미지 입니다. 패딩을 주어서 28 x 28에서 32 x 32 이미지로 만듭니다.
train_images = numpy.pad(train_images, [[0, 0], [2,2], [2,2], [0,0]], 'constant')
test_images = numpy.pad(test_images, [[0, 0], [2,2], [2,2], [0,0]], 'constant')
print('train_images :', train_images.shape, type(train_images))
print('test_images :', test_images.shape, type(test_images))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델을 구조를 선언합니다.
model = models.Sequential()
model.add(layers.Conv2D(6, (5,5), strides=(1,1), activation='tanh'))
# model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D((2,2), strides=(2,2)))

model.add(layers.Conv2D(16, (5,5), strides=(1,1), activation='tanh'))
# model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D((2,2), strides=(2,2)))

model.add(layers.Conv2D(120, (5,5), strides=(1,1), activation='tanh'))
# model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(84, 'tanh'))
model.add(layers.Dense(10, 'softmax'))

# 모델을 컴파일 합니다.
model.compile(loss=losses.sparse_categorical_crossentropy, 
              optimizer=optimizers.Adam(),
              metrics=[metrics.categorical_accuracy])

# 모델을 학습합니다.
model.fit(train_images, train_labels, epochs=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)

# 모델에 테스트 이미지를 넣고 예측값을 확인해봅니다.
test_img = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)

# 입력 이미지의 픽셀을 0~1 사이로 정규화 합니다.
test_img = test_img / 255.0
row, col, channel = test_img.shape[0], test_img.shape[1], 1
confidence = model.predict(test_img.reshape((1, row, col, channel)))

for i in range(confidence.shape[1]):
    print(f"{i} 일 확률 = {confidence[0][i]}")

# 학습 결과를 출력합니다.아래 내용을 수정하면 채점이 되지 않습니다.
# 배치 미포함시 : [2] 0.11 0.1 -   loss: 0.0569 - categorical_accuracy: 0.0988
# 배치 포함시   : [2] 0.07 0.1 - - loss: 0.0365 - categorical_accuracy: 0.0994
print(numpy.argmax(confidence, axis=1), round(test_loss, 2), round(test_acc, 2))