"""
셔플링을 진행하여 모델 성능 개선하기
shuffle은 데이터셋을 랜덤으로 섞는 것을 의미합니다.

데이터 순서가 학습에 영항을 끼치지 않도록 데이터 순서를 섞어야 합니다. 매번 epoch 마다 데이터의 순서가 같다면 순서가 학습에 영향을 끼칠 수 있습니다.

0~9 까지의 숫자 데이터가 100개씩 10종류가 있을 때, batch를 10으로 하면 첫 번째 batch에서는 0만 100개, 두 번째 batch에서는 1만 100개 학습하게 됩니다. 이렇게 학습하는 것보다 데이터를 잘 섞어서(shuffle) 학습했을 때 더 높은 성능(낮은 loss)을 얻을 수 있습니다.

Keras에서 shuffle
Keras에서 shuffle은 fit()함수의 shuffle=옵션으로 기능을 활성화 할 수 있습니다.
shuffle=True가 되면 Epoch마다 데이터가 랜덤하게 섞이게 되고, shuffle=False가 되면 데이터가 섞이지 않습니다.

두 가지 옵션을 줘서 각각의 fit()을 해보고 loss를 비교하는 실습을 해봅니다. 코드의 빈 부분을 채우고, 결과를 확인해 보세요.
--------------------------------------------------------------------------
지시사항
results_no_shuffle 변수에 shuffle을 하지 않은 모델을 학습하여 저장하세요.

results_shuffle 변수에 shuffle을 한 모델을 학습하여 저장하세요.
"""
import os 
import cv2
import numpy
import matplotlib.pyplot as plt

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

train_cnt, test_cnt = 5000, 1000
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
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32, 32, 1)))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(120, kernel_size=(1, 1), strides=(5, 5), activation='tanh'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))


# 모델을 컴파일합니다.
model.compile(loss=losses.sparse_categorical_crossentropy, 
              optimizer=optimizers.Adam(),
              metrics=[metrics.categorical_accuracy])


# 모델을 학습하는 코드를 작성하세요. (shuffle을 하지 않습니다.)
results_no_shuffle = model.fit(test_images,  test_labels, epochs=5, shuffle=False)
no_shuffle_test_loss, no_shuffle_test_acc = model.evaluate(test_images, test_labels)


model.compile(loss=losses.sparse_categorical_crossentropy, 
              optimizer=optimizers.Adam(),
              metrics=[metrics.categorical_accuracy])

# 모델을 학습하는 코드를 작성하세요. (shuffle을 사용해 봅니다.)
results_shuffle = model.fit(test_images,  test_labels, epochs=5, shuffle=True)
shuffle_test_loss, shuffle_test_acc = model.evaluate(test_images, test_labels)

# 코드 작성 후 epoch별 loss를 비교한 결과를 확인해보세요.
print('No Shuffle loss :', [round(x, 3) for x in results_no_shuffle.history['loss']]) 
print('   Shuffle loss :', [round(x, 3) for x in results_shuffle.history['loss']])

# categorical_accuracy: 0.0280
# No Shuffle loss : [1.863, 1.39, 1.236, 1.133, 1.059]
#    Shuffle loss : [1.038, 0.988, 0.933, 0.899, 0.877] --> No Shuffle 보다 성능이 좋다.
# 1.0594766006469727 0.8774073152542115
print(results_no_shuffle.history['loss'][-1], results_shuffle.history['loss'][-1])



