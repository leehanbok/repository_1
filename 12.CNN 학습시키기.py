"""
CNN 학습시키기
앞선 실습에서 구현하였던 모델을 이제 학습을 시켜보려고 합니다.

Dateset 로드하기
keras는 저명한 데이터셋들을 다운로드하고 바로 로드할 수 있도록 해주는 datasets 모듈을 지원합니다.

앞 실습에서 소개된 MNIST 데이터셋을 로드하기 위해선 아래와 같은 코드를 쓰면 됩니다.
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

데이터 shape 변경
MNIST의 데이터는 모두 이진 이미지이기 때문에 image의 shape이 28×28 이 됩니다. 컨볼루션을 위해서는 채널수가 필요하므로 다음과 같이 이미지들을 reshaping 해야 합니다.
# 학습 셋은 60000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
train_images = train_images.reshape((60000, 28, 28, 1))

# 테스트 셋은 10000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
test_images = test_images.reshape((10000, 28, 28, 1))

데이터 정규화
서로 다른 범위의 명도값을 갖더라도 분류가 될 수 있게 하기 위해 정규화를 진행합니다.
# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

모델의 컴파일
모델에 optimizer와 손실 함수, 평가 지표를 설정합니다.
optimizer에선 GD, SGD, Adagrad, Adam 등이 있고 손실 함수는 MSE, categorical cross entropy 등이 있습니다.
마직막으로 모델의 평가지표는 accuracy, bianry accuracy 등등이 있습니다.
모델 컴파일 예시
# 모델을 컴파일 합니다.
model.compile(optimizer=optimizers.Adam(),
              loss=losses.sparse_categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

모델 학습 함수
model.fit(train_images, train_labels, epochs=1)

모델 평가 함수
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

모델 추론 함수
confidence = model.predict(test_img.reshape((1, row, col, channel)), verbose=2)

----------------------------------
지시사항
앞선 실습을 참고하여 모델 구조를 선언하세요.

모델을 학습하고, 평가하세요.

본 실습에는 Adam을 optimizer로
손실 함수는 sparse categorical crossentropy를, 평가 지표는 categorical_accuracy를 사용하세요.
모델을 학습하고 테스트 데이터 셋에 대한 loss값과 accuracy를 구해주세요.
학습 epoch는 1로 지정하세요.
학습과 테스트 데이터 수(train_cnt, test_cnt)를 조정하여 주어진 이미지 7.png를 입력으로 넣었을 때 해당 손글씨 이미지를 올바르게 분류하는지 모델의 예측 결과를 구하세요.
"""
import os
import cv2
import numpy
from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics

# mnist 데이터 셋을 로드합니다.
# 각각 학습셋(이미지, 라벨), 테스트 셋(이미지, 라벨)으로 구성이 되어 있습니다.
data_path = os.path.abspath("./mnist.npz")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)

# 학습과 테스트에 사용할 데이터의 수를 조정합니다.
# 참고, 원래 MNIST의 학습 이미지 셋은 60000개, 테스트 셋은 10000개 입니다.
# train_cnt, test_cnt = 5, 1  # 못맞춤
train_cnt, test_cnt = 60000, 10000  # 못맞춤
# train_cnt, test_cnt = 60000, 10000
train_images, train_labels = train_images[:train_cnt], train_labels[:train_cnt]
test_images, test_labels = test_images[:test_cnt], test_labels[:test_cnt]

# 학습 셋은 사용할 갯수만큼 28x28 이진 이미지이므로 reshaping을 해줍니다.
train_images = train_images.reshape((train_cnt, 28, 28, 1))

# 테스트 셋 역시 사용할 갯수만큼 28x28 이진 이미지이므로 reshaping을 해줍니다.
test_images = test_images.reshape((test_cnt, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델을 구조를 선언하세요.
# 모델 변수를 선언합니다.
model = models.Sequential()

# 모델에 첫 번째 입력 레이어를 추가합니다.
model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# 아래에 지시상항에 있는 모델 구조가 되도록 나머지 모델 구조를 선언해주세요.
model.add(layers.Convolution2D(64, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Convolution2D(64, (3, 3), activation=activations.relu))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 모델을 컴파일 합니다.
adam_optimizer = optimizers.Adam()
loss_function = losses.sparse_categorical_crossentropy
metric = metrics.categorical_accuracy
"""
구현부
"""
model.compile(optimizer=adam_optimizer,
              loss=loss_function,
              metrics=[metric])

# 모델을 학습데이터로 학습하세요.
model.fit(train_images, train_labels, epochs=1)

# 모델을 평가하세요.
test_loss, test_acc = model.evaluate(test_images, test_labels)

# 학습 결과를 출력합니다.
print("test_loss:", test_loss, "test_acc:", test_acc)

# 모델에 테스트 이미지를 넣고 예측값을 확인해봅니다.
test_img = cv2.imread("7.png", cv2.IMREAD_GRAYSCALE)

# 입력 이미지의 픽셀을 0~1 사이로 정규화 합니다.
test_img = test_img / 255.0
row, col, channel = test_img.shape[0], test_img.shape[1], 1
confidence = model.predict(test_img.reshape((1, row, col, channel)))

for i in range(confidence.shape[1]):
    print(f"{i} 일 확률 = {confidence[0][i]}")

print(f"정답은 : {numpy.argmax(confidence, axis=1)}")