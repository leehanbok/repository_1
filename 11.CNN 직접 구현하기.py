"""
CNN 직접 구현하기
MNIST 데이터셋
MNIST 데이터셋은 아래와 같이 숫자 0부터 9까지의 수를 손으로 쓴 28 × 28의 이진 이미지 데이터셋입니다.

60,000개의 학습 셋과 10,000개의 테스트셋이 있습니다.

우리는 이 데이터셋으로 우리만의 CNN을 학습 시켜 숫자 이미지를 분리하는 분류기를 만들고자 합니다.

케라스에는 모델을 쉽게 구현할 수 있는 models, layers와 활성화 함수들이 있는 activations 모듈들이 있습니다.
from tensorflow.keras import layers, models, activations
모델 생성 예시
아래의 코드들을 참고하여 지시상황에 제시된 모델 구조를 갖는 모델을 생성하고 모델의 구조를 출력하세요.
# 순차적으로 레이어가 쌓이는 모델 만들기
model = models.Sequential()

# 컨볼루션 레이어 만들기
model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))

# 풀링 적용하기
model.add(layers.MaxPooling2D((2, 2)))

# 1차원  텐서로 변환하기
model.add(layers.Flatten())

# FC레이어 만들기
model.add(layers.Dense(64, activation='relu'))

# 모델 구조 출력하기
model.summary()

--------------------------------------------------------------------
지시사항
다음은 간단한 CNN 구조 입니다.
그림 11.CNN_직접_구현하기.png 참조
keras의 models, layers, activations 모듈을 활용하여 위의 구조를 갖는 CNN을 만들고 모델의 구조를 summary()함수를 통해 출력하세요.

모든 Convolution 레이어의 activation함수는 relu를 사용합니다.

마지막에 Dense Layer는 첫 번째 레이어는 relu를 두 번째 레이어는 softmax를 쓰도록 합니다.

Convolution 레이어는 패딩을 주지 않습니다.

Convolution 레이어는 스트라이드를 주지 않습니다.

모든 Convolution 레이어의 커널 사이즈는 (3, 3)으로 합니다.
"""
from tensorflow.keras import datasets, layers, models, activations


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


# Model 구조를 출력합니다.
model.summary()