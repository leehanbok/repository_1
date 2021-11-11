"""
모델 레이어 만들기

앞장에서 구현했던 모든 것들을 종합하여 YOLO를 구현하고 직접 구현한 YOLO를 학습하여 물체 감지를 하는 것이 4장의 최종목표입니다.
그 첫 번째 단계로 구조도를 보고 직접 구현해보도록 합시다.
YOLO 모델 구조

Leaky Relu의 수식과 그래프
Leaky Relu란 Relu 함수에서 x가 음수 영역에서 그래디언트가 0.01씩 증가하도록 약간의 변형을 가한 함수입니다. 음수 영역에서 그래디언트가 무조건 0이 된다는 단점을 극복하기 위해 고안되었습니다.
image

YOLO는 컨볼루션 레이어의 활성화 함수로 Leaky Relu 함수를 채택하고 있습니다.

Keras에서 Leaky Relu
아래의 그림은 Keras의 Relu 코드 내부를 캡처한 것입니다. Leaky Relu를 사용하고 싶다면 파라미터로 alpha 값은 0.1을 주면 Leaky Relu가 됩니다. 이를 참고해서 모델을 구현해주세요.

------------------------------
지시사항
YOLO의 그래프 구조를 그래프로 그린 yolo.png를 참조하여 model.py에서 YOLO 모델을 구현하세요.

keras에는 LeakyRelu를 쉽게 activation으로 사용할 수 있는 layers.LeakyReLU 가 있습니다.
이 레이어를 컨볼루션 레이어 다음에 넣어 활성화 함수처럼 동작하도록 하세요.

사용 예시
```python

self.add(layers.Convolution2D(64, (7, 7), strides=(2, 2), input_shape=(448, 448, 3), padding=’same’))
self.add(layers.LeakyReLU(alpha=0.1))
```

Tips!
YOLO 모델 구조를 살펴보면 Block 2의 Conv Layer는 영상과 달리 strides가 없는 것에 유의하시기를 바랍니다.
"""
import logging, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import tf
logger = tf.get_logger()
logger.setLevel(logging.FATAL)

#import model


from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics

def create_yolo():
    model = models.Sequential()
    
    # Block1
    model.add(layers.Convolution2D(64, (7, 7), strides=(2, 2), input_shape=(448, 448, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Block2
    model.add(layers.Convolution2D(192, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))    
    
    """
    이곳에 모델을 레이어를 추가하는 코드를 구현하세요.
    모델의 구조는 문제의 설명의 모델 구조도를 참고 하세요.    
    """
    
    # Block3
    model.add(layers.Convolution2D(128, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    
    # Block4
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    """
    이곳에 모델을 레이어를 추가하는 코드를 구현하세요.
    모델의 구조는 문제의 설명의 모델 구조도를 참고 하세요.
    """
    
    # Block5
    model.add(layers.Convolution2D(512, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same', strides=(2,2)))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    # Block6
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    # Last Block
    model.add(layers.Flatten())
    model.add(layers.Dense(4096))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7 * 7 * 30))
    model.add(layers.Reshape(target_shape=(7, 7, 30)))
    
    return model


if __name__ == "__main__":
    yolo = model.create_yolo()
    print(yolo.summary())