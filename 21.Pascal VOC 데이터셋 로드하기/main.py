"""
Pascal VOC 데이터셋 로드하기
PASCAL VOC(Visual Object Classes) 데이터셋은 다음과 같은 구조로 되어있습니다.

VOC2007/
├── images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── 000003.jpg
└── labels
    ├── 000001.xml
    ├── 000002.xml
    └── 000003.xml

위 구조의 데이터셋을 정해진 포맷대로 읽을 수 있도록 dataset.py 파일에 있는 데이터 로더를 만드세요.
Python 코드 예시
# in_image를 (200, 200)으로 resize 해서 out_img에 저장합니다.
out_img = cv2.resize(in_img, (200, 200))

# (5, 10, 15)크기의 0으로 채워진 벡터를 생성합니다.
label_matrix = numpy.zeros([5, 10, 15])

--------------------------------------------
지시사항
dataset.py 내 voc_load_data() 함수를 완성하세요. 아래의 image format과 label format에 맞춰 코드를 반환하세요.

---------------------
image format

이미지 크기는 (448, 448, 3)으로 하고 값은 0~1 사이의 소수로 합니다.

예를 들어 이미지 크기가 (700 × 700)일 때, 이 이미지는 (100 × 100) 크기의 셀 (7 × 7) 그리드로 나눠집니다.

원본 이미지에서 (50, 50) 좌표는 (0, 0) 번째 그리드에서 (0.5, 0.5) 좌표가 됩니다.
원본 이미지에서 (130, 220) 좌표는 (1, 2) 번째 그리드에서 (0.3, 0.2) 좌표가 됩니다.
이 점에 유의해서 xml에 있는 좌표를 데이터 포맷에 맞게 변환하세요.
---------------------
label format

label 크기는 (7, 7, 25)로 하고 상세한 포맷은 다음을 따릅니다.

(7 × 7)은 이미지에 대한 그리드 셀 위치입니다.
(25) 벡터 index별 데이터는 아래와 같습니다.
0~19 : class id,
class id가 0이면 (1, 0, 0, 0, …)
class id가 2이면 (0, 0, 1, 0, …)
20 : x, 셀 내에서의 x좌표 (0~1)
21 : y, 셀 내에서의 y좌표 (0~1)
22 : w, 셀 내에서의 w크기 (0~1)
23 : h, 셀 내에서의 h크기(0~1)
24 : Pbject가 있는지 여부 (0 or 1)

"""
import tensorflow
from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics, utils
import model
import loss
import dataset

if __name__ == "__main__":

    train_images, train_labels = dataset.voc_load_data("./VOC2007/images", "./VOC2007/labels")    

    yolo = model.create_yolo()

    with tensorflow.device("/cpu:0"):
        yolo.compile(optimizer=optimizers.Adam(), loss=loss.yolo_loss)
        yolo.fit(train_images, train_labels, epochs=1, verbose=2)
        result = yolo.evaluate(train_images, train_labels)
        print(result)

    # 수정하지 마세요. 채점에 사용되는 코드입니다.
    print(train_images[0, :, :, :].sum())
    print(train_labels[0, :, :, :].sum())
