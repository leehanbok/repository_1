"""
회선처리
앞장에서는 OpenCV가 제공하는 필터 함수를 이용해 프리윗 필터를 적용해보았습니다.

이번에는 회선처리 함수를 직접 구현하여 프리윗 필터를 적용해보는 연습을 해보겠습니다.

구현의 편리함을 위해 아래와 같은 5 × 5 이미지가 있을 때, 경계선은 회선처리가 어려우므로 경계선을 제외한 내부 화소만 회선처리 하도록 합니다.

지시사항
주어진 커널을 입력 img에 회선 처리한 이미지 dst를 반환하는 함수 convolution2D()를 완성하세요.

Tips!
내부 화소만 순회하기 : 문제에서 회선처리를 쉽게 하기 위해 경계선 화소들은 제외를 합니다.
따라서 화소 순회를 할 때, 필터의 중점 좌표만큼 시작과 끝 범위를 정해주면 내부 경계선을 제외한 내부 화소만 순회가 가능합니다.

center_r = kernel.shape[0] // 2
center_c = kernel.shape[1] // 2

for i in range(center_r, img.shape[0] - center_r):
        for j in range(center_c, img.shape[1] = center_c):

4중 for 반복문이 사용됩니다. 
첫 번째와 두 번째 반복문은 영상의 화소 순회를 하는 데 쓰이고, 세 번째와 네 번째 반복문은 필터의 요소 순회를 하는 데 쓰입니다.
"""
import sys
import numpy
import cv2
from elice_utils import EliceUtils

elice_utils = EliceUtils()


def convolution2D(img, kernel):
    dst = img.copy()
    # 주어진 커널을 입력 img에 회선처리한 이미지 dst를 반환하는 함수를 완성하세요.
    center_r = kernel.shape[0] // 2
    center_c = kernel.shape[1] // 2

    for i in range(center_r, img.shape[0] - center_r):
        for j in range(center_c, img.shape[1] - center_c):
            I = 0
            for r in range(kernel.shape[0]):
                for c in range(kernel.shape[1]):
                    x = i + r  - center_r
                    # [0 0 0]
                    # [0 0 0]
                    # [0 0 0]
                    y = j + c - center_c
                    I += (img[x, y] * kernel[r, c])
            dst[i, j] = I

    return dst


def prewitt(img):
    vertical_kernel = numpy.array(
        [
            [ -1, 0, 1],
            [ -1, 0, 1],
            [-1, 0, 1]
        ]
    )
    
    horizontal_kernel = numpy.array(
        [
            [ -1, -1, -1],
            [ 0, 0, 0],
            [1, 1, 1]
        ]
    )

    # 프리윗 필터에 회선처리를 적용합니다.
    dst_vertical_edge = convolution2D(img, vertical_kernel)
    dst_horizontal_edge = convolution2D(img, horizontal_kernel)
    
    return dst_vertical_edge + dst_horizontal_edge


if __name__ == "__main__":
    # 이미지를 불러옵니다.
    img = cv2.imread("elice.png", cv2.IMREAD_GRAYSCALE)
    
    # 회선처리 함수를 적용한 프리윗 필터를 적용해봅니다.
    filtered_img = prewitt(img)
    cv2.imwrite("result.jpg", filtered_img)    
    elice_utils.send_image('result.jpg')
