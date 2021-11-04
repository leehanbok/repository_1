"""
이진 영상 만들기
이진 영상(Binary image)이란 영상의 명도가 0 또는 255로만 표현되는 영상을 의미합니다.

아래 영상처럼 번호판, 문서 등등 전경과 배경을 확실하게 분리해주는 효과가 있기 때문에 광학문자인식(OCR) 기술에서 전처리로 많이 쓰입니다. 예를 들어 아래와 같이 이진화한다면, 숫자가 더욱 뚜렷하게 구분됩니다.

이때 명도를 0 또는 255로 가르는 기준을 임곗값(Threshold)라고 합니다. 명도가 임곗값 이상이면 255로, 미만이면 0으로 변환됩니다.

주어진 실습 영상을 이진 영상으로 바꾸어보세요.
-------------------------
지시사항
주어진 임곗값을 기준으로 이진 이미지를 반환하는 함수를 구현하세요.
Tips
앞서 화소를 변경했던 방법을 떠올리며 문제를 해결하세요.
"""
import sys
import numpy
import cv2
from elice_utils import EliceUtils

elice_utils = EliceUtils()


def binarization(img, threshold):
    # img를 이진이미지로 만든 뒤, 이진 이미지를 반환하는 함수를 구현하세요.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 255 if img[i,j] >= threshold else  0
    return img


if __name__ == "__main__":
    #threshold = int(sys.argv[1])
    threshold = 200
    img = cv2.imread("elice.jpg", cv2.IMREAD_GRAYSCALE)
    
    binary_img = binarization(img, threshold)
    
    cv2.imwrite("result.jpg", binary_img)
    elice_utils.send_image('result.jpg')
