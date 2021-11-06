"""
이미지의 명도 분포 알아보기
영상은 색의 밝기로 표현이 됩니다.
OpenCV를 사용하면 명도가 0~255 사이의 값들로 표현이 됩니다.

우리는 특정 명도에 얼마나 색이 분포해있는지 알고 싶습니다. 
예를 들어 명도가 10인 화소는 영상에서 몇 개나 되는지 세어보고 
개수가 전체 화소의 90%에 해당한다면 
이 영상은 저조도에서 촬영된 영상이라는 추론을 할 수 있게 됩니다.

그래서 우리는 이번에 특정 명도에 해당하는 영상 화소 수를 세어 
명도별 영상 화소 수를 그래프로 그려보려 합니다.

흰색 바탕의 256 × 화소 수 크기의 numpy.ndarray를 만들고 
이 이 영상에 그래프를 그리는 함수를 작성해주세요.
----------------------------------------------------------------------------------------
지시사항
입력된 영상의 히스토그램을 반환하는 함수를 완성하세요.

Tips!
numpy.ones 함수로 원소값이 1인 행렬을 만든 뒤 255를 곱하면 
원소값이 255가 되어 백색 바탕 이미지를 만들 수 있습니다.
OpenCV의 cv2.line()함수를 이용하면 손 쉽게 라인을 그릴 수 있습니다.
전체 화소수는 image의 shape 변수를 이용해서 구할 수 있습니다. 
shape은 아래와 같기 때문에 튜플의 모든 원소값을 곱한 것이 
이미지가 가지고 있는 전체 화소 수가 됩니다.
shape = (row, column, channel)
예를 들어 명도가 128인 화소는 전체 이미지 중에 70번 등장했을 때 
아래 처럼 코드를 쓸 수 있습니다.

cv2.line(img, (total_pixel, 128), (total_pixel - 70, 128))
우리가 아는 그래프는 영상의 좌측하단 모서리가 (0, 0) 입니다. 
그러나 영상을 표현하는 행렬의 시작점은 좌측 상단이 (0, 0) 이 됨을 
유의하세요.
"""
import sys
import cv2
import numpy

from elice_utils import EliceUtils
elice_utils = EliceUtils()


# 영상을 입력으로 받아 영상의 히스토그램 영상을 리턴하는 함수를 만드세요.
def draw_histogram(img):
    total_pixel = img.shape[0] * img.shape[1]
    # 바탕 백색바탕 이미지
    histogram_img = numpy.ones((total_pixel, 256)) * 255

    histogram = [0 for i in range(256)]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[img[i, j]] += 1
    
    # intensity : 명도
    # frequency : 화소수
    for intensity, frequency in enumerate(histogram):
        start = (intensity, total_pixel - 1)
        end = (intensity, total_pixel - frequency - 1)
        cv2.line(histogram_img, start, end, (0), 2, cv2.LINE_AA)
    return histogram_img
    

if __name__ == "__main__":
    # 이미지를 읽어옵니다.
    img = cv2.imread("./night.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 결과를 확인합니다.
    histogram_img = draw_histogram(img)
    histogram_img = cv2.resize(histogram_img, (256, 256))
    
    cv2.imwrite('result.jpg', histogram_img) 
    elice_utils.send_image('result.jpg')