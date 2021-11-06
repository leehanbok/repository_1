"""
이진화 응용 : 컬러 영상에서 특정 색상영역만 추출하기
앞선 실습에서는 영상을 단순히 임곗값을 가지고 아래처럼 0 또는 255 명도로 사상하였습니다.
img[i, j] = 255 if img[i, j] >= threshold else 0

이를 응용해서 컬러 영상에서 특정 색의 영역대는 명도를 255로 만들고 영역 밖의 색은 명도를 0으로 표시하려고 합니다.

예를 들어 R 채널에서 100 ~ 200 구간에 있는 명도만 살리고 싶은 경우 아래와 같이 표현할 수 있습니다.
img[i, j, 0] = 255 if img[i, j, 0] >= range_start and img[i, j, 0] <= range_end else 0

임곗값으로 아래와 같이 주어질 때, 특정 색상영역을 추출하는 실습을 해봅니다.
입력 : [50, 255, 10, 100, 200, 255]

지시사항
컬러 이미지에서 특정 색상영역을 추출한 결과를 반환하는 함수를 완성하세요.

주어지는 입력 [50, 255, 10, 100, 200, 255]은 다음을 의미합니다.

R채널에서는 [50, 255] 사이의 픽셀은 255로, 범위 밖의 명도는 0으로 표시합니다.
G채널에서는 [10, 200] 사이의 픽셀은 255로, 범위 밖의 명도는 0으로 표시합니다.
B채널에서는 [200, 255] 사이의 픽셀은 255로, 범위 밖의 명도는 0으로 표시합니다.
주어지는 임계값에 따라 특정 색상영역을 추출한 이미지를 반환하세요.


"""
import sys
import numpy
import cv2
from elice_utils import EliceUtils

elice_utils = EliceUtils()

# img에서 특정 색상영역을 추출한 결과를 반환하는 함수를 완성하세요.
def inRange(img, thresholds):
    """
    thresholds는 컬러 채널별 살리고 싶은 명도의 범위가 들어있습니다.
    예를 들어 아래와 같이 입력이 들어온 경우,
    thresholds = [50, 255, 10, 100, 200, 255]
    
    구간 경계선을 포함하여,
    R채널에서는 [50, 255]사이의 픽셀은 255로,범위 밖의 명도는 0으로 표시합니다.
    G채널에서는 [10, 200]사이의 픽셀은 255로,범위 밖의 명도는 0으로 표시합니다.
    B채널에서는 [200, 255]사이의 픽셀은 255로,범위 밖의 명도는 0으로 표시합니다.
    """
    
    for  c in range(img.shape[2]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                I = img[i, j, c]
                start = thresholds[c *2]
                end = thresholds[ c* 2 + 1]

                if start <= I and I <= end:
                    new_I = 255
                else:
                    new_I = 0
                
                img[i, j, c] = new_I

    return img


if __name__ == "__main__":
    # 컬러 채널별 살리고 싶은 명도의 범위가 주어집니다.
    # thresholds = list(map(int, sys.argv[1].split(",")))
    thresholds = [50, 255, 10, 100, 200, 255]
    
    # 컬러 이미지를 불러옵니다.
    img = cv2.imread("elice.png", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 특정 색상영역을 추출한 결과를 확인해봅니다.
    colorFiltered = inRange(img, thresholds)
    colorFiltered = cv2.cvtColor(colorFiltered, cv2.COLOR_BGR2RGB)
    cv2.imwrite("result.jpg", colorFiltered)
    elice_utils.send_image('result.jpg')
