"""
프리윗 필터
OpenCV에는 필터를 영상에 쉽게 적용해주는 함수가 있습니다.

cv2.filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None) -> dst

• src : 입력 영상
• ddepth : 출력 영상 데이터 타입. (e.g) cv2.CV_8U, cv2.CV_32F, cv2.CV_64F, -1을 지정하면 src와 같은 타입의 dst 영상을 생성합니다.
• kernel : 필터 마스크 행렬. 실수형.
• anchor: 고정점 위치. (-1, -1)이면 필터 중앙을 고정점으로 사용
• delta : 추가적으로 더할 값
• borderType : 가장자리 픽셀 확장 방식
• dst : 출력 영상

이 함수를 이용하여 프리윗 필터를 영상에 적용시켜 봅시다.
--------------------------------------------------
지시사항
prewitt() 함수를 완성하세요.

dst_vertical_edge 변수에 수직 프리윗 필터를 적용하세요.

dst_horizontal_edge 변수에 수평 프리윗 필터를 적용하세요.

img에 수직/수평 프리윗 필터를 적용시킨 두 이미지 합을 반환하세요.


"""
import sys
import numpy
import cv2

from elice_utils import EliceUtils
elice_utils = EliceUtils()


def prewitt(img):
    vk = numpy.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]
    )

    hk = numpy.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ]
    )
    # dst_vertical_edge 변수에 수직 프리윗 필터를 적용하세요.
    dst_vertical_edge = cv2.filter2D(img, -1, vk)
    
    # dst_horizontal_edge 변수에 수평 프리윗 필터를 적용하세요.
    dst_horizontal_edge = cv2.filter2D(img, -1, hk)
    
    
    # 결과 이미지는 수직/수평 프리윗 필터를 적용시킨 두 이미지의 합을 반환하세요.
    return dst_vertical_edge + dst_horizontal_edge


if __name__ == "__main__":
    # 이미지를 읽어옵니다.
    img = cv2.imread("elice.png", cv2.IMREAD_GRAYSCALE)
    
    # 프리윗 피터를 적용한 결과를 확인해봅니다.
    filtered_img = prewitt(img)
    cv2.imwrite("result.jpg", filtered_img)
    elice_utils.send_image('result.jpg')
