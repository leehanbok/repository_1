"""
액자 맞추기
어떤 액자 퍼즐이 있습니다. 이 퍼즐은 2x2 조각으로 구성이 되어있습니다.

퍼즐 조각은 아래처럼 2x2 조각으로 나눠져 있고 각 조각의 순서는 아래와 같습니다.

주어진 piece_order 리스트는 현재 액자의 조각이 맞춰져 있는 순서를 나타내는 리스트입니다.

예를 들어 piece_order = [3, 2, 1, 0] 이라면 아래와 같은 의미를 나타냅니다.

0번 조각이 위치해야 할 곳에 3번 조각이 있음
1번 조각이 위치해야 할 곳에 2번 조각이 있음
2번 조각이 위치해야 할 곳에 1번 조각이 있음
3번 조각이 위치해야 할 곳에 0번 조각이 있음
액자 퍼즐을 원래대로 맞춰주세요.
----------------------------------------------------------------------------------------
지시사항
주어진 영상과 조각 순서를 보고 원본 영상으로 복원한 영상을 반환하는 함수를 구현하세요.

Tips!
조각 순서에 따라 현재 위치와 해당 조각이 이동해야 할 위치로 나눠서 생각해보세요.
인덱스를 나눴다면 이제 인덱스를 입력받아 조각의 위치를 조정하고, 조정된 이미지를 반환하는 함수를 구현해서 문제를 해결할 수 있습니다.
"""
import sys
import cv2
from elice_utils import EliceUtils

elice_utils = EliceUtils()


# 주어진 영상과 조각 순서를 보고 원본 영상(출력 예시 참고)으로 복원한 영상을 반환하는 함수를 구현하세요.
def solve_puzzle(img, piece_order):

    return img
    

if __name__ == "__main__":
    # 퍼즐의 순서가 리스트 형태로 입력됩니다.
    piece_order = list(map(int, list(sys.argv[1].split(','))))
    
    # 이미지를 불러옵니다.
    img = cv2.imread("./puzzle.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 복원 결과를 확인합니다.
    result_img = solve_puzzle(img, piece_order)
    cv2.imwrite('result.jpg', result_img) 
    elice_utils.send_image('result.jpg')