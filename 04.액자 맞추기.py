"""
액자 맞추기
어떤 액자 퍼즐이 있습니다. 
이 퍼즐은 2x2 조각으로 구성이 되어있습니다.

퍼즐 조각은 아래처럼 2x2 조각으로 나눠져 있고 
각 조각의 순서는 아래와 같습니다.

주어진 piece_order 리스트는 
현재 액자의 조각이 맞춰져 있는 순서를 나타내는 리스트입니다.

예를 들어 piece_order = [3, 2, 1, 0] 이라면 
아래와 같은 의미를 나타냅니다.

0번 조각이 위치해야 할 곳에 3번 조각이 있음
1번 조각이 위치해야 할 곳에 2번 조각이 있음
2번 조각이 위치해야 할 곳에 1번 조각이 있음
3번 조각이 위치해야 할 곳에 0번 조각이 있음
액자 퍼즐을 원래대로 맞춰주세요.
----------------------------------------------------------------------------------------
지시사항
주어진 영상과 조각 순서를 보고 원본 영상으로 복원한 영상을 
반환하는 함수를 구현하세요.

Tips!
조각 순서에 따라 현재 위치와 해당 조각이 이동해야 할 위치로 나눠서 
생각해보세요.
인덱스를 나눴다면 이제 인덱스를 입력받아 조각의 위치를 조정하고, 
조정된 이미지를 반환하는 함수를 구현해서 문제를 해결할 수 있습니다.
"""
import sys
import cv2
from elice_utils import EliceUtils
import os

elice_utils = EliceUtils()


# 주어진 영상과 조각 순서를 보고 원본 영상(출력 예시 참고)으로 
# 복원한 영상을 반환하는 함수를 구현하세요.
def solve_puzzle(img, piece_order):
    # print(piece_order)   # 3,2,1,0
    for to_idx, from_idx in enumerate(piece_order):
        if to_idx == from_idx:
            continue
        
        proper_idx = piece_order.index(to_idx)
        piece_order[to_idx] = piece_order[proper_idx]
        piece_order[proper_idx] = from_idx
    
    #print(piece_order)  # [0, 1, 2, 3]
    
        # row , column
        r, c= img.shape[0] // 2, img.shape[1] // 2
        # [0 1
        #  2 3]
        # -->
        # [ (0, 0) (0, 1)
        #   (1, 0) (1, 1)]
        p1 = tuple(map(lambda x: x*r, divmod(from_idx, 2)))
        p2 = tuple(map(lambda x: x*r, divmod(to_idx, 2)))
    
        tmp = img[p1[0]:p1[0] + r, p1[1]:p1[1] + c].copy()
        img[p1[0]:p1[0] + r, p1[1]:p1[1] + c] = img[p2[0]:p2[0] + r, p2[1]:p2[1] + c]
        img[p2[0]:p2[0] + r, p2[1]:p2[1] + c] = tmp[:]

    return img
    

if __name__ == "__main__":
    
    #C:\Users\LHB\Desktop\image_processing
    #print(os.getcwd())
    #C:\Users\LHB\Desktop\image_processing
    #os.chdir("C:/Users/LHB/Desktop/image_processing")
    #print(os.getcwd())

    # print(sys.argv[1])
    # list(sys.argv[1].split(','))
    #piece_order = list(map(int, list(sys.argv[1].split(','))))
    piece_order = [3, 2, 1, 0]  # 입력받아야 됨.

    
    # 이미지를 불러옵니다.
    img = cv2.imread("./puzzle.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 복원 결과를 확인합니다.
    result_img = solve_puzzle(img, piece_order)
    cv2.imwrite('result.jpg', result_img) 
    elice_utils.send_image('result.jpg')