"""
Average Precision 평가 코드 구현
클래스가 1개이고 올바르게 검출된 물체의 총 개수가 15개인 어떤 데이터셋이 있습니다.

문제에서 주어진 변수 detection_results는 어떤 객체 감지 알고리즘을 돌려 나온 결과입니다.

각각의 행이 하나의 감지된 물체를 나타내며
첫 번째 열의 값은 다음을 의미합니다.
- 1이면 올바르게 검출되고 분류된 경우를 말합니다. 즉 True Positive입니다.
- 0이면 물체를 검출되었다고 생각했으나 물체가 아닌 경우를 의미합니다.

두 번째 열은 클래스에 속할 확률입니다.
예를 들어, 아래와 같이 입력이 주어졌다면
감지된 물체는 총 10개입니다.  --> 5개는 빠졌다?? TN ? -> Threshold ?
detection_results = numpy.array([
    [1, 0.95],
    [1, 0.91],
    [1, 0.85],
    [1, 0.81],
    [1, 0.78],
    [0, 0.68],
    [1, 0.57],
    [1, 0.45],
    [0, 0.43],
    [0, 0.13],
])
이 알고리즘의 성능을 평가하기 위한 average_precision() 함수를 구현하고 average precision값을 구하세요.
-------------------
지시사항
detection_results 변수에는 클래스가 1개이고 검출된 물체의 총 개수가 15개인 어떤 객체 감지 알고리즘의 결과가 들어있습니다.

첫 번째 열의 값은 다음을 의미합니다.

1이면 올바르게 검출되고 분류된 경우를 말합니다. (True Positive)
0이면 물체를 검출되었다고 생각했으나 물체가 아닌 경우를 말합니다. (False Positive)
두 번째 열은 클래스에 속할 확률입니다.

이 함수가 average precision을 계산해서 반환하도록 average_precision() 함수를 구현하세요.
--------------------
Tips!
Recall과 Precision 구하기
average_precision 함수는 입력으로 detection_results와 Ground Truth의 개수를 입력으로 받습니다. 
즉 Recall의 분모가 Ground Truth가 됩니다.

객체 감지 알고리즘에서 Recall과 Precision
-. Recall : 원래 맞춰야 되는 정답 중에 알고리즘이 맞춘 정답 비율 (정답율 ?)  =  7/ 15
-. Precision : 알고리즘이 맞춘 것들 중 실제 정답 비율  (내가 얼마나 잘하나 ?) = TP / 10
그리고, Precision의 분모는 detection_results 에 담긴 데이터 행 개수가 됩니다.
"""
import numpy

# average precision을 반환하는 함수를 완성하세요.
def average_precision(detection_results, ground_truth):
    ap = 0

    for i in range(detection_results.shape[0]):
        threshold = detection_results[i, 1]

        detected = detection_results[numpy.where(detection_results[:,1] >= threshold)]

        TP = detected[numpy.where(detected[:,0] == 1)]

        # precision = TP / 알고리즘이 맞췄다고 한거
        # recall = TP / ground_truth
        precision = TP.shape[0] / detected.shape[0]
        recall = TP.shape[0] / ground_truth
        print(f"precision = {TP.shape[0]} / {detected.shape[0]}")
        ap += (precision) * recall

    return ap


if __name__ == "__main__":
    detection_results = numpy.array([
        [1, 0.95],
        [1, 0.91],
        [1, 0.85],
        [1, 0.81],
        [1, 0.78],
        [0, 0.68],
        [1, 0.57],
        [1, 0.45],
        [0, 0.43],
        [0, 0.13],
    ])
    
    # 정의한 함수를 호출한 결과를 소수점 다섯째 자리에서 반올림하여 확인합니다.
    ap = average_precision(detection_results, 15)
    print(round(ap, 4))
