# 5G IoT - OD_Framework Docs 1.0

본 문서는 5G IoT 과제를 지원하기 위한 객체 검출 모델 학습 프레임워크로

Poc 별 객체 검출 모델을 학습하기 위한 데이터 수집 및 학습 기능,

Jetson board와 같은 임베디드 보드에서 동작하기 위한 TensorRT 기반 모델 변환,

그리고 변환된 TensorRT engine을 활용한 영상 객체 검출 기능을 지원함으로써 

과제 수행을 위한 Poc별 모델 학습에 편의성을 제공한다. 

구버전 영문 readme 

.

[previous_readme](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/previous_readme%207a9f801e04284694b8e0c4f85c7a40e3.md)

## 시작하기

[프레임워크 셋팅 및 동작 방법 ](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/%E1%84%91%E1%85%B3%E1%84%85%E1%85%A6%E1%84%8B%E1%85%B5%E1%86%B7%E1%84%8B%E1%85%AF%E1%84%8F%E1%85%B3%20%E1%84%89%E1%85%A6%E1%86%BA%E1%84%90%E1%85%B5%E1%86%BC%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%83%E1%85%A9%E1%86%BC%E1%84%8C%E1%85%A1%E1%86%A8%20%E1%84%87%E1%85%A1%E1%86%BC%E1%84%87%E1%85%A5%E1%86%B8%206bb17222b11d4853872f66aba291f629.md)

## 데이터 수집 모듈

[OID_tools Package ](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/OID_tools%20Package%201127d76c2e924c319554de75d504b70f.md)

[COCO_tools Package](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/COCO_tools%20Package%2070c2eb25d4e34fa9b78739ebd6e4aab4.md)

## 모델 학습 및 테스트 관련 모듈

[weights](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/weights%206a82cf9230cc4bd9a8556a7381429acc.md)

[utils package ](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/utils%20package%20d716b552e77f4e4c9d0ea593c64133a0.md)

[Config directory ](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/Config%20directory%201f404e67c41d4349a18b4acf181baa31.md)

## Jetson board 동작을 위한 Detection Package

[Detection_tools package ](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/Detection_tools%20package%20e34919286f574563ab1bd87e0f5e3be8.md)

[trt_detection.py](5G%20IoT%20-%20OD_Framework%20Docs%201%200%20109f70eae5844c6d8d82fe66c6019225/trt_detection%20py%20820dff34d0834f7dabf41d5bafaf666e.md)