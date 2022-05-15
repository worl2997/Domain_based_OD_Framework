# 5G IoT - OD_Framework Docs 1.0

본 문서는 5G IoT 과제를 지원하기 위한 객체 검출 모델 학습 프레임워크로

Poc 별 객체 검출 모델을 학습하기 위한 데이터 수집 및 학습 기능,

Jetson board와 같은 임베디드 보드에서 동작하기 위한 TensorRT 기반 모델 변환,

그리고 변환된 TensorRT engine을 활용한 영상 객체 검출 기능을 지원함으로써 

과제 수행을 위한 Poc별 모델 학습에 편의성을 제공한다  
(현재 프레임워크 셋팅 및 동작 방법 작성, 나머지 문서 추후 업로드 예정)
  
구버전 영문 readme -> 
[previous_readme](Documentation/previous_readme.md)

## 시작하기

[프레임워크 셋팅 및 동작 방법 ](Documentation/framework_setting_run.md)

## 데이터 수집 모듈

[OID_tools Package ](Documentation/OID_tools_package.md)

[COCO_tools Package](Documentation/COCO_tools_package.md)

## 모델 학습 및 테스트 관련 모듈

[weights](Documentation/weights.md)

[utils package ](Documentation/utils_package.md)

[Config directory ](Documentation/Config_directory.md)

## Jetson board 동작을 위한 Detection Package

[Detection_tools package ](Documentation/Detection_tools_package.md)

[trt_detection.py](Documentation/trt_detection.md)