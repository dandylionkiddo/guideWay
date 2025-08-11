# guideWay
---
* 데이터셋 저장 X, 코드 위주로 저장하기

## 시작 코드
<pre>conda create -n efficientvit python=3.10
conda activate efficientvit
pip install -U -r requirements.txt  </pre>

## 학습 실행 코드
<pre>PYTHONUTF8=1 python applications/efficientvit_seg/train.py \
  --config applications/efficientvit_seg/mapillary.yaml \
  --path output/seg_test_run  </pre>

## 현재 문제점
1. inference를 수행하는 코드의 코드 단일 책임 원칙 지키기
2. Mapillary 클래스 수 줄여서 학습 및 추론 가능한지 확인
