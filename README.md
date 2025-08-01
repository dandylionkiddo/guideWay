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
1. pretrained checkpoint를 불러오는 경우 파라미터 개수 에러
2. 체크포인트 저장 디렉토리를 구분하지 않아 실험마다 체크포인트가 계속 갱신 (저장해둘 수 없음)
3. valid.log 파일에 로그 결과가 남지 않음 (체크포인트를 불러올 수 없다는 오류)
4. validation이나 test를 단독으로 진행하여 성능을 확인하는 코드가 없음
