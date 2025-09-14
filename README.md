# guideWay
---

## Project Overview

This repository contains the implementation of the semantic segmentation module developed for [**Bedivere**](https://aidall.ai/), a walking-assistance robot for the visually impaired by [**Aidall**](https://aidall.ai/).  

We conducted experiments to enable real-time semantic segmentation on [**Jetson Orin Nano**](https://developer.nvidia.com/embedded/jetson-orin).  

This project is based on the official [**EfficientViT**](https://github.com/mit-han-lab/efficientvit) repository, from which we imported the segmentation-related components and implemented complete training and validation pipelines.  
In addition, for comparison, we fully re-implemented [**SegFormer**](https://github.com/NVlabs/SegFormer) from scratch to make both training and validation available within the same framework.  

---

## Environment Setup

<pre>conda create -n efficientvit python=3.10
conda activate efficientvit
pip install -U pip setuptools wheel
pip install -U "Cython>=0.29.36"
pip install -U -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH  </pre>

---

## Training

### EfficientViT
<pre>PYTHONUTF8=1 python applications/efficientvit_seg/train.py \
  --config applications/efficientvit_seg/custom_seg.yaml \
  --path output/seg_test_run \
  --arch efficientvit </pre>
  
### SegFormer
<pre>PYTHONUTF8=1 python applications/efficientvit_seg/train.py \
  --config applications/efficientvit_seg/custom_seg.yaml \
  --path output/seg_test_run \
  --arch segformer </pre>

---

## Evaluation

### EfficientViT
<pre>python applications/efficientvit_seg/eval.py \
  --config applications/efficientvit_seg/eval_config.yaml \
  --arch efficientvit </pre>

### SegFormer
<pre>PYTHONUTF8=1 python applications/efficientvit_seg/train.py \
  --eval_only --eval_checkpoint output/final_models/segformer-b0.pt \
  --config applications/efficientvit_seg/custom_seg.yaml \
  --arch segformer --path ./output/evaluation_results \
  --save_eval_results </pre>

---
