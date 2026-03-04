
# 🚶‍♂️ guideWay: Real-Time Semantic Segmentation for Walking-Assistance Robot

<br>

## 📌 Project Overview

**guideWay** is the semantic segmentation module developed for [**Bedivere**](https://aidall.ai/), a walking-assistance robot for the visually impaired by [**Aidall**](https://aidall.ai/).

We conducted experiments to enable real-time semantic segmentation on the [**Jetson Orin Nano**](https://developer.nvidia.com/embedded/jetson-orin), a lightweight embedded platform. Unlike conventional binary segmentation approaches, our framework adopts **multi-class segmentation** to capture diverse risk factors such as curbs, puddles, and construction zones — enabling safer and more context-aware navigation for visually impaired users.

- **Core Model:** **EfficientViT-L1**, selected for its optimal balance between accuracy and inference speed
- **Comparison:** [SegFormer](https://github.com/NVlabs/SegFormer) was fully re-implemented from scratch within the same training/evaluation framework
- **Custom Dataset:** 920 images captured at robot-level viewpoint (~23cm from ground) in Korean sidewalk environments, with a two-stage training strategy (pre-training on Mapillary → fine-tuning on custom data)

The full implementation and dataset are available at [github.com/dandylionkiddo/guideWay](https://github.com/dandylionkiddo/guideWay).

<br>

## 🛠 Environment & Testbed

### Hardware Testbed

A mobile testbed was constructed to validate the model's performance in real-world sidewalk environments. The camera was positioned at 23cm height to simulate the viewing angle of a walking-assistance robot.

![Mobile Testbed](<./images/Figure 2.png>)
*▲ Mobile testbed with Jetson Orin Nano and camera module mounted on a foldable platform cart (Figure 2)*

### Tech Stack

| Component | Specification |
| :--- | :--- |
| **Hardware** | NVIDIA Jetson Orin Nano Super Developer Kit (15W Mode) |
| **Camera** | IMX219-PQH5-C (8MP) |
| **Framework** | PyTorch, TensorRT, OpenCV |
| **Models** | EfficientViT, SegFormer |
| **Language** | Python 3.10 |

### Setup

```bash
conda create -n efficientvit python=3.10
conda activate efficientvit
pip install -U pip setuptools wheel
pip install -U "Cython>=0.29.36"
pip install -U -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

<br>

## 📊 Results

### Performance Metrics

| Metric | Result | Notes |
| :--- | :---: | :--- |
| **Inference Speed** | **14 FPS** | TensorRT FP32 + memory pre-allocation (exceeds 10 FPS target) |
| **Walkable Area Precision** | **86.67%** | High precision on sidewalk, crosswalk, minor-road, curb-cut |
| **Binary mIoU** | **83.02%** | Walkable vs. Non-walkable binary classification |
| **Overall mIoU / F1-Score** | 48.22% / 58.71% | Affected by class imbalance; strong performance on major classes |

### Qualitative Evaluation

![Qualitative Results](<./images/Figure 1.png>)
*▲ Ground truth masks (top) vs. model inference masks (bottom). The model successfully detected walkable roads and obstacles. In some cases, it even outperformed ground truth by detecting occluded objects and missing curb annotations (Figure 1)*

<br>

## 🚀 Usage

### Training

**EfficientViT**
```bash
PYTHONUTF8=1 python applications/efficientvit_seg/train.py \
  --config applications/efficientvit_seg/custom_seg.yaml \
  --path output/seg_test_run \
  --arch efficientvit
```

**SegFormer**
```bash
PYTHONUTF8=1 python applications/efficientvit_seg/train.py \
  --config applications/efficientvit_seg/custom_seg.yaml \
  --path output/seg_test_run \
  --arch segformer
```

### Evaluation

**EfficientViT**
```bash
python applications/efficientvit_seg/eval.py \
  --config applications/efficientvit_seg/eval_config.yaml \
  --arch efficientvit
```

**SegFormer**
```bash
PYTHONUTF8=1 python applications/efficientvit_seg/train.py \
  --eval_only --eval_checkpoint output/final_models/segformer-b0.pt \
  --config applications/efficientvit_seg/custom_seg.yaml \
  --arch segformer --path ./output/evaluation_results \
  --save_eval_results
```

---
