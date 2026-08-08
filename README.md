## SpikeViMFormer 2025 [[Paper](https://arxiv.org/pdf/2512.19365)] [[Models](#pre-trained-checkpoints)] [[Cite](#citation)]

<p align="left">
  <img src="access/1.svg" alt="SpikeViMFormer overview" style="width:80%;">
</p>

<h1 align="center">Efficient Spike-driven Transformer for High-performance Drone-View Geo-Localization</h1>

<h3 align="center">
  <strong>Zhongwei Chen</strong><sup>1,2,3</sup>,
  <strong>Hai-Jun Rong</strong><sup>1,2,3</sup>,
  <strong>Zhao-Xu Yang*</strong><sup>1,2,3</sup>,
  <strong>Guoqi Li*</strong><sup>4,5,6</sup>
</h3>

<div align="center">
  <sup>1</sup>School of Aerospace Engineering, Xi'an Jiaotong University, China<br>
  <sup>2</sup>State Key Laboratory for Strength and Vibration of Mechanical Structures<br>
  <sup>3</sup>Shaanxi Key Laboratory of Environment and Control for Flight Vehicle<br>
  <sup>4</sup>Institute of Automation, Chinese Academy of Sciences, China<br>
  <sup>5</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
  <sup>6</sup>Peng Cheng Laboratory<br>
  <sup>*</sup>Corresponding authors
</div>

<div align="center">
  <p>
    <a href="https://ieeexplore.ieee.org/abstract/document/11622533/"><img src="https://img.shields.io/badge/Paper-IEEE-00629B?logo=ieee&logoColor=white" alt="IEEE paper"></a>
    <a href="https://arxiv.org/abs/2512.19365"><img src="https://img.shields.io/badge/arXiv-2512.19365-B31B1B?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
    <a href="#pre-trained-checkpoints"><img src="https://img.shields.io/badge/Model-Download-2E8B57" alt="Download model"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-D22128" alt="Apache 2.0 license"></a>
  </p>
</div>

## <a id="motivation"></a>💡 Motivation
<p align="center">
  <img src="access/2.svg" alt="SpikeViMFormer framework" style="width:100%;">
  <img src="access/3.svg" alt="SpikeViMFormer results" style="width:100%;">
</p>

## <a id="method-overview"></a>🧩 Method Overview
<p align="center">
  <img src="access/4.svg" alt="SpikeViMFormer visualization" style="width:100%;">
</p>
This repository provides the official PyTorch implementation of **Efficient Spike-driven
Transformer for High-performance Drone-View Geo-Localization**. SpikeViMFormer is a
hardware-friendly spiking neural network framework for drone-view geo-localization (DVGL).
It combines a lightweight spike-driven Transformer backbone with Spike-Driven Selective
Attention (SSA), a Spike-Driven Hybrid State-Space block (SHS), and Hierarchical Re-ranking
Alignment Learning (HRAL).

The implementation supports the experiments on
[University-1652](https://github.com/layumi/University1652-Baseline) and
[SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark).

## <a id="news"></a>🔥 News

- **July 19, 2025:** SpikeViMFormer was accepted by IEEE TCSVT 2026. 🎉
- **2025:** Training code, evaluation code, and pre-trained checkpoints were released.

---

## <a id="table-of-contents"></a>📚 Table of Contents
- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [Highlights](#highlights)
- [TODOs](#todos)
- [Installation](#installation)
- [Required Backbone Pre-training](#required-backbone-pre-training)
- [Dataset Access](#dataset-access)
- [Dataset Structure](#dataset-structure)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [Python API](#python-api)
- [Testing](#testing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## <a id="highlights"></a>✨ Highlights

- A shared dual-stream spike-driven Transformer for drone and satellite imagery.
- Direct initialization from the official E-SpikeFormer 10M and 19M ImageNet checkpoints.
- A self-contained Normalized Integer Leaky Integrate-and-Fire (NI-LIF) implementation.
- SSA for preserving critical local information under sparse spiking activation.
- SHS for capturing long-range dependencies with linear sequence complexity.
- HRAL for supervising the backbone with current-batch and historical neighborhood context.
- Lightweight inference: SSA and SHS are used during training and automatically bypassed
  after `model.eval()`.
- Unified training and evaluation commands for University-1652 and SUES-200.

## <a id="todos"></a>📜 TODOs

- [x] Release the training code.
- [x] Release the evaluation code.
- [x] Release the pre-trained SpikeViMFormer checkpoints.
- [ ] Add additional visualization and analysis tools.

## <a id="installation"></a>🛠️ Installation

Python 3.10 or later is recommended. Install the PyTorch build that matches your CUDA
environment, then install SpikeViMFormer in editable mode.

```bash
conda create -n spikevimformer python=3.10 -y
conda activate spikevimformer

# Install the appropriate PyTorch build for your CUDA environment first.
pip install torch torchvision
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

The core implementation depends only on PyTorch, TorchVision, NumPy, Pillow, and tqdm.
NI-LIF is implemented directly in this repository, so no embedded third-party spiking
framework is required.

## <a id="required-backbone-pre-training"></a>📦 Required Backbone Pre-training

SpikeViMFormer is **not trained from a randomly initialized spike backbone**. A new DVGL
training run must initialize its backbone from the corresponding ImageNet-pretrained
[E-SpikeFormer](https://github.com/BICLab/Spike-Driven-Transformer-V3) checkpoint.

Download the base-model weights listed in the official
[E-SpikeFormer training guide](https://github.com/BICLab/Spike-Driven-Transformer-V3/blob/main/SDT_V3/Classification/Model_Base/Train_Base.md):

| SpikeViMFormer model | CLI variant | Required E-SpikeFormer backbone | Download |
| --- | --- | --- | --- |
| SpikeViMFormer-T | `tiny` | `V3_10.0M_1x4.pth` | [Google Drive](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link) |
| SpikeViMFormer-S | `small` | `V3_19.0M_1x4.pth` | [Google Drive](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing) |

Store the downloaded files locally, for example:

```text
pretrained/
├── V3_10.0M_1x4.pth
└── V3_19.0M_1x4.pth
```

Pass the matching file through `--backbone-checkpoint` when starting a new training run.
The loader discards the original ImageNet classification head and loads the complete
E-SpikeFormer feature extractor. It also rejects a 10M/19M checkpoint if it does not match
the selected `tiny`/`small` variant.

> `--backbone-checkpoint` is required for new training. It is not required with `--resume`
> because a SpikeViMFormer training checkpoint already contains the initialized backbone.

## <a id="dataset-access"></a>💾 Dataset Access

Please download and prepare the following datasets:

- [University-1652](https://github.com/layumi/University1652-Baseline)
- [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark)

The `--data-root` argument may point either to a parent directory containing the dataset
folders or directly to the selected dataset folder.

## <a id="dataset-structure"></a>📁 Dataset Structure

### University-1652

```text
data/
└── University-1652/
    ├── train/
    │   ├── drone/
    │   │   ├── 0001/
    │   │   ├── 0002/
    │   │   └── ...
    │   └── satellite/
    │       ├── 0001/
    │       ├── 0002/
    │       └── ...
    └── test/
        ├── query_drone/
        ├── gallery_drone/
        ├── query_satellite/
        └── gallery_satellite/
```

### SUES-200

```text
data/
└── SUES-200/
    ├── Training/
    │   ├── 150/
    │   │   ├── drone/<location_id>/*
    │   │   └── satellite/<location_id>/*
    │   ├── 200/
    │   ├── 250/
    │   └── 300/
    └── Testing/
        ├── 150/
        │   ├── query_drone/<location_id>/*
        │   ├── gallery_drone/<location_id>/*
        │   ├── query_satellite/<location_id>/*
        │   └── gallery_satellite/<location_id>/*
        ├── 200/
        ├── 250/
        └── 300/
```

The drone and satellite training folders must use matching location directory names. The
data loader automatically keeps shared identities and maps them to contiguous class labels.

## <a id="project-structure"></a>🗂️ Project Structure

```text
SpikeViMFormer/
├── spikevimformer/
│   ├── data/                 # Dataset discovery and image transformations
│   ├── engine/               # Training and retrieval evaluation loops
│   ├── losses/               # InfoNCE, SSA alignment, and HRAL
│   ├── models/               # NI-LIF, backbone, SSA, SHS, and full model
│   └── utils/                # Checkpoint and reproducibility utilities
├── scripts/
│   ├── train.py              # Unified training entry point
│   └── evaluate.py           # Unified evaluation entry point
├── tests/
│   └── test_smoke.py         # Model and HRAL smoke tests
├── pyproject.toml
└── requirements.txt
```

## <a id="training"></a>🚀 Training

The default configuration follows the main paper settings: 384 × 384 input resolution,
5 epochs, batch size 64, AdamW with an initial learning rate of `1e-4`, HRAL `k=15`,
SSA weight `0.6`, and SHS weight `0.54`.

Two model variants are available:

| Paper name | CLI variant | Required backbone checkpoint | Descriptor dimension | Backbone parameters |
| --- | --- | --- | ---: | ---: |
| SpikeViMFormer-T | `tiny` | `V3_10.0M_1x4.pth` | 240 | 9.78 M |
| SpikeViMFormer-S | `small` | `V3_19.0M_1x4.pth` | 360 | 18.63 M |

### University-1652: Drone to Satellite

```bash
spikevimformer-train \
  --data-root ./data \
  --dataset University-1652 \
  --direction drone2satellite \
  --variant small \
  --backbone-checkpoint pretrained/V3_19.0M_1x4.pth \
  --output outputs/u1652-small-d2s
```

### University-1652: Satellite to Drone

```bash
spikevimformer-train \
  --data-root ./data \
  --dataset University-1652 \
  --direction satellite2drone \
  --variant small \
  --backbone-checkpoint pretrained/V3_19.0M_1x4.pth \
  --output outputs/u1652-small-s2d
```

### SUES-200

Use `--altitude` to select one of the four flight heights.

```bash
spikevimformer-train \
  --data-root ./data \
  --dataset SUES-200 \
  --direction drone2satellite \
  --altitude 300 \
  --variant small \
  --backbone-checkpoint pretrained/V3_19.0M_1x4.pth \
  --output outputs/sues200-300-small
```

If the package has not been installed in editable mode, run the module from the repository
root:

```bash
python -m scripts.train \
  --data-root ./data \
  --dataset University-1652 \
  --direction drone2satellite \
  --variant tiny \
  --backbone-checkpoint pretrained/V3_10.0M_1x4.pth
```

Each output directory contains:

- `config.json`: the complete experiment configuration;
- `last.pt`: the checkpoint from the latest epoch;
- `best.pt`: the checkpoint with the highest validation R@1.

Resume interrupted training with:

```bash
spikevimformer-train \
  --data-root ./data \
  --dataset University-1652 \
  --resume outputs/u1652-small-d2s/last.pt \
  --output outputs/u1652-small-d2s
```

Use `--variant tiny` or reduce `--batch-size` if GPU memory is limited. Automatic mixed
precision is enabled on CUDA by default and can be disabled with `--no-amp`.

## <a id="evaluation"></a>📊 Evaluation

### University-1652

```bash
spikevimformer-evaluate \
  --checkpoint outputs/u1652-small-d2s/best.pt \
  --data-root ./data \
  --dataset University-1652 \
  --direction drone2satellite
```

### SUES-200

```bash
spikevimformer-evaluate \
  --checkpoint outputs/sues200-300-small/best.pt \
  --data-root ./data \
  --dataset SUES-200 \
  --direction drone2satellite \
  --altitude 300
```

Evaluation reports Recall@1, Recall@5, Recall@10, and Average Precision (AP). The model
variant and number of classes are read from checkpoints produced by the current training
script. For external checkpoints without metadata, provide `--variant` and `--num-classes`.

## <a id="pre-trained-checkpoints"></a>🤗 Pre-trained Checkpoints

### E-SpikeFormer backbone initialization

The backbone weights required before DVGL training come from the official
[E-SpikeFormer repository](https://github.com/BICLab/Spike-Driven-Transformer-V3), not from
the SpikeViMFormer checkpoint folder:

- SpikeViMFormer-T: [`V3_10.0M_1x4.pth`](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link)
- SpikeViMFormer-S: [`V3_19.0M_1x4.pth`](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing)

These files are passed to `spikevimformer-train` with `--backbone-checkpoint` and are only
used to initialize the spike backbone.

### Fine-tuned SpikeViMFormer checkpoints

The following folder contains the final DVGL checkpoints used for evaluation:

- **Google Drive:**
  [SpikeViMFormer checkpoints](https://drive.google.com/drive/folders/1l_cMkAlHdEytL7SCkZkynEiIRTKvcNBQ?usp=drive_link)
- **Baidu Netdisk:** Coming soon.

Download a fine-tuned SpikeViMFormer checkpoint and pass its path to
`spikevimformer-evaluate` using the `--checkpoint` argument. Evaluation does not separately
load an E-SpikeFormer file because the backbone is already included in the fine-tuned
checkpoint.

> **Checkpoint compatibility:** the refactored package uses a clean parameter hierarchy.
> Checkpoints created by the previous `sample4geo` implementation may require a one-time
> key conversion. Do not silently ignore unmatched parameters when converting old weights.

## <a id="python-api"></a>🐍 Python API

```python
import torch

from spikevimformer import build_model

model = build_model(num_classes=701, variant="small")
model.load_backbone_checkpoint("pretrained/V3_19.0M_1x4.pth")
model.eval()

images = torch.randn(2, 3, 384, 384)
with torch.inference_mode():
    descriptors = model(images).descriptor

print(descriptors.shape)  # torch.Size([2, 360])
```

Descriptors returned in evaluation mode are L2-normalized. During training, the model also
returns SSA descriptors, SHS logits, and SHS embeddings for the auxiliary objectives.

## <a id="testing"></a>✅ Testing

Install the optional development dependencies and run:

```bash
python -m compileall -q spikevimformer scripts
pytest -q
```

The smoke tests verify the Tiny model's training and inference interfaces, HRAL feature
shapes, numerical stability, and gradient propagation.

## <a id="license"></a>🎫 License

This project is licensed under the [Apache License 2.0](LICENSE).

## <a id="acknowledgments"></a>🙏 Acknowledgments

This repository builds upon ideas and code from
[DAC](https://github.com/SummerpanKing/DAC),
[Meta-SpikeFormer](https://github.com/BICLab/Spike-Driven-Transformer-V2), and
[E-SpikeFormer](https://github.com/BICLab/Spike-Driven-Transformer-V3). We thank the authors
for making their excellent work publicly available.

## <a id="citation"></a>📌 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{chen2025efficient,
  title   = {Efficient Spike-driven Transformer for High-performance Drone-View Geo-Localization},
  author  = {Chen, Zhongwei and Rong, Hai-Jun and Yang, Zhao-Xu and Li, Guoqi},
  journal = {arXiv preprint arXiv:2512.19365},
  year    = {2025}
}
```
