# UAHC

Uncertainty-Aware Hypergraph Consistency Learning for semi-supervised medical image segmentation.

This repository contains the training code for the UAHC segmentation model. The repository intentionally does not include the large model-weight file or the dataset.

## Repository contents

- `train_seg2020.py`: training and evaluation entry point.
- `config_seg.yaml`: dataset, model, optimizer, and checkpoint configuration.
- `models/`: encoder and decoder definitions.
- `loders/`: Kvasir-SEG dataset and data-loader code. The directory name is kept for compatibility with the original project.
- `utils/`: augmentation, loss, metric, and training utilities.

## Requirements

Use a Python environment with a CUDA-enabled PyTorch installation when GPU training is desired. Install the remaining Python dependencies with:

```bash
pip install pyyaml opencv-python medpy scikit-learn umap-learn matplotlib pillow scipy numpy torchvision
```

Install `torch` and a matching `torchvision` build from the official PyTorch instructions for your operating system and CUDA version.

## Model weight

The encoder is configured as `models.resnet.resnet101` with `pretrained=True`, so the code expects this file:

```text
models/resnet101.pth
```

The file is intentionally excluded from Git because it is approximately 179 MB and exceeds GitHub's 100 MB regular-file limit. The source code does not publish an official UAHC weight-release URL. A public third-party mirror with the matching checksum is available here:

<https://huggingface.co/Chrisathy/CrackNex/resolve/main/resnet101.pth>

This is an external mirror and is not maintained by this repository. Download it only if you trust the source, verify the checksum below, and then copy it to `models/resnet101.pth` before starting training. Otherwise, obtain the compatible weight file from the project owner or another trusted project release.

For the weight file currently used during development:

```text
SHA256: D25B500A332E97A7BCBC11DC1153EC07C2F5C8334FBAC2B36D882FE78A01AF98
```

Do not substitute an arbitrary torchvision ResNet-101 checkpoint: this repository uses a customized ResNet implementation and expects compatible parameter names and shapes.

## Dataset layout

The default paths in `config_seg.yaml` are relative to the repository root and expect the dataset in a sibling `dataset` directory:

```text
parent-directory/
├── UAHC/
└── dataset/
    └── Kvasir-SEG/
        ├── images/
        └── masks/
```

Each image and mask must have the same filename. The current loader splits the first 1,000 files into 800 training, 100 validation, and 100 test samples, with 80 labeled training samples by default. If your data is stored elsewhere, update `data_root` and `label_root` in `config_seg.yaml`.

The dataset is not included in this repository. Download it from the official Kvasir-SEG page:

<https://datasets.simula.no/kvasir-seg/>

Follow the dataset's license and terms of use. In particular, the dataset is intended for research and educational use unless you obtain the appropriate permission for another use.

## Train and evaluate

From the repository root, after preparing the weight file and dataset:

```bash
python train_seg2020.py --config config_seg.yaml --seed 12345
```

Training checkpoints are written below `checkpoint/`, which is ignored by Git. The default checkpoint path is `checkpoint/seg_best_128.pth`.

## License and citation

No license file was included in the source provided for this repository. Add the appropriate license before redistributing the code. Please cite the UAHC publication when using this implementation.
