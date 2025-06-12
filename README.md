# Domain-Adaptive Pretraining Improves Primate Behavior Recognition

**Felix B. Mueller, Timo Luedeckke, Richard Vogg, Alxander S. Ecker**

CV4Animals@CVPR 2025

## Installation

Install all necessary dependencies using

```
conda create -f environment.yaml
```

## Model Checkpoints

| Description   | Pretraining Data | Download |
| -------- | ------- | ------- |
| V-JEPA | VideoMix2M | [checkpoint](https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar) by Meta
| Ours | V-JEPA + ChimpACT     | [checkpoint](https://owncloud.gwdg.de/index.php/s/6yhr2IBaR9wJKlK)
| Ours    | V-JEPA + PanAf20k    | [checkpoint](https://owncloud.gwdg.de/index.php/s/rDCphhP4ktJBtN7)

All pretrained encoder are ViT-L models.

## Attentive Classifier Training & Evaluation

### Build Evaluation Data

**ChimpACT**

1. Clone the [ChimpACT](https://github.com/ShirleyMaxx/ChimpACT) repo and follow their instructions to download preprocess the ChimpACT dataset including action annotations. Place all data in `data/`
2. Run `python dap_behavior/data/eval/chimpact.py data/ChimpACT_processed` to create annotation files for use with this repo. The new annotations files are placed in `ChimpACT_processed/annotations/action`.
3. Adjust the paths in `configs/eval/chimpact.yaml` to match your video location and label file location if needed

**PanAf500**

1. Download and unzip the PanAf20k dataset to `data`
2. Run `python dap_behavior/data/eval/panaf500.py data/panaf/panaf500/ data/`. This may take a moment as we create both label files and cropped video snippets.
3. Adjust the paths in `configs/eval/panaf500.yaml` to match your video location and label file location if needed

### Run Training

Download the model checkpoints and place them under `models/`.

To train and evaluate an attentive classifier, run

```
python dap_behavior/jepa/evals/main.py --fname configs/eval/EVAL_SETUP
```

on a compute node with one A100. 

| Encoder   | Eval Dataset | EVAL_SETUP |
| -------- | ------- | ------- |
| V-JEPA | ChimpACT | `jepa_chimpact.yaml`
| V-JEPA | PanAf500 | `jepa_panaf500.yaml`
| Ours | ChimpACT     | `chimpact.yaml`
| Ours    | PanAf500    | `panaf500.yaml`

For PanAf500, this will produce a csv-file `models/video_classification_frozen/panaf500-TIMESTAMP/panaf20k-e48_r0.csv` containing train and validation accuracies for every epoch. For ChimpACT, there will be one JSON-file per epoch under `models/video_classification_frozen/chimpact-TIMESTAMP/` containing the validation mAP scores (the csv-file also exists, but only contains the training and validation loss).

## Domain-Adaptive Pretraining

Code for pretraining coming soon!

## Misc

This repository contains a fork of [facebookresearch/jepa](https://github.com/facebookresearch/jepa) by Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas. Our changes are mainly in `evals/video_classification_frozen/eval.py`, `app/jepa/train.py`, and `src/datasets/video_dataset.py`.
