# Domain-Adaptive Pretraining Improves Primate Behavior Recognition

**Felix B. Mueller, Timo Luedeckke, Richard Vogg, Alxander S. Ecker**

[CV4Animals@CVPR](https://www.cv4animals.com/) 2025 Oral 

**Paper:** [[arXiv]](https://arxiv.org/abs/2509.12193) [[pdf]](https://arxiv.org/pdf/2509.12193)

**NEW: pretraining code available!**

## Installation

Install all necessary dependencies using

```
conda create -f environment.yaml
```

## Model Checkpoints

| Description   | Pretraining Data | Download |
| -------- | ------- | ------- |
| V-JEPA | VideoMix2M | [checkpoint](https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar) by Meta
| Ours | V-JEPA + ChimpACT     | [checkpoint](https://owncloud.gwdg.de/index.php/s/l1ayAUXfAg4BPO8)
| Ours    | V-JEPA + PanAf20k    | [checkpoint](https://owncloud.gwdg.de/index.php/s/rDCphhP4ktJBtN7)

All pretrained encoder are ViT-L models.

## Attentive Classifier Training & Evaluation

### Build Evaluation Data

**ChimpACT**

1. Clone the [ChimpACT](https://github.com/ShirleyMaxx/ChimpACT) repo and follow their instructions to download preprocess the ChimpACT dataset including action annotations. Place all data in `data/`
2. Run `python dap_behavior/data/eval/chimpact.py data/ChimpACT_processed` to create annotation files for use with this repo. The new annotations files are placed in `ChimpACT_processed/annotations/action`.

**PanAf500**

1. Download and unzip the PanAf20k dataset to `data/`
2. Run `python dap_behavior/data/eval/panaf500.py data/panaf/panaf500/ data/`. This may take a moment as we create both label files and cropped video snippets.

If you placed your data somewhere else than in `data/`, you have to adjust the paths in the `configs/eval` config files to match your video location and label file location.

### Run Attentive Classifier Training

Download the model checkpoints and place them under `models/`.

To train and evaluate an attentive classifier, run

```bash
python dap_behavior/jepa/evals/main.py --fname configs/eval/EVAL_SETUP
```

on a compute node with one A100. 

| Encoder   | Eval Dataset | EVAL_SETUP |
| -------- | ------- | ------- |
| V-JEPA (no DAP) | ChimpACT | `jepa_chimpact.yaml`
| V-JEPA (no DAP)| PanAf500 | `jepa_panaf500.yaml`
| Ours (DAP)| ChimpACT     | `dap_chimpact.yaml`
| Ours (DAP)   | PanAf500    | `dap_panaf500.yaml`

For PanAf500, this will produce a csv-file `models/video_classification_frozen/panaf500-TIMESTAMP/panaf20k-e48_r0.csv` containing train and validation accuracies for every epoch. For ChimpACT, there will be one JSON-file per epoch under `models/video_classification_frozen/chimpact-TIMESTAMP/` containing the validation mAP scores (the csv-file also exists, but only contains the training and validation loss).

## Domain-Adaptive Pretraining

Extract center frames, run primate detection, chunk videos and create a label file by running

```bash
python dap_behavior/data/pretrain.py chimp_act data/ChimpACT_release_v1 data/
python dap_behavior/data/pretrain.py panaf data/panaf/ data/
```

`data/pretrain/videos` contains the chunked 3s videos. `data/pretrain/DATASET.json` is the label file listing videos and corresponding detected bounding boxes. `data/pretrain/DATASET/` contains temporary data (extracted center frames, detection results before non-maximum supression) and can be deleted.

Ensure that you copied the V-JEPA checkpoint `vitl16.pth.tar` to `model/`.

Start pretraining on a machine with 4x A100 40GB using

```bash
python dap_behavior/jepa/app/main.py --fname configs/pretrain/chimp_act.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3
```

If you want to use `submitit` instead, check if the SLURM settings in `dap_pretraining/jepa/app/main_distributed.py` work on your system and submit a SLURM job for pretraining using

```bash
python dap_pretraining/jepa/app/main_distributed.py --fname configs/pretrain/chimp_act.yaml
```

You can evaluate your pretrained model by adjusting `pretrain.folder` and `pretrain.checkpoint` in the config file.


## Misc

This repository contains a fork of [facebookresearch/jepa](https://github.com/facebookresearch/jepa) by Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas. Our changes are mainly in `evals/video_classification_frozen/eval.py`, `app/jepa/train.py`, and `src/datasets/video_dataset.py`. The code in `dap_behavior/eval` is adapted from [MMAction](https://github.com/open-mmlab/mmaction2).

## Citation

If you found this repository useful, please consider giving a ⭐️ and cite

```
@article{mueller2025domain,
  title={Domain-Adaptive Pretraining Improves Primate Behavior Recognition},
  author={Mueller, Felix B and Lueddecke, Timo and Vogg, Richard and Ecker, Alexander S},
  journal={arXiv preprint arXiv:2509.12193},
  year={2025}
}
```

You likely also want to cite [V-JEPA](https://github.com/facebookresearch/jepa?tab=readme-ov-file#citation).
