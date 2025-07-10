<h1 align='center'>Feedback Schrödinger Bridge Matching (FSBM) [ICLR 2025 Oral] </h1>
<div align="center">
<a href="https://panostheo98.github.io/" target="_blank">Panagiotis Theodoropoulos</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
<a href="https://scholar.google.com/citations?user=cNuoyO4AAAAJ&hl=en" target="_blank">Nikos Komianos Liu</a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
<a href="https://scholar.google.com/citations?user=imMz-oYAAAAJ&hl=en" target="_blank"> Vincent Pacelli </a><sup>1</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://ghliu.github.io/" target="_blank">Guan-Horng Liu</a><sup>2</sup>&ensp;<b>&middot;</b>&ensp;
  <a href="https://sites.gatech.edu/acds/" target="_blank">Evangelos A. Theodorou</a><sup>3</sup><br>
  <sup>1</sup>Georgia Tech &emsp;  <sup>2</sup>FAIR, Meta<br>
</div>

<br>

[Feedback Schrödinger Bridge Matching](https://arxiv.org/abs/2410.14055) (**FSBM**) is a new matching algorithm 
for learning diffusion models between two distributions utilizing partially pre-aligned pairs. 
Examples include crowd navigation tasks, opinion depolarization, and guided image translation.


## Installation
```bash
conda env create -f environment.yml
pip install -e .
```

## Experiments
```bash
python train.py experiment=$EXP 
```
where `EXP` is one of the settings in `configs/experiment/*.yaml`. Each `EXP` also has its corresponding trajectories of keypoints in the `data` folder.
By default, checkpoints and figures are saved under the folder `outputs`.

## Semi-supervised image translation

### Dataset
Run the first cell in `download_data.ipynb` to download the latent vectors and the keypoint trajectories for the semi-supervised image translations. Run the second cell in the `download_data.ipynb` to download the checkpoints for the ALAE.
The latent vectors and the `ALAE` autoencoder for the translation tasks downloaded from [LightSB-Matching](https://github.com/SKholkin/LightSB-Matching/tree/main)
should appear in the `data` and `ALAE/training_artifacts` folders respectively.
Use the pre-aligned pairs for each translation task in the `data` folder to perform age or gender translation. 

### Train
```bash
python train.py experiment=ffhq_$TRNSF
```
where `TRNSF` is either `gen` to denote gender translation or `age` to denote age translation.

### Sample from trained model
To sample from a checkpoint $CKPT saved under `outputs/runs/ffhq_$TRNSF/$CKPT`, run
```bash
python ffhq_sample.py --experiment ffhq_$TRNSF --ckpt_file $CKPT
```

## Implementation

FSBM alternatively solves the Conditional Stochastic Optimal Control (**CondSOC**) problem and the resulting marginal **Matching** problem. We implement FSBM on PyTorch Lightning.

- We solve **CondSOC** and **Matching** respectively in the validation and training epochs. `pl.Trainer` is instantiated with `num_sanity_val_steps=-1` and `check_val_every_n_epoch=1` so that the validation epoch is executed before the initial training epoch and after each subsequent training epoch. 
- The results of **CondSOC** are gathered in [`validation_epoch_end`] and stored as `train_data`, which is then used to initialize [`train_dataloader`]. We set `reload_dataloaders_every_n_epochs=1` to refreash `train_dataloader` with latest **CondSOC** results.
- The training direction (forward or backward) is altered in [`training_epoch_end`], which is called _after_ the validation epoch.

The overall procedure follows

> [validate epoch (sanity)] **CondSOC** with random coupling \
→ [training epoch #0] **Matching** forward drift \
→ [validate epoch #0] **CondSOC** given forward model coupling \
→ [training epoch #1] **Matching** backward drift \
→ [validate epoch #1] **CondSOC** given backward model coupling \
→ [training epoch #2] **Matching** forward drift \
→ ...


## Citation
If you find this repository helpful for your publications,
please consider citing our paper:
```
@article{theodoropoulos2024feedback,
  title={Feedback Schr$\backslash$" odinger Bridge Matching},
  author={Theodoropoulos, Panagiotis and Komianos, Nikolaos and Pacelli, Vincent and Liu, Guan-Horng and Theodorou, Evangelos A},
  journal={arXiv preprint arXiv:2410.14055},
  year={2024}
}
```