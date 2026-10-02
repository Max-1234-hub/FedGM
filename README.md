# FedGM
This repo is the source code of the paper "Target-Oriented Federated Gradient Matching for Assisting Resource-Constrained Client in Medical Image Segmentation".

This release runs binary 2D medical image segmentation with `UNet2D` on Fundus, Polyp, Prostate, and breast ultrasound data.

## Environment

- Linux with an NVIDIA GPU and a working CUDA driver. The training code calls CUDA directly; CPU training is not supported.
- Python 3.10 or newer. FedGM and FedAvg one-round smoke runs were performed with Python 3.12.9 and PyTorch 2.5.1.
- Python dependencies are listed in [`requirements.txt`](requirements.txt). Install a CUDA-enabled PyTorch build suitable for your system using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining packages.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch build selected at pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The final command must print `True` for CUDA availability. `python main_seg.py --help` lists all run-time options.

## Obtain and organize the data

The experiments use public medical segmentation datasets organized as in [FL-MedSegBench](https://github.com/meiluzhu/FL-MedSegBench) by Meilu Zhu et al. Download the data through the [benchmark dataset documentation](https://github.com/meiluzhu/FL-MedSegBench/tree/master/docs/datasets) and the original dataset sources. Follow each source's access, reuse, and citation terms. This repository does not redistribute any images or masks.

Pass `--data_path` as the root containing the client folders for **one** selected dataset:

| `--dataset` | Training clients | Held-out client | Required layout under `--data_path` |
| --- | --- | --- | --- |
| `Fundus` | ChaseDB, DR-Hagis, DRIVE, HRF, LES-AV, ORVS | IOSTAR | `<client>/{Train,Test}/Original/{Images,Labels}/<same filename>` |
| `Polyp` | CVC-300, CVC-ClinicDB, CVC-ColonDB, EndoTect, ETISLaribPolypDB | Kvasir-SEG | `<client>/{images,masks}/<same filename>` |
| `Prostate` | BIDMC, BMC, HK, I2CVB, RUNMC, UCL | MSD | Training clients: `<client>/<patient>.nii.gz` plus `<patient>_segmentation.nii.gz`; BMC uses `_Segmentation.nii.gz`. MSD: `MSD/{imagesTr,labelsTr}/<same filename>` |
| `FL_Breast_Ultrasound` | BUID, BUSI, BUS_UC, BUS_UCLM | BUS | `<client>/{images,masks}/<same filename>` |

`datasets.py` contains the preprocessing and seed-based splits. FedGM uses the held-out client's validation subset to compute the server gradient and its separate test subset for evaluation. Ensure each client has enough samples to create nonempty splits. `--data_size` controls the held-out validation fraction for Fundus, Polyp, and Prostate.

## Train FedGM

Run from the repository root, replacing the data path with the appropriate absolute path:

```bash
python -u main_seg.py \
  --dataset Fundus \
  --data_path /absolute/path/to/fundus-root \
  --method FedGM \
  --device 0 \
  --T 200 --E 2 --batchsize 8 \
  --optimizer adam --loss dice_bce \
  --lr 0.001 --gm_lr 0.005 --server_epochs 1 \
  --save_path results/Fundus/FedGM
```

`--device` selects the GPU index through `CUDA_VISIBLE_DEVICES`. `--T` is communication rounds, `--E` is local epochs per round, `--lr` is the client learning rate, `--gm_lr` is the FedGM aggregation-weight optimizer learning rate, and `--server_epochs` is its optimization steps per round. Runs use seeds `0 1 2` by default; add `--seeds 0` for one seed.

Checkpoints, training logs, and Dice curves are written under `--save_path`. The `fedgm` branch in `server_funct.py` computes a server validation gradient and optimizes softmax client aggregation weights by cosine alignment.

## Run comparison methods

Use the same command with a different `--method` and output directory. For example:

```bash
python -u main_seg.py \
  --dataset Fundus \
  --data_path /absolute/path/to/fundus-root \
  --method FedAvg \
  --device 0 \
  --T 200 --E 2 --batchsize 8 \
  --optimizer adam --loss dice_bce --lr 0.001 \
  --save_path results/Fundus/FedAvg
```

The original comparison launchers used `FedAvg`, `FedProx`, `MOON`, `FedNova`, `PN`, and `FedAWA`; additional implemented methods are listed by `python main_seg.py --help`. [`examples/train.sh`](examples/train.sh) provides a shorter launcher:

```bash
bash examples/train.sh /absolute/path/to/fundus-root Fundus FedGM
bash examples/train.sh /absolute/path/to/fundus-root Fundus FedAvg
```

Set `GPU=1` before the script to select another GPU. Adjust dataset-specific learning rates and other hyperparameters to match the experiment you wish to reproduce.

## Code structure

| File | Purpose |
| --- | --- |
| `main_seg.py` | Training loop and experiment settings |
| `server_funct.py` | FedGM aggregation and comparison server updates |
| `client_funct_seg.py` | Local training and comparison client updates |
| `datasets.py` | Dataset readers, preprocessing, and splits |
| `models_dict/unet2d.py` | Segmentation network |
| `nodes.py`, `utils.py`, `loss.py` | Training state, metrics, and losses |

## Citation and acknowledgement
@inproceedings{2026FedGM,
  title={Target-Oriented Federated Gradient Matching for Assisting Resource-Constrained Client in Medical Image Segmentation},
  author={Mao, Axiu and Hang, Junlong and Zhu, Meilu},
  booktitle={2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026}
}
...

## License

...
