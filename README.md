# Asvspoof5 Baselines 

By [ASVspoof5 challenge organizers](https://www.asvspoof.org/)

## Baseline CMs (track 1)

These baselines for CMs are available: 

* Baseline-RawNet2 (PyTorch) <br/> End-to-End DNN classifier
* Baseline-AASIST (PyTorch) <br/> End-to-End graph attention-based classifier


## Baseline SASV (track 2)

Two baselines for SASV are available: 

* SASV Fusion-based baseline from SASV 2022 challenge [here](https://github.com/sasv-challenge/SASVC2022_Baseline)
  * Input file should be replaced with those of ASVspoof5
* Single integrated SASV baseline: [here](https://github.com/sasv-challenge/SASV2_Baseline/tree/asvspoof5)
  * This is an adapted version of a work previously introduced in Interspeech 2023. Use the above link to access the `asvspoof5` branch.
  * Download the code directly: [https://github.com/sasv-challenge/SASV2_Baseline/archive/refs/tags/ASVspoof.v0.0.1.tar.gz](https://github.com/sasv-challenge/SASV2_Baseline/archive/refs/tags/ASVspoof.v0.0.1.tar.gz)

## Evaluation metrics

Track 1
* minDCF (primary)
* CLLR, EER, actDCF (secondary)

Track 2
* a-DCF (primary)
* min t-DCF and t-EER (secondary)


## Other tools

Tool-score-fusion: a reference tool to fuse ASV and CM scores for track 2.

---

# Enhanced Project Structure

This repository includes an enhanced, clean implementation for anti-spoofing research:

## Clean Project Structure

```
asvspoof5/
├── src/                    # Source code
│   ├── models/model.py    # Model implementation  
│   ├── data/datasets.py   # Data handling
│   └── utils/training_utils.py
├── config/config.py       # Training configuration
├── scripts/               # Executable scripts
│   ├── train.py          # Main training script
│   └── convert_protocols.py
├── data/ASVspoof5/        # Dataset
├── evaluation-package/    # Official evaluation metrics
└── Baseline-AASIST/       # Original baseline (unchanged)
```

## Usage

```bash
# Training
python scripts/train.py                    # Standard training
python scripts/train.py --comment exp1     # With experiment name
python scripts/train.py --eval            # Evaluation mode
```

## Key Features

- **Clean Architecture**: Modular design without AASIST-specific naming
- **Simple Configuration**: Single `config/config.py` file
- **Official Evaluation**: Uses evaluation-package for metrics
- **Optimized Performance**: 8 workers, memory pinning, TensorBoard logging

## Data Format

The project expects data in `data/ASVspoof5/` with:
- `flac_T/`: Training audio (182,357 files)  
- `flac_D/`: Development audio (142,134 files)
- Protocol files: `ASVspoof5.train.metainfor.txt`, `ASVspoof5.dev.metainfor.txt`
