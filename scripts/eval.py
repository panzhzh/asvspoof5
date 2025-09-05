#!/usr/bin/env python3
"""
Evaluation script for ASVspoof5 eval dataset
"""

import argparse
import json
import sys
from pathlib import Path
from importlib import import_module

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TestDataset, genSpoof_list


def get_model(model_config: dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("src.models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    return model


def produce_evaluation_file(data_loader: DataLoader, model, device: torch.device, save_path: Path, trial_path: Path) -> None:
    """Perform evaluation and save the score to a file (ASVspoof format)"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader, desc="Evaluating"):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            fields = trl.strip().split()
            # TSV format: original_id file_id gender codec_type track original_speaker codec algorithm label -
            if len(fields) >= 9:
                original_id = fields[0]  # E_1607
                file_id = fields[1]      # E_0009538969
                label = fields[8]        # spoof/bonafide
                assert fn == file_id, f"File ID mismatch: {fn} != {file_id}"
                fh.write("{} {} {} {}\n".format(original_id, file_id, sco, label))
            else:
                # Fallback for shorter format
                print(f"Warning: Unexpected line format: {trl.strip()}")
                continue
    print("Scores saved to {}".format(save_path))


def load_config(config_path: str) -> dict:
    """Load configuration from either JSON or Python file"""
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        # Load JSON config
        with open(config_path, "r") as f:
            return json.load(f)
    elif config_path.suffix == '.py':
        # Load Python config
        import sys
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config_module
        spec.loader.exec_module(config_module)
        
        return config_module.get_config()
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def main(args: argparse.Namespace) -> None:
    # Load experiment configurations
    config = load_config(args.config)
    model_config = config["model_config"]
    
    # Define database related paths
    database_path = Path(config["database_path"])
    feature_path = Path(config["feature_path"])
    eval_feature_path = feature_path / "eval"
    eval_trial_path = database_path / "ASVspoof5.eval.track_1.tsv"
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    
    # Define model architecture
    model = get_model(model_config, device)
    
    # Load model weights
    model_path = Path(args.model_path)
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare evaluation dataset
    _, file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=True)
    print("no. evaluation files:", len(file_eval))
    
    eval_set = TestDataset(list_IDs=file_eval, feature_dir=eval_feature_path)
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    
    # Perform evaluation
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    produce_evaluation_file(eval_loader, model, device, output_path, eval_trial_path)
    
    print(f"Evaluation completed! Results saved to: {output_path}")
    print("You can now use the evaluation-package to calculate metrics:")
    print(f"cd evaluation-package && python evaluation.py --mode t1 --score_cm {output_path} --key_cm ../data/ASVspoof5/ASVspoof5.eval.track_1.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof5 evaluation script")
    parser.add_argument("--config", type=str, required=True, help="path to config file (.json or .py)")
    parser.add_argument("--model_path", type=str, required=True, help="path to trained model weights")
    parser.add_argument("--output", type=str, default="./eval_scores.txt", help="output path for evaluation scores")
    args = parser.parse_args()
    main(args)