#!/usr/bin/env python3
"""
Evaluation script for ASVspoof5 eval dataset
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from importlib import import_module

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import EvalDataset, genSpoof_list


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
        fh.write("filename\tcm-score\n")  # evaluation-package expects exactly two columns
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            fields = trl.strip().split()
            if len(fields) >= 2:
                file_id = fields[1]
                assert fn == file_id, f"File ID mismatch: {fn} != {file_id}"
                fh.write("{}\t{}\n".format(file_id, sco))
            else:
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
    eval_trial_path = database_path / "ASVspoof5.eval.track_1.tsv"
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    
    # Inspect checkpoint to decide eval feature shape (layers/D)
    model_path = Path(args.model_path)
    print(f"Loading model from: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    # Defaults
    ckpt_layers = 6
    ckpt_in_dim = 768
    if isinstance(ckpt, dict):
        if 'layer_weights' in ckpt:
            try:
                ckpt_layers = int(ckpt['layer_weights'].shape[0])
            except Exception:
                pass
        if 'adapter_proj.weight' in ckpt:
            try:
                ckpt_in_dim = int(ckpt['adapter_proj.weight'].shape[1])
            except Exception:
                pass

    # Define model architecture and adapt shapes before loading weights
    model = get_model(model_config, device)
    try:
        # Adapt input fusion and adapter to checkpoint shapes (before loading)
        if hasattr(model, 'layer_weights') and model.layer_weights.numel() != ckpt_layers:
            model.layer_weights = torch.nn.Parameter(torch.zeros(ckpt_layers, device=device, dtype=torch.float32))
        if hasattr(model, 'adapter_proj') and getattr(model.adapter_proj, 'in_channels', None) != ckpt_in_dim:
            out_ch = model.adapter_proj.out_channels
            model.adapter_proj = torch.nn.Conv1d(in_channels=ckpt_in_dim, out_channels=out_ch, kernel_size=1, bias=False).to(device)
    except Exception as e:
        print(f"Warning: shape adaptation before load failed: {e}")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    # Prepare evaluation dataset
    file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=True)
    if args.test:
        file_eval = file_eval[:int(len(file_eval) * 0.01)]
    print("no. evaluation files:", len(file_eval))
    
    # Use real-time WavLM extraction; align to checkpoint input shape
    print("Using real-time WavLM extraction")
    audio_dir = database_path / "flac_E"
    target_frames = int(model_config.get("target_frames", 512))
    feature_root = Path(config["feature_path"]) if "feature_path" in config else (database_path / "features")
    pca_npz = None
    if ckpt_in_dim == 288:
        # Use PCA to project WavLM to 288 dims per layer
        candidate = feature_root / "pca_l8_d288.npz"
        if candidate.exists():
            pca_npz = str(candidate)
        else:
            print(f"Warning: expected PCA file not found at {candidate}; will attempt without PCA (may mismatch)")
    eval_set = EvalDataset(list_IDs=file_eval,
                           audio_dir=audio_dir,
                           target_frames=target_frames,
                           layers_to_use=ckpt_layers,
                           pca_npz=pca_npz)
    
    bs_test = int(config.get("batch_size_test", 32))
    nw_test = int(config.get("num_workers_test", 0))
    eval_loader = DataLoader(
        eval_set,
        batch_size=bs_test,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=nw_test,
        persistent_workers=False,
    )
    
    # Perform evaluation
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    produce_evaluation_file(eval_loader, model, device, output_path, eval_trial_path)
    
    print(f"Evaluation completed! Results saved to: {output_path}")
    
    # Auto-run evaluation metrics
    print("\nCalculating evaluation metrics...")
    eval_package_dir = Path(__file__).resolve().parents[1] / "evaluation-package"
    key_file = database_path / "ASVspoof5.eval.track_1.tsv"
    
    # Create temporary key file with header and matching samples
    temp_key_file = output_path.parent / "temp_key.tsv"
    with open(key_file, 'r') as f_in, open(temp_key_file, 'w') as f_out:
        # Strict two-column header as evaluation-package expects
        f_out.write("filename\tcm-label\n")
        eval_file_set = set(file_eval)
        matched_count = 0
        for line in f_in:
            fields = line.strip().split()
            if len(fields) >= 9:
                file_id = fields[1]
                label = fields[8]
                if file_id in eval_file_set:
                    f_out.write(f"{file_id}\t{label}\n")
                    matched_count += 1
    print(f"Matched {matched_count} key entries out of {len(file_eval)} eval files")
    
    result = subprocess.run([
        "python", "evaluation.py", "--m", "t1", 
        "--cm", str(output_path.resolve()), 
        "--cm_keys", str(temp_key_file.resolve())
    ], cwd=eval_package_dir, capture_output=True, text=True)
    
    print("Evaluation metrics:")
    print(result.stdout)
    temp_key_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof5 evaluation script")
    parser.add_argument("--config", type=str, required=True, help="path to config file (.json or .py)")
    parser.add_argument("--model_path", type=str, required=True, help="path to trained model weights")
    parser.add_argument("--output", type=str, default="./eval_scores.txt", help="output path for evaluation scores")
    parser.add_argument("--test", action="store_true", help="use only 1% of data for quick testing")
    args = parser.parse_args()
    main(args)
