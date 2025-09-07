#!/usr/bin/env python3
"""
Training, validation, and evaluation script (refactored from Baseline-AASIST).
Uses generic module and file names under src/ and results/.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from importlib import import_module
from pathlib import Path
from shutil import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TrainDataset, TestDataset, genSpoof_list
from src.eval import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from src.utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def get_model(model_config: dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("src.models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print("no. model params:{}".format(nb_params))
    return model


def get_loader(database_path: Path, feature_path: Path, seed: int, config: dict, test_mode: bool = False):
    """Make PyTorch DataLoaders for train / development using pre-extracted features"""
    trn_feature_path = feature_path / "train"
    dev_feature_path = feature_path / "dev"

    trn_list_path = database_path / "ASVspoof5.train.tsv"
    dev_trial_path = database_path / "ASVspoof5.dev.track_1.tsv"

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
    if test_mode:
        file_train = file_train[:int(len(file_train) * 0.01)]
        file_train_set = set(file_train)
        d_label_trn = {k: v for k, v in d_label_trn.items() if k in file_train_set}
    print("no. training files:", len(file_train))

    target_frames = int(config["model_config"].get("target_frames", 512))
    train_set = TrainDataset(list_IDs=file_train, labels=d_label_trn, feature_dir=trn_feature_path, target_frames=target_frames)
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Variable-length collate for training (returns list of tensors)
    def collate_varlen_train(batch):
        feats, labels = zip(*batch)
        return list(feats), torch.tensor(labels, dtype=torch.long)

    # DataLoader performance knobs
    num_workers = min(max(os.cpu_count() // 2, 4), 8)
    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen,
        collate_fn=collate_varlen_train,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
    )

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
    if test_mode:
        file_dev = file_dev[:int(len(file_dev) * 0.01)]
    else:
        file_dev = file_dev[:2000]
    print("no. validation files:", len(file_dev))

    dev_set = TestDataset(list_IDs=file_dev, feature_dir=dev_feature_path, target_frames=target_frames)
    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return trn_loader, dev_loader, dev_trial_path


def produce_evaluation_file(data_loader: DataLoader, model, device: torch.device, save_path: Path, trial_path: Path) -> None:
    """Perform evaluation and save the score to a file (ASVspoof format)"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        with torch.no_grad():
            with autocast():
                _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            fields = trl.strip().split()
            # TSV format: original_id file_id gender codec_type track original_speaker codec algorithm label -
            if len(fields) >= 9:
                original_id = fields[0]  # E.g., T_4850
                file_id = fields[1]      # E.g., T_0000000000
                label = fields[8]        # spoof/bonafide
                assert fn == file_id, f"File ID mismatch: {fn} != {file_id}"
                fh.write("{} {} {} {}\n".format(original_id, file_id, sco, label))
            else:
                # Fallback for shorter format
                print(f"Warning: Unexpected line format: {trl.strip()}")
                continue
    print("Scores saved to {}".format(save_path))


def train_epoch(trn_loader: DataLoader,
                model,
                optim: torch.optim.Optimizer,
                device: torch.device,
                scheduler,
                config: dict):
    running_loss = 0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    target_frames = int(config["model_config"].get("target_frames", 512))
    scaler = GradScaler()

    def gpu_time_pad_crop(batch_list, target_len, device, is_train=True):
        processed = []
        for x in batch_list:
            # x: [L, T, D] on CPU
            x = x.to(device, non_blocking=True)
            L, T, D = x.shape
            if T == target_len:
                processed.append(x)
            elif T > target_len:
                start = torch.randint(0, T - target_len + 1, (1,), device=device).item() if is_train else 0
                processed.append(x[:, start:start + target_len, :])
            else:
                # repeat-pad along time on GPU
                repeats = (target_len // T) + 1
                x_rep = x.repeat(1, repeats, 1)[:, :target_len, :]
                processed.append(x_rep)
        return torch.stack(processed, dim=0)  # [B, L, target_len, D]

    for batch_list, batch_y in tqdm(trn_loader):
        batch_size = len(batch_list)
        num_total += batch_size

        batch_x = gpu_time_pad_crop(batch_list, target_frames, device, is_train=True)
        batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

        with autocast():
            _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
            batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optim)
        scaler.update()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


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
    # load experiment configurations
    config = load_config(args.config)
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    feature_path = Path(config["feature_path"])
    
    # define model related paths with timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_tag = "{}_ep{}_bs{}_{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"], timestamp)
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, dev_trial_path = get_loader(database_path, feature_path, args.seed, config, args.test)

    # Warm-up adapt: run a quick forward pass before creating optimizer
    # to ensure any shape-dependent parameters (e.g., layer fusion weights or
    # input adapters) are initialized with the correct [L, D]. This prevents
    # missing new parameters in the optimizer.
    try:
        model.eval()
        target_frames = int(config["model_config"].get("target_frames", 512))
        with torch.no_grad():
            batch_list, _ = next(iter(trn_loader))

            # GPU pad/crop like in training to form [B, L, T, D]
            def gpu_time_pad_crop(batch_list_local, target_len, device_local, is_train=False):
                processed = []
                for x in batch_list_local:
                    x = x.to(device_local, non_blocking=True)
                    L, T, D = x.shape
                    if T == target_len:
                        processed.append(x)
                    elif T > target_len:
                        start = 0  # deterministic for warm-up
                        processed.append(x[:, start:start + target_len, :])
                    else:
                        repeats = (target_len // T) + 1
                        x_rep = x.repeat(1, repeats, 1)[:, :target_len, :]
                        processed.append(x_rep)
                return torch.stack(processed, dim=0)

            warm_x = gpu_time_pad_crop(batch_list, target_frames, torch.device(device), is_train=False)
            _ = model(warm_x, Freq_aug=False)
            del warm_x
    except StopIteration:
        # Empty training loader in rare cases; skip warm-up
        pass

    # get optimizer and scheduler (after warm-up so new params are included)
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_dev_dcf = 1.
    best_dev_cllr = 1.
    n_swa_update = 0

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("training epoch{:03d}".format(epoch))

        running_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config)

        produce_evaluation_file(dev_loader, model, device, metric_path/"dev_score.txt", dev_trial_path)
        dev_dcf, dev_eer, dev_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
            running_loss, dev_eer, dev_dcf, dev_cllr))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_dcf", dev_dcf, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)
        torch.save(model.state_dict(), model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

        best_dev_dcf = min(dev_dcf, best_dev_dcf)
        best_dev_cllr = min(dev_cllr, best_dev_cllr)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1

        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_dcf, epoch)
        writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system (refactored)")
    parser.add_argument("--config", dest="config", type=str, default="config/config.py", help="configuration file (.json or .py)")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./results", help="output directory for results")
    parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
    parser.add_argument("--comment", type=str, default=None, help="comment to describe the saved model")
    parser.add_argument("--test", action="store_true", help="use only 1% of data for quick testing")
    args = parser.parse_args()
    main(args)
