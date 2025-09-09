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
import gc
from datetime import datetime
from importlib import import_module
from pathlib import Path
from shutil import copy

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TrainDataset, TestDataset, genSpoof_list
from src.data.samplers import BlockBatchSampler
from src.eval import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from src.utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


# --- Minimal RaggedIndex (embedded) for reading E/V ---
class RaggedIndex:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.index = self.root / "index.jsonl"
        self.records = {}
        self.mm_cache = {}
        with self.index.open() as f:
            for line in f:
                r = json.loads(line)
                self.records[r["utt_id"]] = r

    def _mm(self, shard: str):
        if shard not in self.mm_cache:
            self.mm_cache[shard] = np.lib.format.open_memmap(self.root / shard, mode='r')
        return self.mm_cache[shard]

    def get(self, uid: str):
        r = self.records[uid]
        mm = self._mm(r["shard"])
        off = int(r["offset_elems"])
        cnt = int(r["elem_count"])
        T = int(r.get("T", r.get("real_len")))
        D = int(r["D"]) if "D" in r else cnt // T
        flat = mm[off:off + cnt]
        return np.asarray(flat).reshape(T, D)


def get_model(model_config: dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("src.models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print("no. model params:{}".format(nb_params))
    return model


def get_loader(database_path: Path,
               feature_path: Path,
               seed: int,
               config: dict,
               test_mode: bool = False,
               dev_limit: int | None = None):
    """Create training DataLoader and a builder for dev DataLoader (on-demand)."""
    trn_feature_path = feature_path / "train"
    # Use eval split for validation if configured
    use_eval = False
    try:
        from config.config import use_eval_for_dev as _USE_EVAL_FOR_DEV
        use_eval = bool(_USE_EVAL_FOR_DEV)
    except Exception:
        use_eval = False
    dev_feature_path = feature_path / ("eval" if use_eval else "dev")

    trn_list_path = database_path / "ASVspoof5.train.tsv"
    dev_trial_path = database_path / ("ASVspoof5.eval.track_1.tsv" if use_eval else "ASVspoof5.dev.track_1.tsv")

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
    if test_mode:
        file_train = file_train[:int(len(file_train) * 0.01)]
        file_train_set = set(file_train)
        d_label_trn = {k: v for k, v in d_label_trn.items() if k in file_train_set}
    print("no. training files:", len(file_train))

    target_frames = int(config["model_config"].get("target_frames", 512))
    bs_train = int(config.get("batch_size_train", config.get("batch_size", 128)))
    bs_dev = int(config.get("batch_size_dev", 32))
    nw_train = int(config.get("num_workers_train", 4))
    nw_dev = int(config.get("num_workers_dev", 0))
    io_block_shuffle = bool(config.get("io_block_shuffle", False))
    io_block_mb = int(config.get("io_block_mb", 512))
    io_crop_on_load = bool(config.get("io_crop_on_load", False))
    io_train_layers = config.get("io_train_layers", None)
    # When using ragged_memmap + block shuffle, return pointer to enable batch-level aggregated I/O
    return_pointer = bool(io_block_shuffle)
    # If C2S training enabled, also return key for per-sample weighting
    c2s_enable_train = bool(C2S_CFG.get("enable_train", False)) if 'C2S_CFG' in globals() else False
    train_set = TrainDataset(
        list_IDs=file_train,
        labels=d_label_trn,
        feature_dir=trn_feature_path,
        target_frames=target_frames,
        return_pointer=return_pointer,
        return_key=c2s_enable_train,
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Variable-length collate for training (returns list of tensors)
    def collate_varlen_train(batch):
        # Support (x,y) or (x,y,key)
        if len(batch[0]) == 3:
            feats, labels, keys = zip(*batch)
            return list(feats), torch.tensor(labels, dtype=torch.long), list(keys)
        feats, labels = zip(*batch)
        return list(feats), torch.tensor(labels, dtype=torch.long), None

    # Build block-wise batch sampler for near-sequential disk access (ragged_memmap only)
    if io_block_shuffle and getattr(train_set.feature_loader, "format_type", None) == "ragged_memmap":
        # Build per-shard, offset-sorted index blocks and batch sampler
        file_to_record = train_set.feature_loader.file_to_record
        # bytes per element from dtype (default float16 -> 2)
        def bytes_per_elem(dtype_str: str | None) -> int:
            if not dtype_str:
                return 2
            ds = dtype_str.lower()
            if "16" in ds:
                return 2
            if "32" in ds:
                return 4
            if "64" in ds:
                return 8
            return 2

        # Collect records keyed by shard
        per_shard = {}
        for ds_idx, utt_id in enumerate(file_train):
            rec = file_to_record[utt_id]
            shard = rec["shard"]
            off = int(rec["offset_elems"])
            cnt = int(rec["elem_count"])
            bpe = bytes_per_elem(rec.get("dtype"))
            size_bytes = cnt * bpe
            per_shard.setdefault(shard, []).append((off, ds_idx, size_bytes))

        # Build blocks within each shard
        block_bytes_limit = max(1, io_block_mb) * 1024 * 1024
        index_blocks: list[list[int]] = []
        # Map dataset index -> covering block contiguous byte range for its shard
        dsidx_to_blockrange: dict[int, tuple[int, int, str]] = {}
        for shard, items in per_shard.items():
            items.sort(key=lambda x: x[0])  # by offset
            cur_items: list[tuple[int, int, int]] = []  # (off, ds_idx, sz)
            acc = 0
            for off, ds_idx, sz in items:
                if acc > 0 and acc + sz > block_bytes_limit:
                    # finalize current block and index mapping
                    blk_start = cur_items[0][0]
                    blk_end = cur_items[-1][0] + cur_items[-1][2]
                    index_blocks.append([ds for (_o, ds, _s) in cur_items])
                    for _off, _ds, _sz in cur_items:
                        dsidx_to_blockrange[_ds] = (blk_start, blk_end, shard)
                    # reset
                    cur_items = []
                    acc = 0
                cur_items.append((off, ds_idx, sz))
                acc += sz
            if cur_items:
                blk_start = cur_items[0][0]
                blk_end = cur_items[-1][0] + cur_items[-1][2]
                index_blocks.append([ds for (_o, ds, _s) in cur_items])
                for _off, _ds, _sz in cur_items:
                    dsidx_to_blockrange[_ds] = (blk_start, blk_end, shard)

        batch_sampler = BlockBatchSampler(index_blocks=index_blocks,
                                          batch_size=bs_train,
                                          drop_last=True,
                                          seed=seed)

        # Instrumentation: report block stats for visibility
        try:
            block_sizes = []
            for blk in index_blocks:
                if not blk:
                    continue
                br = dsidx_to_blockrange.get(blk[0])
                if br is None:
                    continue
                start, end, _ = br
                block_sizes.append((end - start) * train_set.feature_loader.get_shard_memmap(list(per_shard.keys())[0]).dtype.itemsize)
            if block_sizes:
                import statistics as _stat
                mean_mb = (sum(block_sizes) / len(block_sizes)) / (1024 * 1024)
                med_mb = _stat.median(block_sizes) / (1024 * 1024)
                print(f"I/O blocks: {len(index_blocks)} ; mean {mean_mb:.1f} MB ; median {med_mb:.1f} MB ; io_block_mb={io_block_mb}; io_crop_on_load={io_crop_on_load}")
        except Exception:
            pass

        # Aggregated I/O collate: read contiguous block(s) per shard for current batch
        # Simple per-shard block cache to reuse contiguous reads across consecutive batches
        block_cache: dict[tuple[str, int, int], dict[str, object]] = {}

        def collate_varlen_train_batchio(batch):
            # batch: list of (ptr, label)
            # group by shard
            by_shard = {}
            for ptr, y in batch:
                by_shard.setdefault(ptr['shard'], []).append((ptr, y))
            out_feats = []
            out_labels = []
            for shard, items in by_shard.items():
                # sort by offset within shard
                items.sort(key=lambda t: t[0]['offset_elems'])
                mm = train_set.feature_loader.get_shard_memmap(shard)
                if not io_crop_on_load:
                    # Read full sample ranges as one contiguous block per shard group
                    # Compute enclosing block range from precomputed map (use first item's ds_idx)
                    blk_meta = dsidx_to_blockrange.get(int(items[0][0]['ds_idx']))
                    if blk_meta is None:
                        # fallback to per-batch union
                        start = items[0][0]['offset_elems']
                        end = max(t[0]['offset_elems'] + t[0]['elem_count'] for t in items)
                        cache_key = (shard, start, end)
                    else:
                        start, end, _sh = blk_meta
                        cache_key = (shard, start, end)
                    entry = block_cache.get(cache_key)
                    if entry is None:
                        flat = mm[start:end]
                        flat_np = np.array(flat, copy=True)
                        block_cache.clear()  # keep only one block to bound memory
                        block_cache[cache_key] = {'array': flat_np}
                    else:
                        flat_np = entry['array']  # type: ignore[assignment]
                    for p, y in items:
                        s = p['offset_elems'] - start
                        e = s + p['elem_count']
                        arr = flat_np[s:e]
                        L, T, D = p['L'], p['real_len'], p['D']
                        arr = arr.reshape(L, T, D)
                        out_feats.append(torch.tensor(arr, dtype=torch.float32))
                        out_labels.append(y)
                    # Keep flat_np in cache for potential reuse in next batch
                else:
                    # Read only first target_frames frames (per sample) and optional layer subset
                    tf = target_frames
                    for p, y in items:
                        L_total, T_total, D = p['L'], p['real_len'], p['D']
                        L_read = int(L_total if io_train_layers in (None, 'None') else min(L_total, int(io_train_layers)))
                        T_read = min(T_total, tf)
                        # Allocate output array
                        arr = np.empty((L_read, T_read, D), dtype=mm.dtype)
                        base = p['offset_elems']
                        stride_layer = T_total * D
                        seg_elems = T_read * D
                        # Read per-layer contiguous segment
                        for l in range(L_read):
                            seg_start = base + l * stride_layer
                            seg = mm[seg_start: seg_start + seg_elems]
                            # materialize to ndarray and reshape to [T_read, D]
                            seg_np = np.array(seg, copy=True).reshape(T_read, D)
                            arr[l] = seg_np
                        out_feats.append(torch.tensor(arr, dtype=torch.float32))
                        out_labels.append(y)
            return out_feats, torch.tensor(out_labels, dtype=torch.long)

        trn_loader = DataLoader(
            dataset=train_set,
            batch_sampler=batch_sampler,
            pin_memory=True,
            worker_init_fn=seed_worker if nw_train > 0 else None,
            collate_fn=collate_varlen_train_batchio,
            num_workers=nw_train,
            persistent_workers=True if nw_train > 0 else False,
        )
    else:
        # Fallback: regular random shuffle
        trn_kwargs = dict(
            dataset=train_set,
            batch_size=bs_train,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=gen,
            collate_fn=collate_varlen_train,
            num_workers=nw_train,
            persistent_workers=True if nw_train > 0 else False,
        )
        if nw_train > 0:
            trn_kwargs["prefetch_factor"] = 2
        trn_loader = DataLoader(**trn_kwargs)

    # Prepare dev file list now; actual DataLoader built on-demand each epoch
    _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
    if test_mode:
        file_dev = file_dev[:int(len(file_dev) * 0.01)]
    elif dev_limit is not None and dev_limit > 0:
        file_dev = file_dev[:dev_limit]
    print("no. validation files:", len(file_dev))

    def build_dev_loader():
        dev_set_local = TestDataset(list_IDs=file_dev, feature_dir=dev_feature_path, target_frames=target_frames)
        # dev loader: configurable batch size; typically num_workers=0; non-persistent
        dev_kwargs = dict(
            dataset=dev_set_local,
            batch_size=bs_dev,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=nw_dev,
            persistent_workers=False,
        )
        if nw_dev > 0:
            dev_kwargs["prefetch_factor"] = 2
        return DataLoader(**dev_kwargs)

    return trn_loader, build_dev_loader, dev_trial_path


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
                config: dict,
                epoch: int):
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

    # Set epoch on custom batch sampler if available (enables block shuffle per-epoch)
    try:
        bs = getattr(trn_loader, "batch_sampler", None)
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)
    except Exception:
        pass

    for batch in tqdm(trn_loader):
        if len(batch) == 3:
            batch_list, batch_y, batch_keys = batch
        else:
            batch_list, batch_y = batch
            batch_keys = None
        batch_size = len(batch_list)
        num_total += batch_size

        batch_x = gpu_time_pad_crop(batch_list, target_frames, device, is_train=True)
        batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

        # Optional per-sample weights from precomputed utter-level map
        sample_w = None
        if batch_keys is not None and isinstance(batch_keys, list) and isinstance(batch_keys[0], str):
            # fetch weights if available
            w = [1.0] * batch_size
            # try to capture closure variable if present
            try:
                c2s_wm = globals().get('_C2S_WEIGHT_MAP', None)
                if c2s_wm is not None:
                    w = [float(c2s_wm.get(k, 1.0)) for k in batch_keys]
            except Exception:
                pass
            sample_w = torch.tensor(w, dtype=torch.float32, device=device)

        with autocast():
            _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
            if sample_w is None:
                batch_loss = criterion(batch_out, batch_y)
            else:
                per_sample = F.cross_entropy(batch_out, batch_y, weight=torch.FloatTensor([0.1, 0.9]).to(device), reduction='none')
                batch_loss = (per_sample * sample_w).mean()
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
    bs_train = int(config.get("batch_size_train", config.get("batch_size", 128)))
    model_tag = "{}_ep{}_bs{}_{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], bs_train, timestamp)
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
    trn_loader, build_dev_loader, dev_trial_path = get_loader(
        database_path,
        feature_path,
        args.seed,
        config,
        args.test,
        args.dev_limit,
    )

    # Warm-up adapt: run a quick forward pass before creating optimizer
    # to ensure any shape-dependent parameters (e.g., layer fusion weights or
    # input adapters) are initialized with the correct [L, D]. This prevents
    # missing new parameters in the optimizer.
    try:
        model.eval()
        target_frames = int(config["model_config"].get("target_frames", 512))
        with torch.no_grad():
            # Ensure deterministic block order for warm-up
            try:
                bs = getattr(trn_loader, "batch_sampler", None)
                if hasattr(bs, "set_epoch"):
                    bs.set_epoch(0)
            except Exception:
                pass
            batch = next(iter(trn_loader))
            # Collate may return (feats, labels) or (feats, labels, keys)
            if isinstance(batch, (list, tuple)):
                batch_list = batch[0]
            else:
                batch_list = batch

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

    # Early stopping setup
    es_enabled = bool(config.get("early_stopping", False))
    es_metric_name = str(config.get("es_metric", "minDCF"))
    es_mode = str(config.get("es_mode", "min"))
    es_min_delta = float(config.get("es_min_delta", 0.002))
    es_patience = int(config.get("es_patience", 3))
    es_warmup = int(config.get("es_warmup_epochs", 1))
    es_best = None
    es_wait = 0

    def _metric_value():
        if es_metric_name.lower() == "mindcf":
            return dev_min_dcf
        if es_metric_name.lower() == "actdcf":
            return dev_actdcf
        if es_metric_name.lower() == "eer":
            return dev_eer
        if es_metric_name.lower() == "cllr":
            return dev_cllr
        # default
        return dev_min_dcf

    def _is_improved(curr, best):
        if best is None:
            return True
        if es_mode == "min":
            return curr < (best - es_min_delta)
        else:
            return curr > (best + es_min_delta)

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # ---- Prepare C2S utter-level weights if enabled ----
    _C2S_WEIGHT_MAP = None
    try:
        c2s_cfg = config.get("c2s", {}) if isinstance(config, dict) else {}
        if bool(c2s_cfg.get("enable_train", False)) and str(c2s_cfg.get("train_mode", "none")).lower() == "loss_weight":
            from src.data.datasets import FeatureLoader
            # Load C, E, V indices
            c_loader = FeatureLoader(feature_path / "train")
            e_index = RaggedIndex(feature_path / "E" / "train")
            v_index = RaggedIndex(feature_path / "V" / "train")
            # Get train list + labels
            d_label_trn, file_train = genSpoof_list(dir_meta=database_path / "ASVspoof5.train.tsv", is_train=True, is_eval=False)
            # Fit PCA with small sample to control memory
            pca_dim = int(c2s_cfg.get("pca_dim", 64))
            ridge_alpha = float(c2s_cfg.get("ridge_alpha", 1e-2))
            var_floor = float(c2s_cfg.get("var_floor", 1e-4))
            # Select bonafide subset
            bon = [u for u in file_train if d_label_trn.get(u, 0) == 1]
            rng = np.random.default_rng(0)
            sample_ratio = 0.1
            sel = bon if len(bon) < 2000 else list(rng.choice(bon, size=int(len(bon)*sample_ratio), replace=False))
            # Incremental PCA
            from sklearn.decomposition import IncrementalPCA
            pca = IncrementalPCA(n_components=pca_dim, batch_size=4096)
            for uid in sel:
                C_LTD = c_loader.get_features(uid)
                C_TD = C_LTD.mean(axis=0).astype(np.float32)
                T = C_TD.shape[0]
                if T <= 0:
                    continue
                idx = rng.choice(T, size=max(1, int(T*0.2)), replace=False)
                pca.partial_fit(C_TD[idx])
            # Ridge closed form accumulators
            XT_X = None
            XT_Y = None
            for uid in sel:
                C_LTD = c_loader.get_features(uid)
                C_TD = C_LTD.mean(axis=0).astype(np.float32)
                E_TD = e_index.get(uid).astype(np.float32)
                T = min(C_TD.shape[0], E_TD.shape[0])
                if T <= 0:
                    continue
                C64 = pca.transform(C_TD[:T])
                Xb = np.concatenate([C64, np.ones((T,1), dtype=np.float32)], axis=1)
                Y = E_TD[:T]
                xx = Xb.T @ Xb
                xy = Xb.T @ Y
                XT_X = xx if XT_X is None else XT_X + xx
                XT_Y = xy if XT_Y is None else XT_Y + xy
            if XT_X is None:
                _C2S_WEIGHT_MAP = None
            else:
                XT_X.flat[::XT_X.shape[0]+1] += ridge_alpha
                W = np.linalg.solve(XT_X, XT_Y)
                A = W[:-1].T  # [32,pca_dim]
                b = W[-1].astype(np.float32)
                # variance
                var_acc = np.zeros((32,), dtype=np.float64)
                cnt = 0
                for uid in sel:
                    C_LTD = c_loader.get_features(uid)
                    C_TD = C_LTD.mean(axis=0).astype(np.float32)
                    E_TD = e_index.get(uid).astype(np.float32)
                    T = min(C_TD.shape[0], E_TD.shape[0])
                    if T <= 0:
                        continue
                    C64 = pca.transform(C_TD[:T])
                    pred = (C64 @ A.T + b)
                    resid = (E_TD[:T] - pred)
                    var_acc += (resid ** 2).mean(axis=0)
                    cnt += 1
                sigma = np.sqrt(np.maximum(var_acc / max(1, cnt), var_floor)).astype(np.float32)
                # Weights for all train uttrs
                wm = {}
                for uid in file_train:
                    C_LTD = c_loader.get_features(uid)
                    C_TD = C_LTD.mean(axis=0).astype(np.float32)
                    E_TD = e_index.get(uid).astype(np.float32)
                    V_T = v_index.get(uid).astype(np.uint8).reshape(-1)
                    T = min(C_TD.shape[0], E_TD.shape[0], V_T.shape[0])
                    if T <= 0:
                        wm[uid] = 1.0
                        continue
                    C64 = pca.transform(C_TD[:T])
                    pred = (C64 @ A.T + b)
                    z = (E_TD[:T] - pred) / sigma
                    nll = 0.5 * (z ** 2).sum(axis=1) + 0.5 * np.log((sigma ** 2)).sum()
                    if bool(c2s_cfg.get("train_voiced_only", True)) and V_T[:T].sum() > 0:
                        n = nll[V_T[:T] > 0]
                    else:
                        n = nll
                    wm[uid] = float(n.mean()) if n.size > 0 else float(nll.mean())
                vals = np.array(list(wm.values()), dtype=np.float32)
                z = (vals - vals.mean()) / (vals.std() + 1e-6)
                gamma = float(c2s_cfg.get("train_weight_gamma", 0.2))
                w_min = float(c2s_cfg.get("train_weight_w_min", 0.5))
                w_max = float(c2s_cfg.get("train_weight_w_max", 1.5))
                ws = np.clip(1.0 + gamma * z, w_min, w_max)
                _C2S_WEIGHT_MAP = {k: float(w) for k, w in zip(wm.keys(), ws)}
                # expose to training loop via module globals
                globals()['_C2S_WEIGHT_MAP'] = _C2S_WEIGHT_MAP
                print("[C2S] Train-time utter weights prepared.")
    except Exception as e:
        print(f"[C2S] disabled: {e}")

    # Training
    for epoch in range(config["num_epochs"]):
        print("training epoch{:03d}".format(epoch))

        running_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config, epoch)

        # Build fresh dev loader each epoch, evaluate, then release
        dev_loader = build_dev_loader()
        produce_evaluation_file(dev_loader, model, device, metric_path/"dev_score.txt", dev_trial_path)
        del dev_loader
        gc.collect()
        dev_min_dcf, dev_eer, dev_cllr, dev_actdcf = calculate_minDCF_EER_CLLR(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}\n  minDCF: {:.6f} | actDCF: {:.6f} | EER: {:.3f}% | CLLR: {:.6f} bits".format(
            running_loss, dev_min_dcf, dev_actdcf, dev_eer * 100.0, dev_cllr))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_minDCF", dev_min_dcf, epoch)
        writer.add_scalar("dev_actDCF", dev_actdcf, epoch)
        writer.add_scalar("dev_eer", dev_eer * 100.0, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)
        torch.save(model.state_dict(), model_save_path / "epoch_{}_{:05.2f}EER.pth".format(epoch, dev_eer * 100.0))

        best_dev_dcf = min(dev_min_dcf, best_dev_dcf)
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

        # Early stopping check
        if es_enabled and epoch >= es_warmup:
            curr = _metric_value()
            if _is_improved(curr, es_best):
                es_best = curr
                es_wait = 0
                # Save best-by-ES checkpoint
                torch.save(model.state_dict(), model_save_path / f"best_ES_{es_metric_name}_{curr:.6f}.pth")
                print(f"[ES] New best {es_metric_name}={curr:.6f} at epoch {epoch}")
            else:
                es_wait += 1
                print(f"[ES] No improvement on {es_metric_name} ({curr:.6f}); wait {es_wait}/{es_patience}")
                if es_wait >= es_patience:
                    print(f"Early stopping at epoch {epoch} (best {es_metric_name}={es_best:.6f})")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system (refactored)")
    parser.add_argument("--config", dest="config", type=str, default="config/config.py", help="configuration file (.json or .py)")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./results", help="output directory for results")
    parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
    parser.add_argument("--comment", type=str, default=None, help="comment to describe the saved model")
    parser.add_argument("--test", action="store_true", help="use only 1% of data for quick testing")
    parser.add_argument("--dev-limit", dest="dev_limit", type=int, default=None,
                        help="limit number of dev samples (default: None = use full dev)")
    args = parser.parse_args()
    main(args)
