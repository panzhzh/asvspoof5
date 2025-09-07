#!/usr/bin/env python3
"""
WavLM Feature Extraction with Ragged Memmap Storage
Extracts L1–L8 features from microsoft/wavlm-base, applies PCA to 256 dims,
and stores in space-efficient ragged memmap format (train/dev only).
"""

import argparse
import warnings
import json

# Suppress PyTorch MHA mask dtype deprecation warning (harmless for our usage)
warnings.filterwarnings(
    "ignore",
    message=r"Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)
import numpy as np
import soundfile as sf
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sklearn.decomposition import IncrementalPCA

def setup_wavlm():
    """Load WavLM-base model and feature extractor"""
    model = WavLMModel.from_pretrained("microsoft/wavlm-base")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, processor

def get_file_list(meta_path):
    """Parse TSV file with header/column tolerance and return list of file IDs"""
    with open(meta_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        return []
    
    # Check if first line is header (contains letters/common column names)
    first_line = lines[0]
    is_header = any(word in first_line.lower() for word in ['file', 'utt', 'id', 'name', 'label', 'speaker'])
    
    data_lines = lines[1:] if is_header else lines
    file_ids = []
    
    for line in data_lines:
        parts = line.split('\t')  # Explicit tab split
        if len(parts) < 2:
            parts = line.split()  # Fallback to whitespace split
        
        if len(parts) >= 2:
            # Usually file_id is in second column (index 1)
            candidate_id = parts[1].strip()
            
            # Check if second column looks like a valid ID (no paths, no spaces)
            if candidate_id and '/' not in candidate_id and ' ' not in candidate_id:
                file_ids.append(candidate_id)
            else:
                # Fallback to first column if second column has issues
                file_ids.append(parts[0].strip())
        elif len(parts) == 1:
            # Sometimes only file_id is present
            file_ids.append(parts[0].strip())
    
    return file_ids

def load_and_preprocess_audio(audio_path):
    """Load audio and convert to mono 16kHz float32 [-1,1]"""
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Convert to float32 and clip to [-1, 1]
    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio

def extract_wavlm_features(audio_path, model, processor):
    """Extract WavLM L1–L8 features from audio file.

    Returns float32 array of shape [8, seq_len, 768]
    """
    # Load and preprocess audio
    audio = load_and_preprocess_audio(audio_path)
    
    # Preprocess with WavLM processor
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    # Ensure attention_mask dtype matches expected boolean type to avoid PyTorch warnings
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.bool)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Extract features
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get L1–L8 (indices 1..8). Skip layer 0 (input embeddings)
    hidden_states = outputs.hidden_states[1:9]
    features = torch.stack(hidden_states, dim=1)  # [1, 8, seq_len, 768]
    
    # Return float32 to avoid early quantization before PCA fitting/projection
    return features.squeeze(0).cpu().numpy().astype(np.float32)  # [8, seq_len, 768]

def get_sequence_length_only(audio_path, model, processor):
    """Get sequence length without extracting full features"""
    audio = load_and_preprocess_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(torch.bool)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs.hidden_states[1].shape[1]  # seq_len from L1


def fit_pca_models(data_root: Path,
                   model,
                   processor,
                   target_splits=['train'],
                   n_components: int = 320,
                   max_files: int = 2000,
                   samples_per_file: int = 200,
                   test_mode: bool = False) -> Path:
    """Fit per-layer IncrementalPCA (8 layers, 768→256) on sampled frames and save to NPZ.

    Returns the path to the saved PCA NPZ file.
    """
    print(f"Fitting IncrementalPCA (n_components={n_components}) for L1–L8...")
    feature_root = data_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    pca_path = feature_root / f"pca_l8_d{n_components}.npz"

    # Prepare file lists for train/dev
    all_splits = {
        'train': (data_root / 'ASVspoof5.train.tsv', data_root / "flac_T/"),
        'dev': (data_root / 'ASVspoof5.dev.track_1.tsv', data_root / "flac_D/")
    }
    if target_splits is None:
        target_splits = ['train']
    splits = {k: v for k, v in all_splits.items() if k in target_splits}

    # Create IncrementalPCA objects per layer
    ipc_list = [IncrementalPCA(n_components=n_components, batch_size=2048) for _ in range(8)]
    # Accumulate frames per layer to ensure each partial_fit sees >= n_components samples
    buffers = [[] for _ in range(8)]  # list of lists of arrays per layer
    counts = [0 for _ in range(8)]
    flush_threshold = max(n_components, 4096)  # aim for stable SVD batches
    n_seen = 0

    rng = np.random.default_rng(0)  # reproducible sampling

    for split, (meta_path, audio_dir) in splits.items():
        if not audio_dir.exists() or not any(audio_dir.glob("*.flac")):
            print(f"Skipping {split} for PCA fit: no audio files in {audio_dir}")
            continue
        file_list = get_file_list(meta_path)
        if test_mode:
            file_list = file_list[:max(1, int(len(file_list) * 0.01))]
        # Cap number of files used for PCA fit
        for utt_id in tqdm(file_list, desc=f"Fitting PCA on {split}"):
            if n_seen >= max_files:
                break
            audio_path = audio_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                continue
            try:
                feats = extract_wavlm_features(audio_path, model, processor)  # [8, T, 768]
                T = feats.shape[1]
                if T <= 0:
                    continue
                k = min(samples_per_file, T)
                # Random sample time indices without replacement (more robust than linspace)
                idx = rng.choice(T, size=k, replace=False)
                # For each layer, append sampled frames to buffer; flush when enough
                for l in range(8):
                    X = feats[l, idx, :].astype(np.float32)  # [k,768]
                    buffers[l].append(X)
                    counts[l] += X.shape[0]
                    if counts[l] >= flush_threshold:
                        Xbig = np.vstack(buffers[l])  # [N,768], N >= threshold
                        ipc_list[l].partial_fit(Xbig)
                        buffers[l].clear()
                        counts[l] = 0
                n_seen += 1
            except Exception as e:
                print(f"Warning: PCA fit skip {utt_id}: {e}")
                continue
        if n_seen >= max_files:
            break

    # Final flush for remaining buffers (only if enough samples)
    for l in range(8):
        if counts[l] >= n_components and len(buffers[l]) > 0:
            Xbig = np.vstack(buffers[l])
            ipc_list[l].partial_fit(Xbig)
        buffers[l].clear()
        counts[l] = 0

    # Save means and components
    npz_data = {}
    for l in range(8):
        npz_data[f"mean_{l}"] = ipc_list[l].mean_.astype(np.float32)
        npz_data[f"components_{l}"] = ipc_list[l].components_.astype(np.float32)  # [256,768]
        # Save and print cumulative explained variance ratio for verification
        evr_cum = ipc_list[l].explained_variance_ratio_.cumsum().astype(np.float32)
        npz_data[f"evr_{l}"] = evr_cum
        if evr_cum.size > 0:
            print(f"Layer {l+1}: EVR@{n_components} = {evr_cum[-1]:.4f}")
    np.savez(pca_path, **npz_data, n_layers=8, in_dim=768, out_dim=n_components)
    print(f"Saved PCA to {pca_path}")
    return pca_path

def pass1_collect_lengths(data_root, test_mode=False, target_splits=None):
    """Pass 1: Collect ALL sequence lengths (skip if lengths.jsonl exists)"""
    lengths_file = data_root / "lengths.jsonl"
    
    # 优先使用现有的完整lengths文件
    if lengths_file.exists():
        print(f"Found existing {lengths_file}, loading length records...")
        all_records = load_existing_lengths(lengths_file)
        
        # 如果指定了target_splits，过滤出相关的记录
        if target_splits is not None:
            filtered_records = {}
            split_prefixes = {'train': 'T_', 'dev': 'D_', 'eval': 'E_'}
            target_prefixes = [split_prefixes[split] for split in target_splits if split in split_prefixes]
            
            for utt_id, seq_len in all_records.items():
                if any(utt_id.startswith(prefix) for prefix in target_prefixes):
                    filtered_records[utt_id] = seq_len
            
            print(f"Filtered {len(filtered_records)} records for {target_splits} from {len(all_records)} total")
            return filtered_records
        
        return all_records
    
    if target_splits is None:
        target_splits = ['train', 'dev']
    
    print(f"Pass 1: Collecting sequence lengths for {target_splits}...")
    
    model, processor = setup_wavlm()
    all_lengths = []
    length_records = {}  # {utt_id: seq_len}
    
    all_splits = {
        'train': (data_root / 'ASVspoof5.train.tsv', data_root / "flac_T/"),
        'dev': (data_root / 'ASVspoof5.dev.track_1.tsv', data_root / "flac_D/")
    }
    
    # 只处理指定的splits
    splits = {k: v for k, v in all_splits.items() if k in target_splits}
    
    for split, (meta_path, audio_dir) in splits.items():
        # Skip if audio directory doesn't exist or is empty
        if not audio_dir.exists() or not any(audio_dir.glob("*.flac")):
            print(f"Skipping {split}: no audio files found in {audio_dir}")
            continue
            
        print(f"Processing ALL {split} lengths...")
        file_list = get_file_list(meta_path)
        if test_mode:
            file_list = file_list[:int(len(file_list) * 0.01)]
        
        for utt_id in tqdm(file_list, desc=f"{split} lengths"):
            audio_path = audio_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                continue  # Skip missing files
            seq_len = get_sequence_length_only(audio_path, model, processor)
            all_lengths.append(seq_len)
            length_records[utt_id] = seq_len
    
    # Save length records for Pass2
    with open(lengths_file, 'w') as f:
        for utt_id, seq_len in length_records.items():
            f.write(json.dumps({"utt_id": utt_id, "seq_len": seq_len}) + '\n')
    
    print(f"Lengths saved to {lengths_file}")
    return length_records

def load_existing_lengths(lengths_file):
    """Load existing sequence lengths from lengths.jsonl"""
    print(f"Loading existing lengths from {lengths_file}")
    length_records = {}
    with open(lengths_file) as f:
        for line in f:
            record = json.loads(line)
            length_records[record["utt_id"]] = record["seq_len"]
    
    print(f"Loaded {len(length_records):,} length records")
    return length_records

def pass2_extract_and_store_ragged(data_root, length_records, test_mode=False, target_splits=None,
                                   pca_npz: Path = None):
    """Pass 2: Extract features, apply PCA(256) per layer, and store ragged memmap (CSR-style)"""
    
    if target_splits is None:
        target_splits = ['train', 'dev']
        
    print(f"Pass 2: Extracting features for {target_splits} with ragged memmap storage...")
    
    model, processor = setup_wavlm()
    # Load PCA projection (fit if absent)
    if pca_npz is None:
        pca_npz = data_root / "features" / "pca_l8_d320.npz"
    if not pca_npz.exists():
        print(f"PCA file not found: {pca_npz}. Fitting PCA first...")
        # Fit using only TRAIN split to avoid leakage
        pca_npz = fit_pca_models(data_root, model, processor, target_splits=['train'], test_mode=test_mode)
    pca_data = np.load(pca_npz)
    means = [pca_data[f"mean_{l}"] for l in range(8)]            # (768,)
    comps = [pca_data[f"components_{l}"] for l in range(8)]      # (256,768)
    feature_root = data_root / "features"
    
    all_splits = {
        'train': (data_root / 'ASVspoof5.train.tsv', data_root / "flac_T/"),
        'dev': (data_root / 'ASVspoof5.dev.track_1.tsv', data_root / "flac_D/")
    }
    
    # 只处理指定的splits
    splits = {k: v for k, v in all_splits.items() if k in target_splits}
    
    # Shard size: 64GB per shard (in elements)
    SHARD_SIZE_GB = 64
    DTYPE = np.float16
    BYTES_PER_ELEM = np.dtype(DTYPE).itemsize
    SHARD_SIZE_ELEMS = int(SHARD_SIZE_GB * 1024**3 / BYTES_PER_ELEM)
    
    for split, (meta_path, audio_dir) in splits.items():
        # Skip if audio directory doesn't exist or is empty
        if not audio_dir.exists() or not any(audio_dir.glob("*.flac")):
            print(f"Skipping {split}: no audio files found in {audio_dir}")
            continue
            
        print(f"Processing {split} features with ragged storage...")
        split_dir = feature_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file list
        file_list = get_file_list(meta_path)
        if test_mode:
            file_list = file_list[:max(1, int(len(file_list) * 0.01))]  # Ensure at least 1 file
        
        # Filter files that exist in length_records and audio files
        valid_files = []
        bad_files = []
        
        for utt_id in file_list:
            if utt_id in length_records:
                audio_path = audio_dir / f"{utt_id}.flac"
                if audio_path.exists():
                    valid_files.append(utt_id)
                else:
                    bad_files.append((utt_id, "audio_missing"))
            else:
                bad_files.append((utt_id, "not_in_lengths"))
        
        # Sort for reproducible order
        valid_files.sort()
        
        print(f"Processing {len(valid_files):,} valid files for {split}")
        if bad_files:
            print(f"Warning: {len(bad_files)} files skipped (see bad.list)")
            bad_list_path = split_dir / "bad.list"
            with open(bad_list_path, 'w') as f:
                for utt_id, reason in bad_files:
                    f.write(f"{utt_id}\t{reason}\n")
        
        # Initialize shard tracking
        current_shard_id = 0
        current_shard_offset = 0
        current_shard_memmap = None
        index_records = []
        
        def create_new_shard():
            nonlocal current_shard_id, current_shard_offset, current_shard_memmap
            shard_path = split_dir / f"data_{current_shard_id:03d}.npy"
            print(f"Creating shard: {shard_path}")
            current_shard_memmap = np.lib.format.open_memmap(
                shard_path, mode='w+', 
                shape=(SHARD_SIZE_ELEMS,), 
                dtype=DTYPE
            )
            current_shard_offset = 0
        
        # Create first shard
        create_new_shard()
        
        # Process files with error handling
        failed_files = []
        
        for utt_id in tqdm(valid_files, desc=f"Extracting {split}"):
            audio_path = audio_dir / f"{utt_id}.flac"
            
            try:
                # Extract L1–L8 features and apply PCA to 256 dims per layer
                features = extract_wavlm_features(audio_path, model, processor)  # [8, T, 768]
                real_len = features.shape[1]

                proj_layers = []
                for l in range(8):
                    X = features[l].astype(np.float32)  # [T,768]
                    Xc = X - means[l]
                    Y = Xc @ comps[l].T  # [T,256]
                    proj_layers.append(Y.astype(np.float16))
                    del X, Xc, Y
                del features
                proj = np.stack(proj_layers, axis=0)  # [8,T,256]

                # Flatten for shard storage
                flat_features = proj.ravel()  # [8*T*256]
                elem_count = flat_features.size
                
                # Check if we need a new shard
                if current_shard_offset + elem_count > SHARD_SIZE_ELEMS:
                    # Flush current shard
                    current_shard_memmap.flush()
                    current_shard_id += 1
                    create_new_shard()
                
                # Store in current shard
                current_shard_memmap[current_shard_offset:current_shard_offset + elem_count] = flat_features
                
                # Record in index
                index_records.append({
                    "utt_id": utt_id,
                    "shard": f"data_{current_shard_id:03d}.npy",
                    "offset_elems": current_shard_offset,
                    "elem_count": elem_count,
                    "real_len": real_len,
                    "L": 8,  # number of layers
                    "D": 256,  # PCA-projected feature dimension
                    "dtype": str(DTYPE).split('.')[-1],  # "float16"
                    "layers": [1, 2, 3, 4, 5, 6, 7, 8],
                    "model": "microsoft/wavlm-base",
                    "sr": 16000,
                    "stride_ms": 20,
                    "version": "ragged_v1.0",
                    "storage_format": "ragged_memmap_csr"
                })
                
                # Update offset
                current_shard_offset += elem_count
                
            except Exception as e:
                failed_files.append((utt_id, str(e)))
                print(f"Warning: Failed to process {utt_id}: {e}")
                continue
        
        # Write failed files to bad.list
        if failed_files:
            failed_list_path = split_dir / "failed.list"
            with open(failed_list_path, 'w') as f:
                for utt_id, error in failed_files:
                    f.write(f"{utt_id}\t{error}\n")
            print(f"Warning: {len(failed_files)} files failed processing (see failed.list)")
        
        # Flush final shard and trim to actual size
        if current_shard_memmap is not None:
            current_shard_memmap.flush()
            
            # Trim the last shard to actual used size
            if current_shard_offset < SHARD_SIZE_ELEMS:
                final_shard_path = split_dir / f"data_{current_shard_id:03d}.npy"
                print(f"Trimming final shard to {current_shard_offset:,} elements")
                
                # Create properly sized final shard
                trimmed_shard = np.lib.format.open_memmap(
                    final_shard_path.with_suffix('.tmp.npy'), mode='w+',
                    shape=(current_shard_offset,), dtype=DTYPE
                )
                trimmed_shard[:] = current_shard_memmap[:current_shard_offset]
                trimmed_shard.flush()
                
                # Release handles before file operations
                del trimmed_shard
                del current_shard_memmap
                current_shard_memmap = None
                
                # Replace original with trimmed version
                final_shard_path.unlink()
                final_shard_path.with_suffix('.tmp.npy').rename(final_shard_path)
        
        # Write index.jsonl
        index_path = split_dir / "index.jsonl"
        with open(index_path, 'w') as f:
            for record in index_records:
                f.write(json.dumps(record) + '\n')
        
        # Calculate and report storage efficiency
        total_elems = sum(r["elem_count"] for r in index_records)
        total_gb = total_elems * BYTES_PER_ELEM / (1024**3)
        print(f"{split} complete: {len(index_records):,} samples, {total_gb:.2f}GB total ({current_shard_id + 1} shards)")
        print(f"Average length: {total_elems / len(index_records) / (8 * 256):.1f} frames")
        
        # Random sample verification (quick self-check)
        if len(index_records) > 0:
            print(f"Running quick self-check for {split}...")
            verify_random_samples(split_dir, index_records[:min(10, len(index_records))])

def verify_random_samples(split_dir, sample_records):
    """Verify a few random samples can be read back correctly"""
    import random
    
    samples_to_check = random.sample(sample_records, min(3, len(sample_records)))
    
    for record in samples_to_check:
        try:
            # Load shard
            shard_path = split_dir / record["shard"]
            shard_memmap = np.lib.format.open_memmap(shard_path, mode='r')
            
            # Extract sample
            offset = record["offset_elems"]
            elem_count = record["elem_count"]
            flat_data = shard_memmap[offset:offset + elem_count]
            
            # Reshape
            L, D, real_len = record["L"], record["D"], record["real_len"]
            features = flat_data.reshape(L, real_len, D)
            
            # Basic sanity checks
            assert features.shape == (L, real_len, D), f"Shape mismatch: {features.shape} vs ({L}, {real_len}, {D})"
            assert not np.any(np.isnan(features)), "Contains NaN values"
            assert not np.any(np.isinf(features)), "Contains infinite values"
            
            print(f"✓ {record['utt_id']}: shape {features.shape}, range [{features.min():.3f}, {features.max():.3f}]")
            
        except Exception as e:
            print(f"✗ {record['utt_id']}: verification failed - {e}")
    
    print("Self-check complete.")

def main():
    parser = argparse.ArgumentParser(description="Extract WavLM features (train/dev) with PCA(256) and ragged memmap storage")
    parser.add_argument("--test", action="store_true", help="Use only 1% of data for testing")
    parser.add_argument("--dev", action="store_true", help="Extract only dev dataset features")
    parser.add_argument("--train", action="store_true", help="Extract only train dataset features")
    args = parser.parse_args()
    
    data_root = Path("data/ASVspoof5")
    
    if args.test:
        print("Running in TEST MODE - using 1% of data")
    
    # 确定要处理的数据集（仅 train/dev）
    target_splits = None
    if args.dev:
        target_splits = ['dev']
        print("Extracting DEV dataset only")
    elif args.train:
        target_splits = ['train']
        print("Extracting TRAIN dataset only")
    else:
        target_splits = ['train', 'dev']
        print("Extracting both TRAIN and DEV datasets")
    
    # Pass 0: Fit PCA if missing
    feature_root = data_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    pca_file = feature_root / "pca_l8_d320.npz"
    if not pca_file.exists():
        model, processor = setup_wavlm()
        # Fit PCA using only TRAIN split to avoid leakage
        pca_file = fit_pca_models(data_root, model, processor, target_splits=['train'], test_mode=args.test)
    else:
        print(f"Found existing PCA file: {pca_file}")
    
    # Pass 1: Collect lengths (skip if lengths.jsonl exists)
    length_records = pass1_collect_lengths(data_root, test_mode=args.test, target_splits=target_splits)
    
    # Pass 2: Extract and store features with ragged memmap (with PCA projection)
    pass2_extract_and_store_ragged(data_root, length_records, test_mode=args.test, target_splits=target_splits, pca_npz=pca_file)
    
    print("Feature extraction complete!")
    print(f"Features saved to: {data_root}/features/")
    print("Storage format: Ragged memmap with CSR-style indexing (zero padding waste!)")

if __name__ == "__main__":
    main()
