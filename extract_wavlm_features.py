#!/usr/bin/env python3
"""
WavLM Feature Extraction with Memmap Bucketing
Extracts L1-L6 features from microsoft/wavlm-base and stores in bucketed memmap files
"""

import argparse
import json
import numpy as np
import soundfile as sf
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from collections import defaultdict
import math

def setup_wavlm():
    """Load WavLM-base model and feature extractor"""
    model = WavLMModel.from_pretrained("microsoft/wavlm-base")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, processor

def get_file_list(meta_path):
    """Parse official TSV file and return list of file IDs"""
    with open(meta_path) as f:
        lines = f.readlines()
    
    # Official TSV format: second column is always the file_id
    return [line.strip().split()[1] for line in lines if line.strip()]

def load_and_preprocess_audio(audio_path):
    """Load audio and convert to mono 16kHz"""
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    return audio

def extract_wavlm_features(audio_path, model, processor):
    """Extract WavLM L1-L6 features from audio file"""
    # Load and preprocess audio
    audio = load_and_preprocess_audio(audio_path)
    
    # Preprocess with WavLM processor
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get L1-L6 (indices 0-5 in hidden_states)
    hidden_states = outputs.hidden_states[1:7]  # Skip layer 0 (input embeddings)
    features = torch.stack(hidden_states, dim=1)  # [1, 6, seq_len, 768]
    
    return features.squeeze(0).cpu().numpy().astype(np.float16)  # [6, seq_len, 768]

def get_sequence_length_only(audio_path, model, processor):
    """Get sequence length without extracting full features"""
    audio = load_and_preprocess_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return outputs.hidden_states[1].shape[1]  # seq_len from L1

def pass1_collect_lengths(data_root, test_mode=False):
    """Pass 1: Collect ALL sequence lengths for bucket planning"""
    print("Pass 1: Collecting sequence lengths (FULL DATA)...")
    
    model, processor = setup_wavlm()
    all_lengths = []
    length_records = {}  # {utt_id: seq_len}
    
    splits = {
        'train': (data_root / 'ASVspoof5.train.tsv', data_root / "flac_T/"),
        'dev': (data_root / 'ASVspoof5.dev.track_1.tsv', data_root / "flac_D/"),
        'eval': (data_root / 'ASVspoof5.eval.track_1.tsv', data_root / "flac_E/")
    }
    
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
    lengths_file = data_root / "lengths.jsonl"
    with open(lengths_file, 'w') as f:
        for utt_id, seq_len in length_records.items():
            f.write(json.dumps({"utt_id": utt_id, "seq_len": seq_len}) + '\n')
    
    # Analyze distribution and create buckets
    percentiles = np.percentile(all_lengths, [70, 85, 95, 99.5])
    print(f"Length percentiles (70/85/95/99.5): {percentiles}")
    
    # Align to multiples of 16 and add max bucket
    buckets = []
    for p in percentiles:
        aligned = math.ceil(p / 16) * 16
        buckets.append(aligned)
    
    # Add max bucket to avoid truncation
    max_len = max(all_lengths)
    max_bucket = math.ceil(max_len / 16) * 16
    buckets.append(max_bucket)
    
    # Remove duplicates and sort
    buckets = sorted(set(buckets))
    print(f"Final bucket sizes (aligned to 16): {buckets}")
    print(f"Max length: {max_len}, will use max bucket: {max_bucket}")
    
    return buckets

def pass2_extract_and_store(data_root, buckets, test_mode=False):
    """Pass 2: Extract features and store in bucketed memmap files"""
    print("Pass 2: Extracting features and storing...")
    
    model, processor = setup_wavlm()
    feature_root = data_root / "features"
    
    # Load length records from Pass1
    print("Loading length records from Pass1...")
    length_lookup = {}
    with open(data_root / "lengths.jsonl") as f:
        for line in f:
            record = json.loads(line)
            length_lookup[record["utt_id"]] = record["seq_len"]
    
    splits = {
        'train': (data_root / 'ASVspoof5.train.tsv', data_root / "flac_T/"),
        'dev': (data_root / 'ASVspoof5.dev.track_1.tsv', data_root / "flac_D/"),
        'eval': (data_root / 'ASVspoof5.eval.track_1.tsv', data_root / "flac_E/")
    }
    
    for split, (meta_path, audio_dir) in splits.items():
        # Skip if audio directory doesn't exist or is empty
        if not audio_dir.exists() or not any(audio_dir.glob("*.flac")):
            print(f"Skipping {split}: no audio files found in {audio_dir}")
            continue
            
        print(f"Processing {split} features...")
        split_dir = feature_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get files and assign buckets using saved lengths
        file_list = get_file_list(meta_path)
        if test_mode:
            file_list = file_list[:int(len(file_list) * 0.01)]
            
        file_bucket_map = {}
        bucket_counts = defaultdict(int)
        
        # Assign buckets using cached lengths (no forward pass)
        for utt_id in file_list:
            if utt_id not in length_lookup:
                continue  # Skip files not in length_lookup
            seq_len = length_lookup[utt_id]
            bucket = min([b for b in buckets if b >= seq_len], default=buckets[-1])
            file_bucket_map[utt_id] = (bucket, seq_len)
            bucket_counts[bucket] += 1
        
        # Create memmap files for each bucket
        bucket_arrays = {}
        bucket_indices = defaultdict(int)
        
        for bucket in buckets:
            if bucket_counts[bucket] > 0:
                bucket_path = split_dir / f"bucket_{bucket}.npy"
                bucket_arrays[bucket] = np.lib.format.open_memmap(
                    bucket_path, mode='w+', 
                    shape=(bucket_counts[bucket], 6, bucket, 768), 
                    dtype=np.float16
                )
        
        # Extract and store features (single forward pass per file)
        index_records = []
        
        for utt_id in tqdm(file_list, desc=f"Extracting {split} features"):
            audio_path = audio_dir / f"{utt_id}.flac"
            features = extract_wavlm_features(audio_path, model, processor)
            
            bucket, real_len = file_bucket_map[utt_id]
            idx = bucket_indices[bucket]
            
            # Pad to bucket size (never truncate)
            if features.shape[1] < bucket:
                padded = np.zeros((6, bucket, 768), dtype=np.float16)
                padded[:, :features.shape[1], :] = features
                features = padded
            
            # Store in memmap
            bucket_arrays[bucket][idx] = features
            
            # Record in index with metadata
            index_records.append({
                "utt_id": utt_id,
                "bucket": bucket,
                "idx": idx,
                "real_len": real_len,
                "dtype": "float16",
                "layers": [1, 2, 3, 4, 5, 6],
                "model": "microsoft/wavlm-base",
                "sr": 16000
            })
            
            bucket_indices[bucket] += 1
        
        # Flush memmap files
        for bucket_array in bucket_arrays.values():
            bucket_array.flush()
        
        # Write index.jsonl
        index_path = split_dir / "index.jsonl"
        with open(index_path, 'w') as f:
            for record in index_records:
                f.write(json.dumps(record) + '\n')
        
        print(f"{split} complete: {len(index_records)} samples across {len(bucket_arrays)} buckets")

def main():
    parser = argparse.ArgumentParser(description="Extract WavLM features with memmap bucketing")
    parser.add_argument("--test", action="store_true", help="Use only 1% of data for testing")
    args = parser.parse_args()
    
    data_root = Path("data/ASVspoof5")
    
    if args.test:
        print("Running in TEST MODE - using 1% of data")
    
    # Pass 1: Collect lengths and determine buckets
    buckets = pass1_collect_lengths(data_root, test_mode=args.test)
    
    # Pass 2: Extract and store features
    pass2_extract_and_store(data_root, buckets, test_mode=args.test)
    
    print("Feature extraction complete!")
    print(f"Features saved to: {data_root}/features/")

if __name__ == "__main__":
    main()