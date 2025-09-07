import json
import warnings
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import librosa
import soundfile as sf
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# Suppress PyTorch MHA mask dtype deprecation warning during on-the-fly extraction
warnings.filterwarnings(
    "ignore",
    message=r"Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """Parse TSV/metainfor file and return list of file IDs and labels"""
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            fields = line.strip().split()
            if len(fields) >= 6:
                # TSV format: original_id file_id gender codec_type track original_speaker codec algorithm label -
                key = fields[1]  # file_id (e.g., T_0000000000)
                label = fields[-2] if len(fields) > 8 else fields[-1]  # label (spoof/bonafide)
                file_list.append(key)
                d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            fields = line.strip().split()
            if len(fields) >= 2:
                key = fields[1]  # file_id
                file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            fields = line.strip().split()
            if len(fields) >= 6:
                key = fields[1]  # file_id
                label = fields[-2] if len(fields) > 8 else fields[-1]  # label
                file_list.append(key)
                d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


class FeatureLoader:
    """Load and cache npy feature files (supports both bucketed and ragged memmap formats)"""
    def __init__(self, feature_dir: Path):
        self.feature_dir = Path(feature_dir)
        self.index_file = self.feature_dir / "index.jsonl"
        self.file_to_record = {}
        self.shard_data = {}  # For ragged memmap format
        self.bucket_data = {}  # For bucketed format
        self.format_type = None
        self._load_index()
    
    def _load_index(self):
        """Load file mapping from index.jsonl (auto-detect format)"""
        with open(self.index_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                utt_id = record['utt_id']
                
                # Auto-detect format type from first record
                if self.format_type is None:
                    if 'shard' in record:
                        self.format_type = 'ragged_memmap'
                    elif 'bucket' in record:
                        self.format_type = 'bucketed'
                    else:
                        raise ValueError(f"Unknown index format in {self.index_file}")
                
                self.file_to_record[utt_id] = record
        
        print(f"FeatureLoader: Detected {self.format_type} format with {len(self.file_to_record)} files")
    
    def _load_shard(self, shard_name):
        """Load shard data if not already loaded (ragged memmap format)"""
        if shard_name not in self.shard_data:
            shard_file = self.feature_dir / shard_name
            self.shard_data[shard_name] = np.lib.format.open_memmap(shard_file, mode='r')
    
    def _load_bucket(self, bucket_size):
        """Load bucket data if not already loaded (bucketed format)"""
        if bucket_size not in self.bucket_data:
            bucket_file = self.feature_dir / f"bucket_{bucket_size}.npy"
            self.bucket_data[bucket_size] = np.load(bucket_file, mmap_mode='r')
    
    def get_features(self, utt_id: str):
        """Get features for a specific utterance ID"""
        if utt_id not in self.file_to_record:
            raise KeyError(f"Utterance {utt_id} not found in feature index")
        
        record = self.file_to_record[utt_id]
        
        if self.format_type == 'ragged_memmap':
            # Ragged memmap CSR format
            shard = record['shard']
            offset_elems = record['offset_elems']
            elem_count = record['elem_count']
            real_len = record['real_len']
            L = record['L']
            
            # Load shard if not already loaded
            self._load_shard(shard)
            
            # Extract flat features
            flat_data = self.shard_data[shard][offset_elems:offset_elems + elem_count]
            
            # Calculate actual D from data size (handles dimension mismatches)
            D = elem_count // (L * real_len)
            
            # Reshape to (L, real_len, D)
            features = flat_data.reshape(L, real_len, D)
            
            return features
            
        elif self.format_type == 'bucketed':
            # Original bucketed format
            bucket_size = record['bucket']
            idx = record['idx']
            real_len = record['real_len']
            
            # Load bucket if not already loaded
            self._load_bucket(bucket_size)
            
            # Extract features: shape (layers, seq_len, hidden_dim)
            features = self.bucket_data[bucket_size][idx]  # Shape: (6, bucket_size, 768)
            
            # Crop to real length
            features = features[:, :real_len, :]  # Shape: (6, real_len, 768)
            
            return features
        
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")


class TrainDataset(Dataset):
    """Training dataset using pre-extracted WavLM features.

    Returns a FloatTensor of shape [6, T, 768] where T == target_frames.
    """
    def __init__(self, list_IDs, labels, feature_dir, target_frames: int = 512):
        self.list_IDs = list_IDs
        self.labels = labels
        self.feature_loader = FeatureLoader(feature_dir)
        self.target_frames = int(target_frames)

    def __len__(self):
        return len(self.list_IDs)

    def _pad_or_crop_layers(self, features: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or random-crop features along time to fixed length.

        features: np.ndarray with shape [L, T, D]
        returns: np.ndarray with shape [L, target_len, D]
        """
        assert features.ndim == 3, f"expected 3D features [L,T,D], got {features.shape}"
        L, seq_len, D = features.shape

        if seq_len == target_len:
            return features
        if seq_len > target_len:
            start_idx = np.random.randint(0, seq_len - target_len + 1)
            return features[:, start_idx:start_idx + target_len, :]
        # seq_len < target_len: repeat-pad along time
        num_repeats = (target_len // seq_len) + 1
        padded = np.tile(features, (1, num_repeats, 1))
        return padded[:, :target_len, :]

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # Load pre-extracted features [6, real_len, 768]
        features = self.feature_loader.get_features(key)
        features = self._pad_or_crop_layers(features, self.target_frames)  # [6, T, 768]

        # Convert to tensor
        # Copy into a writable CPU tensor to avoid from_numpy on readonly memmap views
        x_inp = torch.tensor(features, dtype=torch.float32)
        y = self.labels[key]
        return x_inp, y


class TestDataset(Dataset):
    """Validation/Evaluation dataset using pre-extracted WavLM features.

    Returns a FloatTensor of shape [6, T, 768] where T == target_frames.
    """
    def __init__(self, list_IDs, feature_dir, target_frames: int = 512):
        self.list_IDs = list_IDs
        self.feature_loader = FeatureLoader(feature_dir)
        self.target_frames = int(target_frames)

    def __len__(self):
        return len(self.list_IDs)

    def _pad_layers(self, features: np.ndarray, target_len: int) -> np.ndarray:
        """Pad layer features to target length (no random cropping for evaluation)."""
        assert features.ndim == 3, f"expected 3D features [L,T,D], got {features.shape}"
        L, seq_len, D = features.shape
        if seq_len >= target_len:
            return features[:, :target_len, :]
        num_repeats = (target_len // seq_len) + 1
        padded = np.tile(features, (1, num_repeats, 1))
        return padded[:, :target_len, :]

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # Load pre-extracted features
        features = self.feature_loader.get_features(key)  # Shape: (6, real_len, 768)
        features = self._pad_layers(features, self.target_frames)  # [6, T, 768]

        # Convert to tensor
        # Copy into a writable CPU tensor to avoid from_numpy on readonly memmap views
        x_inp = torch.tensor(features, dtype=torch.float32)
        return x_inp, key


class EvalDataset(Dataset):
    """Evaluation dataset that loads audio files and extracts WavLM features on-the-fly.

    Returns a FloatTensor of shape [6, T, 768] where T == target_frames.
    """
    def __init__(self, list_IDs, audio_dir, target_frames: int = 512):
        self.list_IDs = list_IDs
        self.audio_dir = Path(audio_dir)
        self.target_frames = int(target_frames)
        
        # Initialize WavLM model and processor for real-time feature extraction
        self.wavlm_model = None
        self.wavlm_processor = None
        self._init_wavlm()
    
    def _init_wavlm(self):
        """Initialize WavLM model and processor"""
        print("Loading WavLM model for real-time feature extraction...")
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
        self.wavlm_model.eval()
        
        if torch.cuda.is_available():
            self.wavlm_model = self.wavlm_model.cuda()
    
    def _load_and_preprocess_audio(self, audio_path):
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
    
    def _extract_wavlm_features(self, audio_path):
        """Extract WavLM L1-L6 features from audio file"""
        # Load and preprocess audio
        audio = self._load_and_preprocess_audio(audio_path)
        
        # Preprocess with WavLM processor
        inputs = self.wavlm_processor(audio, sampling_rate=16000, return_tensors="pt")
        # Ensure attention_mask is boolean to avoid PyTorch MHA warning
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.bool)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Extract features
        with torch.inference_mode():
            outputs = self.wavlm_model(**inputs, output_hidden_states=True)
        
        # Get L1-L6 (indices 1-6 in hidden_states)
        hidden_states = outputs.hidden_states[1:7]  # Skip layer 0 (input embeddings)
        features = torch.stack(hidden_states, dim=1)  # [1, 6, seq_len, 768]

        return features.squeeze(0).cpu().numpy().astype(np.float16)  # [6, seq_len, 768]
    
    def __len__(self):
        return len(self.list_IDs)
    
    def _pad_layers(self, features: np.ndarray, target_len: int) -> np.ndarray:
        """Pad layer features to target length (no random cropping)."""
        assert features.ndim == 3, f"expected 3D features [L,T,D], got {features.shape}"
        L, seq_len, D = features.shape
        if seq_len >= target_len:
            return features[:, :target_len, :]
        num_repeats = (target_len // seq_len) + 1
        padded = np.tile(features, (1, num_repeats, 1))
        return padded[:, :target_len, :]

    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # Construct audio file path
        audio_path = self.audio_dir / f"{key}.flac"
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Extract WavLM features from audio file
        features = self._extract_wavlm_features(audio_path)  # [6, seq_len, 768]
        features = self._pad_layers(features, self.target_frames)  # [6, T, 768]

        # Copy into a writable CPU tensor to avoid from_numpy on readonly memmap views
        x_inp = torch.tensor(features, dtype=torch.float32)
        return x_inp, key
