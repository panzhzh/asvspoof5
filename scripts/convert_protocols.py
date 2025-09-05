#!/usr/bin/env python3
"""
Convert ASVspoof5 protocol files to AASIST expected format
"""
import sys
from pathlib import Path

def convert_protocol_file(input_file, output_file):
    """Convert protocol file to AASIST format"""
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            fields = line.strip().split()  # Split on any whitespace
            if len(fields) >= 9:  # Ensure we have enough fields
                # Extract needed fields: file_id, original_id, gender, codec, algorithm, label
                original_id = fields[0]  # T_4850, D_0062, etc. (original ID)
                file_id = fields[1]  # T_0000000000, D_0000000001, etc. (actual file name)
                gender = fields[2]  # F, M
                codec = fields[6] if len(fields) > 6 else "-"  # AC3, AC1, etc.
                algorithm = fields[7] if len(fields) > 7 else "-"  # A05, A11, etc.
                label = fields[8] if len(fields) > 8 else "spoof"  # spoof, bonafide
                
                # Write in format expected by AASIST: speaker file_id gender codec algorithm label
                # Use file_id (T_0000000000) as both speaker and file identifier since that's the actual filename
                fout.write(f"{original_id} {file_id} {gender} {codec} {algorithm} {label}\n")

if __name__ == "__main__":
    data_dir = Path("./data/ASVspoof5_extracted")
    
    # Convert training protocol
    convert_protocol_file(
        data_dir / "ASVspoof5.train.tsv",
        data_dir / "ASVspoof5.train.metainfor.txt"
    )
    print("Converted training protocol file")
    
    # Convert development protocol  
    convert_protocol_file(
        data_dir / "ASVspoof5.dev.track_1.tsv",
        data_dir / "ASVspoof5.dev.metainfor.txt"
    )
    print("Converted development protocol file")
    
    print("Protocol conversion completed!")