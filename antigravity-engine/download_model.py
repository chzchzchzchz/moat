import os
import sys
from huggingface_hub import hf_hub_download

MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_FILE = "model.safetensors"
TOKENIZER_FILE = "tokenizer.model"
TOKENIZER_JSON = "tokenizer.json"

def download_model(dest_dir="models/tinyllama"):
    os.makedirs(dest_dir, exist_ok=True)
    
    files_to_download = [MODEL_FILE, TOKENIZER_FILE, TOKENIZER_JSON]
    downloaded_paths = []
    
    for file_name in files_to_download:
        dest_path = os.path.join(dest_dir, file_name)
        if os.path.exists(dest_path):
            print(f"File already exists at {dest_path}")
            downloaded_paths.append(dest_path)
            continue
            
        print(f"Downloading {file_name} from {MODEL_REPO}...")
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=file_name,
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded to {path}")
        downloaded_paths.append(path)
        
    return downloaded_paths

if __name__ == "__main__":
    download_model()
