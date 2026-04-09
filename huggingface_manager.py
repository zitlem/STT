#!/usr/bin/env python3
"""
Hugging Face Model Management Script
Handles downloading and uploading models
"""

import os
import json
from huggingface_hub import HfApi, HfFolder, Repository, snapshot_download
from pathlib import Path
import argparse
import zipfile
import shutil

class HuggingFaceManager:
    def __init__(self):
        self.api = HfApi()
        self.token = None
        
    def login(self, token=None):
        """Login to Hugging Face"""
        if token:
            self.token = token
            HfFolder.save_token(token)
        else:
            # Try to get existing token
            self.token = HfFolder.get_token()
            
        if not self.token:
            raise ValueError("No Hugging Face token provided. Use --token or run `huggingface-cli login`")
            
        print(f"✓ Logged in to Hugging Face")
        return self.token
    
    def download_model(self, model_id, local_dir=None, cache_dir=None):
        """Download a model from Hugging Face"""
        print(f"Downloading model: {model_id}")
        
        try:
            if local_dir:
                # Download to specific directory
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    cache_dir=cache_dir
                )
                print(f"✓ Model downloaded to: {local_dir}")
            else:
                # Download to default cache
                path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
                print(f"✓ Model downloaded to cache: {path}")
                
            return True
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            return False
    
    def upload_model(self, model_path, repo_id, private=False, commit_message="Upload model"):
        """Upload a local model to Hugging Face"""
        if not self.token:
            self.login()
            
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        print(f"Uploading model from: {model_path}")
        print(f"Target repository: {repo_id}")
        
        try:
            # Create repository if it doesn't exist
            try:
                self.api.create_repo(
                    repo_id=repo_id,
                    token=self.token,
                    private=private,
                    repo_type="model"
                )
                print(f"✓ Created repository: {repo_id}")
            except Exception:
                print(f"Repository {repo_id} already exists")
            
            # Upload files
            if model_path.is_file():
                # Single file upload
                self.api.upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo=model_path.name,
                    repo_id=repo_id,
                    token=self.token,
                    commit_message=commit_message
                )
            else:
                # Directory upload
                self.api.upload_folder(
                    folder_path=str(model_path),
                    repo_id=repo_id,
                    token=self.token,
                    commit_message=commit_message
                )
            
            print(f"✓ Model uploaded successfully to: https://huggingface.co/{repo_id}")
            return True
            
        except Exception as e:
            print(f"✗ Error uploading model: {e}")
            return False
    
    def list_local_models(self, models_dir="./models"):
        """List local models in directory"""
        models_path = Path(models_dir)
        if not models_path.exists():
            print(f"Models directory not found: {models_dir}")
            return []
        
        models = []
        for item in models_path.iterdir():
            if item.is_dir():
                # Check for model files
                model_files = list(item.glob("*.bin")) + list(item.glob("*.safetensors")) + list(item.glob("*.pt"))
                if model_files:
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "files": [f.name for f in model_files],
                        "size_mb": sum(f.stat().st_size for f in model_files) / (1024 * 1024)
                    })
        
        return models
    
    def create_model_card(self, repo_id, model_type="whisper", description="Custom speech-to-text model"):
        """Create a basic model card for the uploaded model"""
        model_card = f"""---
language: en
license: mit
library_name: transformers
tags:
- automatic-speech-recognition
- {model_type}
- speech-to-text
---

# {repo_id}

{description}

## Model Details
- **Model Type**: {model_type}
- **Language**: English
- **License**: MIT

## Usage

```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="{repo_id}")
result = transcriber("audio.wav")
print(result["text"])
```

## Training

This model was custom trained for speech-to-text tasks.
"""
        return model_card

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Model Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a model')
    download_parser.add_argument('model_id', help='Model ID (e.g., openai/whisper-base)')
    download_parser.add_argument('--local-dir', help='Local directory to save model')
    download_parser.add_argument('--cache-dir', help='Cache directory')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload a model')
    upload_parser.add_argument('model_path', help='Path to model file or directory')
    upload_parser.add_argument('repo_id', help='Target repository ID (e.g., username/model-name)')
    upload_parser.add_argument('--private', action='store_true', help='Make repository private')
    upload_parser.add_argument('--commit', default="Upload model", help='Commit message')
    upload_parser.add_argument('--token', help='Hugging Face access token')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List local models')
    list_parser.add_argument('--models-dir', default="./models", help='Models directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = HuggingFaceManager()
    
    if args.command == 'download':
        success = manager.download_model(
            model_id=args.model_id,
            local_dir=args.local_dir,
            cache_dir=args.cache_dir
        )
        
    elif args.command == 'upload':
        manager.login(args.token)
        success = manager.upload_model(
            model_path=args.model_path,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit
        )
        
        # Create model card
        if success:
            model_card = manager.create_model_card(args.repo_id)
            print("\n--- Model Card ---")
            print(model_card)
            
    elif args.command == 'list':
        models = manager.list_local_models(args.models_dir)
        if models:
            print(f"Found {len(models)} local models:")
            for model in models:
                print(f"  • {model['name']} ({model['size_mb']:.1f} MB)")
                print(f"    Path: {model['path']}")
                print(f"    Files: {', '.join(model['files'])}")
        else:
            print("No models found")

if __name__ == "__main__":
    main()