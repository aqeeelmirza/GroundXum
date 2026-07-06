#!/usr/bin/env python3
"""
Vision-Language Feature Extractor for Video Summarization
=========================================================

This script extracts visual features from videos using various Vision-Language models.
Supports: CLIP, BLIP, BLIP-2, ALBEF, EVA-CLIP, and SigLIP.

Author: Muhammad Aqeel
Date: December 2024

Usage:
------
# Extract CLIP ViT-B/32 features (512-dim)
python extract_vl_features.py \
    --video-dir /path/to/videos \
    --output-dir /path/to/features \
    --model clip-vit-b32

# Extract BLIP-2 features (768-dim)
python extract_vl_features.py \
    --video-dir /path/to/videos \
    --output-dir /path/to/features \
    --model blip2-opt-2.7b \
    --num-frames 50

Requirements:
-------------
pip install torch transformers pillow opencv-python tqdm numpy

Output:
-------
For each video (e.g., video123.mp4), creates video123.npy with shape:
    (num_frames, feature_dim)
    e.g., (50, 512) for CLIP ViT-B/32
"""

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    # CLIP Models (OpenAI)
    'clip-vit-b32': {
        'name': 'openai/clip-vit-base-patch32',
        'dim': 512,
        'type': 'clip',
        'description': 'CLIP ViT-B/32 (your baseline)'
    },
    'clip-vit-b16': {
        'name': 'openai/clip-vit-base-patch16',
        'dim': 512,
        'type': 'clip',
        'description': 'CLIP ViT-B/16 (finer patches)'
    },
    'clip-vit-l14': {
        'name': 'openai/clip-vit-large-patch14',
        'dim': 768,
        'type': 'clip',
        'description': 'CLIP ViT-L/14 (larger model)'
    },
    
    # BLIP Models (Salesforce)
    'blip-base': {
        'name': 'Salesforce/blip-image-captioning-base',
        'dim': 768,
        'type': 'blip',
        'description': 'BLIP Base (unified VL model)'
    },
    'blip-large': {
        'name': 'Salesforce/blip-image-captioning-large',
        'dim': 768,
        'type': 'blip',
        'description': 'BLIP Large'
    },
    
    # BLIP-2 Models (Salesforce) - State-of-the-art
    'blip2-opt-2.7b': {
        'name': 'Salesforce/blip2-opt-2.7b',
        'dim': 768,
        'type': 'blip2',
        'description': 'BLIP-2 with OPT-2.7B (recommended for comparison)'
    },
    'blip2-flan-t5-xl': {
        'name': 'Salesforce/blip2-flan-t5-xl',
        'dim': 768,
        'type': 'blip2',
        'description': 'BLIP-2 with Flan-T5-XL'
    },
    
    # EVA-CLIP (improved CLIP)
    'eva-clip': {
        'name': 'QuanSun/EVA-CLIP',
        'dim': 512,
        'type': 'eva-clip',
        'description': 'EVA-CLIP (improved CLIP training)'
    },
    
    # SigLIP (Google's improved CLIP)
    'siglip-base': {
        'name': 'google/siglip-base-patch16-224',
        'dim': 768,
        'type': 'siglip',
        'description': 'SigLIP Base (improved contrastive learning)'
    },
}


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def extract_frames_from_video(video_path: Path, num_frames: int = 50) -> Optional[List[Image.Image]]:
    """
    Extract uniformly sampled frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (uniform sampling)
    
    Returns:
        List of PIL Images, or None if video cannot be opened
    
    Process:
        1. Open video with OpenCV
        2. Calculate uniform frame indices
        3. Extract and convert frames (BGR -> RGB)
        4. Return as PIL Images
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return None
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate uniform sampling indices
    if total_frames <= num_frames:
        # Video has fewer frames than requested, use all frames
        frame_indices = list(range(total_frames))
    else:
        # Uniformly sample num_frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    
    if len(frames) == 0:
        print(f"⚠️  No frames extracted from: {video_path}")
        return None
    
    return frames


# ============================================================================
# FEATURE EXTRACTORS
# ============================================================================

class VisionLanguageExtractor:
    """
    Unified interface for extracting features from Vision-Language models.
    
    Supports:
        - CLIP (OpenAI)
        - BLIP (Salesforce)
        - BLIP-2 (Salesforce)
        - EVA-CLIP
        - SigLIP (Google)
    
    All models output normalized feature vectors.
    """
    
    def __init__(self, model_key: str, device: str = 'cuda'):
        """
        Initialize the feature extractor.
        
        Args:
            model_key: Key from MODEL_CONFIGS
            device: 'cuda' or 'cpu'
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        self.config = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.model_type = self.config['type']
        self.feature_dim = self.config['dim']
        
        # Setup device
        self.device = device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
        
        # Load model
        print(f"\n{'='*60}")
        print(f"Loading: {self.config['description']}")
        print(f"Model: {self.config['name']}")
        print(f"Type: {self.model_type}")
        print(f"Feature dim: {self.feature_dim}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        self.model, self.processor = self._load_model()
        self.model.eval()  # Set to evaluation mode
        
        print("✅ Model loaded successfully!\n")
    
    def _load_model(self) -> Tuple:
        """Load model and processor based on type."""
        model_name = self.config['name']
        
        try:
            if self.model_type == 'clip':
                from transformers import CLIPModel, CLIPProcessor
                model = CLIPModel.from_pretrained(model_name).to(self.device)
                processor = CLIPProcessor.from_pretrained(model_name)
                return model, processor
            
            elif self.model_type == 'blip':
                from transformers import BlipModel, BlipProcessor
                model = BlipModel.from_pretrained(model_name).to(self.device)
                processor = BlipProcessor.from_pretrained(model_name)
                return model, processor
            
            elif self.model_type == 'blip2':
                from transformers import Blip2Model, Blip2Processor
                # Use float16 for BLIP-2 to save memory
                model = Blip2Model.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                ).to(self.device)
                processor = Blip2Processor.from_pretrained(model_name)
                return model, processor
            
            elif self.model_type == 'eva-clip':
                from transformers import CLIPModel, CLIPProcessor
                model = CLIPModel.from_pretrained(model_name).to(self.device)
                processor = CLIPProcessor.from_pretrained(model_name)
                return model, processor
            
            elif self.model_type == 'siglip':
                from transformers import AutoModel, AutoProcessor
                model = AutoModel.from_pretrained(model_name).to(self.device)
                processor = AutoProcessor.from_pretrained(model_name)
                return model, processor
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nTry installing latest transformers:")
            print("  pip install --upgrade transformers torch")
            raise
    
    def extract_features(self, frames: List[Image.Image], batch_size: int = 8) -> np.ndarray:
        """
        Extract features from a list of frames.
        
        Args:
            frames: List of PIL Images
            batch_size: Number of frames to process at once
        
        Returns:
            numpy array of shape (num_frames, feature_dim)
        
        Process:
            1. Split frames into batches
            2. Preprocess each batch
            3. Extract features (forward pass)
            4. Normalize features
            5. Concatenate all batches
        """
        all_features = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_features = self._extract_batch(batch_frames)
            all_features.append(batch_features)
        
        # Concatenate all batches
        features = np.vstack(all_features)
        return features
    
    def _extract_batch(self, frames: List[Image.Image]) -> np.ndarray:
        """Extract features for a single batch of frames."""
        
        # Preprocess frames
        inputs = self.processor(images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features based on model type
        with torch.no_grad():
            if self.model_type == 'clip' or self.model_type == 'eva-clip':
                # CLIP: Use get_image_features() method
                features = self.model.get_image_features(**inputs)
                # L2 normalization (CLIP standard)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            elif self.model_type == 'blip':
                # BLIP: Extract from vision_model, use pooler_output (CLS token)
                outputs = self.model.vision_model(**inputs)
                features = outputs.pooler_output
                # Normalize
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            elif self.model_type == 'blip2':
                # BLIP-2: Extract vision features before Q-Former
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs['pixel_values']
                )
                features = vision_outputs.pooler_output
                # Normalize
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            elif self.model_type == 'siglip':
                # SigLIP: Similar to CLIP
                outputs = self.model.vision_model(**inputs)
                features = outputs.pooler_output
                # Normalize
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Convert to numpy (handle float16 for BLIP-2)
        features = features.cpu().float().numpy()
        return features


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_videos(
    video_dir: Path,
    output_dir: Path,
    extractor: VisionLanguageExtractor,
    num_frames: int = 50,
    batch_size: int = 8,
    skip_existing: bool = True
):
    """
    Process all videos in a directory and extract features.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save features
        extractor: VisionLanguageExtractor instance
        num_frames: Number of frames to extract per video
        batch_size: Batch size for feature extraction
        skip_existing: Skip videos that already have features
    """
    # Find all video files (search recursively)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f'**/*{ext}')))
    
    print(f"📁 Found {len(video_files)} video files in {video_dir}\n")
    
    if len(video_files) == 0:
        print("❌ No videos found!")
        print(f"   Searched for extensions: {video_extensions}")
        return
    
    # Process each video
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for video_path in tqdm(video_files, desc="Extracting features"):
        # Output path: video_id.npy
        video_id = video_path.stem
        output_path = output_dir / f"{video_id}.npy"
        
        # Skip if already exists
        if skip_existing and output_path.exists():
            skip_count += 1
            continue
        
        try:
            # Step 1: Extract frames
            frames = extract_frames_from_video(video_path, num_frames=num_frames)
            if frames is None:
                error_count += 1
                continue
            
            # Step 2: Extract features
            features = extractor.extract_features(frames, batch_size=batch_size)
            
            # Step 3: Pad or truncate to exactly num_frames
            current_frames = len(features)
            if current_frames < num_frames:
                # Pad with zeros
                padding = np.zeros((num_frames - current_frames, extractor.feature_dim))
                features = np.vstack([features, padding])
            elif current_frames > num_frames:
                # Truncate
                features = features[:num_frames]
            
            # Step 4: Save
            np.save(output_path, features)
            success_count += 1
        
        except Exception as e:
            error_count += 1
            tqdm.write(f"❌ Error processing {video_id}: {str(e)}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {success_count}/{len(video_files)}")
    print(f"⏭️  Skipped (already exists): {skip_count}")
    print(f"❌ Errors: {error_count}")
    print(f"{'='*60}\n")


def verify_features(output_dir: Path, expected_shape: Tuple[int, int]):
    """Verify that extracted features have correct shape."""
    feature_files = list(output_dir.glob('*.npy'))
    
    if len(feature_files) == 0:
        print("⚠️  No feature files found to verify")
        return
    
    # Check a few random files
    import random
    sample_files = random.sample(feature_files, min(5, len(feature_files)))
    
    print("🔍 Verifying feature files...")
    all_correct = True
    
    for fpath in sample_files:
        features = np.load(fpath)
        if features.shape != expected_shape:
            print(f"❌ {fpath.name}: shape {features.shape}, expected {expected_shape}")
            all_correct = False
        else:
            print(f"✅ {fpath.name}: {features.shape}")
    
    if all_correct:
        print(f"\n🎉 All features have correct shape: {expected_shape}")
    else:
        print(f"\n⚠️  Some features have incorrect shape!")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract Vision-Language features from videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract CLIP ViT-B/32 features (your baseline)
  python extract_vl_features.py --video-dir ./videos --output-dir ./features/clip-vit-b32 --model clip-vit-b32
  
  # Extract BLIP-2 features for comparison
  python extract_vl_features.py --video-dir ./videos --output-dir ./features/blip2 --model blip2-opt-2.7b
  
  # Extract with 100 frames per video
  python extract_vl_features.py --video-dir ./videos --output-dir ./features --model clip-vit-b32 --num-frames 100

Available models:
""" + '\n'.join([f"  - {k}: {v['description']} ({v['dim']}-dim)" for k, v in MODEL_CONFIGS.items()])
    )
    
    parser.add_argument('--video-dir', type=str, required=True,
                        help='Directory containing video files (searches recursively)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save .npy feature files')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Vision-language model to use')
    parser.add_argument('--num-frames', type=int, default=50,
                        help='Number of frames to extract per video (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for feature extraction (default: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Reprocess videos even if features already exist')
    
    args = parser.parse_args()
    
    # Setup paths
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = VisionLanguageExtractor(args.model, device=args.device)
    
    # Process videos
    process_videos(
        video_dir=video_dir,
        output_dir=output_dir,
        extractor=extractor,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing
    )
    
    # Verify features
    expected_shape = (args.num_frames, extractor.feature_dim)
    verify_features(output_dir, expected_shape)
    
    # Print usage instructions
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"Features saved to: {output_dir}")
    print(f"Feature dimension: {extractor.feature_dim}")
    print(f"\nTo use these features in training:")
    print(f"  1. Update your training script:")
    print(f"     -visual_hidden_size {extractor.feature_dim}")
    print(f"  2. Point to feature directory:")
    print(f"     -video_feature_path {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()