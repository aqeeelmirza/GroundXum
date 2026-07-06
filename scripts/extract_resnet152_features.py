#!/usr/bin/env python3
"""
Extract ResNet-152 features from videos (for CNN baseline comparison)
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from torchvision import transforms, models


class ResNet152Extractor:
    """Extract ResNet-152 features from videos"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading ResNet-152...")
        # Load pre-trained ResNet-152
        self.model = models.resnet152(pretrained=True).to(self.device)
        
        # Remove final classification layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ ResNet-152 loaded on {self.device}")
        print(f"✓ Feature dimension: 2048")
    
    def extract_features(self, frames, batch_size=8):
        """Extract features from list of PIL Images"""
        features = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_features = self._extract_batch(batch_frames)
            features.append(batch_features)
        
        return np.vstack(features)
    
    def _extract_batch(self, frames):
        """Extract features for a batch"""
        # Preprocess frames
        batch_tensors = torch.stack([self.transform(frame) for frame in frames])
        batch_tensors = batch_tensors.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensors)
            # Remove spatial dimensions: (batch, 2048, 1, 1) -> (batch, 2048)
            features = features.squeeze(-1).squeeze(-1)
        
        return features.cpu().numpy()


def extract_frames_from_video(video_path, num_frames=50):
    """Extract uniformly sampled frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames if len(frames) > 0 else None


def main():
    parser = argparse.ArgumentParser(description='Extract ResNet-152 features')
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num-frames', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = ResNet152Extractor(device=args.device)
    
    # Find videos
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    for ext in video_exts:
        video_files.extend(list(video_dir.glob(f'**/*{ext}')))
    
    print(f"\nFound {len(video_files)} videos\n")
    
    # Process
    success = 0
    for video_path in tqdm(video_files, desc="Extracting ResNet-152 features"):
        video_id = video_path.stem
        output_path = output_dir / f"{video_id}.npy"
        
        if output_path.exists():
            success += 1
            continue
        
        frames = extract_frames_from_video(video_path, args.num_frames)
        if frames is None:
            continue
        
        try:
            features = extractor.extract_features(frames, args.batch_size)
            
            # Pad/truncate
            if len(features) < args.num_frames:
                padding = np.zeros((args.num_frames - len(features), 2048))
                features = np.vstack([features, padding])
            else:
                features = features[:args.num_frames]
            
            np.save(output_path, features)
            success += 1
        except Exception as e:
            print(f"\nError: {video_id}: {e}")
    
    print(f"\n✅ Successfully processed {success}/{len(video_files)} videos")
    
    # Verify
    if success > 0:
        sample = np.load(list(output_dir.glob('*.npy'))[0])
        print(f"\n✓ Sample shape: {sample.shape}")
        print(f"  Expected: ({args.num_frames}, 2048)")


if __name__ == '__main__':
    main()