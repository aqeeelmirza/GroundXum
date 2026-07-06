"""
Video frame loader for real-time CLIP encoding
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


class VideoFrameLoader:
    """
    Loads video frames for CLIP encoding
    """
    
    def __init__(self, video_dir, num_frames=50, frame_size=(224, 224)):
        """
        Args:
            video_dir: Directory containing videos
            num_frames: Number of frames to sample per video
            frame_size: Target frame size (H, W)
        """
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
    
    def load_video_frames(self, video_id):
        """
        Load and sample frames from video
        
        Args:
            video_id: Video ID (e.g., 'video7000')
        
        Returns:
            frames: List of PIL Images
        """
        # Try common video extensions
        video_path = None
        for ext in ['.mp4', '.avi', '.mkv', '.webm']:
            path = self.video_dir / f"{video_id}{ext}"
            if path.exists():
                video_path = path
                break
        
        if video_path is None:
            raise FileNotFoundError(f"Video not found: {video_id}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frame indices uniformly
        if total_frames <= self.num_frames:
            # Use all frames if video is short
            frame_indices = list(range(total_frames))
        else:
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                frame = Image.fromarray(frame)
                
                # Resize
                frame = frame.resize(self.frame_size)
                
                frames.append(frame)
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', self.frame_size))
        
        return frames[:self.num_frames]
    
    def load_frames_as_tensor(self, video_id):
        """
        Load frames as PyTorch tensor
        
        Returns:
            tensor: (num_frames, 3, H, W)
        """
        frames = self.load_video_frames(video_id)
        
        # Convert to tensor
        frames_array = np.stack([np.array(f) for f in frames])  # (num_frames, H, W, 3)
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        
        # Permute to (num_frames, 3, H, W)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        
        return frames_tensor