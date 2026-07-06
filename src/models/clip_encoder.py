"""
CLIP-based video encoder for VG-GPLMs
Replaces pre-extracted features with real-time CLIP encoding

Usage:
    from clip_encoder import CLIPVideoEncoder
    
    encoder = CLIPVideoEncoder()
    features = encoder(pixel_values)  # (batch, num_frames, 512)
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


class CLIPVideoEncoder(nn.Module):
    """
    Encodes video frames using CLIP vision encoder
    Outputs vision-language aligned features
    """
    
    def __init__(self, 
                 model_name="openai/clip-vit-base-patch32",
                 freeze_clip=False,
                 output_dim=512):
        """
        Args:
            model_name: CLIP model from Hugging Face
                Options: 'openai/clip-vit-base-patch32' (512-dim)
                        'openai/clip-vit-large-patch14' (768-dim)
            freeze_clip: If True, freeze CLIP weights (faster training)
            output_dim: Output feature dimension
        """
        super().__init__()
        
        print(f"Loading CLIP model: {model_name}")
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Optionally freeze CLIP weights
        if freeze_clip:
            print("Freezing CLIP weights")
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            print("CLIP weights will be fine-tuned")
        
        # Vision encoder extracts features
        self.vision_encoder = self.clip_model.vision_model
        
        # Get CLIP dimension
        clip_dim = self.clip_model.config.vision_config.hidden_size
        print(f"CLIP dimension: {clip_dim}, Output dimension: {output_dim}")
        
        # Optional: Project to different dimension
        self.output_dim = output_dim
        if output_dim != clip_dim:
            self.projection = nn.Linear(clip_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, pixel_values):
        """
        Forward pass through CLIP encoder
        
        Args:
            pixel_values: Tensor of shape (batch, num_frames, 3, 224, 224)
        
        Returns:
            features: Tensor of shape (batch, num_frames, output_dim)
        """
        batch_size, num_frames = pixel_values.shape[:2]
        
        # Reshape to process all frames together
        # (batch * num_frames, 3, 224, 224)
        pixel_values = pixel_values.view(batch_size * num_frames, *pixel_values.shape[2:])
        
        # Extract CLIP vision features
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        
        # Get pooled features (CLS token)
        features = vision_outputs.pooler_output  # (batch * num_frames, clip_dim)
        
        # Project to desired dimension
        features = self.projection(features)
        
        # Reshape back to (batch, num_frames, output_dim)
        features = features.view(batch_size, num_frames, self.output_dim)
        
        return features
    
    def encode_video_from_frames(self, frames):
        """
        Helper method to encode video frames from PIL Images or numpy arrays
        
        Args:
            frames: List of PIL Images or numpy arrays (H, W, 3)
        
        Returns:
            features: Tensor of shape (1, num_frames, output_dim)
        """
        # Preprocess frames
        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs['pixel_values']  # (num_frames, 3, 224, 224)
        
        # Add batch dimension
        pixel_values = pixel_values.unsqueeze(0)  # (1, num_frames, 3, 224, 224)
        
        # Move to same device as model
        pixel_values = pixel_values.to(next(self.parameters()).device)
        
        # Encode
        features = self.forward(pixel_values)
        
        return features
    
    def extract_and_save_features(self, video_path, output_path, num_frames=50):
        """
        Extract CLIP features from video and save as .npy file
        
        Args:
            video_path: Path to video file
            output_path: Path to save features
            num_frames: Number of frames to sample
        
        Returns:
            features: Extracted features (num_frames, output_dim)
        """
        import cv2
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frame indices
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
        
        cap.release()
        
        # Encode frames
        with torch.no_grad():
            features = self.encode_video_from_frames(frames)
        
        # Remove batch dimension and convert to numpy
        features = features.squeeze(0).cpu().numpy()  # (num_frames, output_dim)
        
        # Save
        np.save(output_path, features)
        
        return features


class CLIPTextEncoder(nn.Module):
    """
    Encodes text using CLIP text encoder
    Can be used to align transcript with visual features
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_clip=False):
        super().__init__()
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        self.text_encoder = self.clip_model.text_model
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
        
        Returns:
            features: Text features (batch, hidden_dim)
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.pooler_output


# Example usage
if __name__ == "__main__":
    # Test CLIP encoder
    print("Testing CLIP encoder...")
    
    encoder = CLIPVideoEncoder(
        model_name="openai/clip-vit-base-patch32",
        freeze_clip=False,
        output_dim=512
    )
    
    # Create dummy input
    batch_size = 2
    num_frames = 50
    dummy_frames = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    # Encode
    features = encoder(dummy_frames)
    
    print(f"Input shape: {dummy_frames.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: ({batch_size}, {num_frames}, 512)")
    
    assert features.shape == (batch_size, num_frames, 512), "Shape mismatch!"
    print("✓ CLIP encoder works correctly!")