#!/usr/bin/env python3
"""
Single Image Friction Inference
Predict friction heatmap for new racing track images.
"""

from pathlib import Path
from typing import Tuple, Optional
import warnings

import cv2
import numpy as np
import joblib
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as colors

warnings.filterwarnings('ignore')


class FrictionInference:
    """Predict friction heatmaps from racing track images"""
    
    def __init__(self):
        self.model_file = Path("models/rf_model_regularized.joblib")
        self.encoder_file = Path("models/label_encoder_regularized.joblib")
        self.output_dir = Path("results/overlays")
        
        # Patch configuration (same as training)
        self.patch_size = 64
        self.patch_stride = 32
        
        # Load trained model
        self.rf = joblib.load(self.model_file)
        self.label_encoder = joblib.load(self.encoder_file)
        
    def preprocess_image(self, image: np.ndarray, target_size: int = 1280) -> np.ndarray:
        """Preprocess image (resize + CLAHE)"""
        h, w = image.shape[:2]
        
        # Resize maintaining aspect ratio
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Apply CLAHE to improve contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image
    
    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract features from a single patch (same as training)"""
        # LBP features
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        from skimage import feature
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        
        # Texture features
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        sobel_features = [np.mean(sobel_mag), np.std(sobel_mag)]
        gray_features = [np.mean(gray), np.std(gray)]
        
        # LAB color features
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        lab_features = [
            np.mean(lab[:,:,0]), np.mean(lab[:,:,1]), np.mean(lab[:,:,2]),
            np.std(lab[:,:,0]), np.std(lab[:,:,1]), np.std(lab[:,:,2])
        ]
        
        return np.concatenate([lbp_hist, sobel_features, gray_features, lab_features])
    
    def extract_grid_patches(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract patches in grid and predict friction"""
        h, w = image.shape[:2]
        patch_predictions = []
        patch_positions = []
        
        for y in range(0, h - self.patch_size + 1, self.patch_stride):
            for x in range(0, w - self.patch_size + 1, self.patch_stride):
                # Extract patch
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                # Extract features and predict
                features = self.extract_patch_features(patch).reshape(1, -1)
                prob = self.rf.predict_proba(features)[0]
                
                # Store prediction and position
                patch_predictions.append(prob)
                patch_positions.append([x + self.patch_size//2, y + self.patch_size//2])
        
        return np.array(patch_positions), np.array(patch_predictions)
    
    def create_heatmap(self, image: np.ndarray, positions: np.ndarray, 
                      predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create smooth friction heatmap"""
        h, w = image.shape[:2]
        
        # Get low friction probability (assuming it's class index for 'Low')
        low_idx = list(self.label_encoder.classes_).index('Low')
        low_friction_scores = predictions[:, low_idx]
        
        # Create grid for interpolation
        xi = np.arange(0, w, 1)
        yi = np.arange(0, h, 1)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate patch scores to full image
        heatmap = griddata(
            positions, low_friction_scores,
            (xi_grid, yi_grid), 
            method='cubic', 
            fill_value=0
        )
        
        # Clip values to [0, 1]
        heatmap = np.clip(heatmap, 0, 1)
        
        return heatmap, low_friction_scores
    
    def generate_overlay(self, image: np.ndarray, heatmap: np.ndarray, 
                        alpha: float = 0.6) -> np.ndarray:
        """Generate colored overlay with proper friction visualization"""
        # Create proper friction colormap
        # heatmap values: 0.0 = high friction (safe), 1.0 = low friction (dangerous)
        
        colored_heatmap = np.zeros((*heatmap.shape, 3), dtype=np.float32)
        
        # High friction areas (0.0 - 0.3) = GREEN
        high_friction_mask = heatmap <= 0.3
        colored_heatmap[high_friction_mask] = [0, 1, 0]  # Pure green
        
        # Medium friction areas (0.3 - 0.7) = YELLOW  
        medium_friction_mask = (heatmap > 0.3) & (heatmap <= 0.7)
        colored_heatmap[medium_friction_mask] = [0, 1, 1]  # Yellow (green + red)
        
        # Low friction areas (0.7 - 1.0) = RED
        low_friction_mask = heatmap > 0.7
        colored_heatmap[low_friction_mask] = [0, 0, 1]  # Pure red
        
        # Smooth transitions
        # Green to Yellow transition
        transition_mask = (heatmap > 0.2) & (heatmap <= 0.4)
        transition_factor = (heatmap[transition_mask] - 0.2) / 0.2
        colored_heatmap[transition_mask, 0] = 0  # Blue = 0
        colored_heatmap[transition_mask, 1] = 1  # Green = 1
        colored_heatmap[transition_mask, 2] = transition_factor  # Red increases
        
        # Yellow to Red transition  
        transition_mask = (heatmap > 0.6) & (heatmap <= 0.8)
        transition_factor = (heatmap[transition_mask] - 0.6) / 0.2
        colored_heatmap[transition_mask, 0] = 0  # Blue = 0
        colored_heatmap[transition_mask, 1] = 1 - transition_factor  # Green decreases
        colored_heatmap[transition_mask, 2] = 1  # Red = 1
        
        # Convert to BGR and scale to 0-255
        colored_heatmap_bgr = (colored_heatmap * 255).astype(np.uint8)
        
        # Only apply heatmap where we have predictions (non-zero areas)
        mask = heatmap > 0.1
        overlay = image.copy()
        overlay[mask] = cv2.addWeighted(
            image[mask], 1-alpha, 
            colored_heatmap_bgr[mask], alpha, 0
        )
        
        return overlay
    
    def predict_image(self, image_path: str, save_overlay: bool = True) -> dict:
        """Complete inference pipeline for single image"""
        image_path = Path(image_path)
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        processed_image = self.preprocess_image(image)
        
        # Extract patches and predict
        positions, predictions = self.extract_grid_patches(processed_image)
        
        # Create heatmap
        heatmap, patch_scores = self.create_heatmap(processed_image, positions, predictions)
        
        # Generate overlay
        overlay = self.generate_overlay(processed_image, heatmap)
        
        # Save overlay if requested
        if save_overlay:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = self.output_dir / f"{image_path.stem}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)
        
        # Calculate summary statistics
        low_patches = np.sum(patch_scores > 0.7)
        total_patches = len(patch_scores)
        percent_low = low_patches / total_patches if total_patches > 0 else 0
        
        worst_idx = np.argmax(patch_scores)
        worst_position = positions[worst_idx] if len(positions) > 0 else [0, 0]
        
        return {
            'image_path': str(image_path),
            'overlay_path': str(overlay_path) if save_overlay else None,
            'heatmap': heatmap,
            'patch_scores': patch_scores,
            'patch_positions': positions,
            'summary': {
                'total_patches': total_patches,
                'low_friction_patches': low_patches,
                'percent_low_friction': percent_low,
                'worst_position': worst_position.tolist(),
                'max_low_score': float(np.max(patch_scores)) if len(patch_scores) > 0 else 0
            }
        }


def main():
    """Test inference on sample image"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py <image_path>")
        print("Example: python src/infer.py data/cvat_coco/images/track_001.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Initialize inference
    predictor = FrictionInference()
    
    # Predict
    print(f"Processing: {image_path}")
    result = predictor.predict_image(image_path)
    
    # Print summary
    summary = result['summary']
    print(f"Analysis complete!")
    print(f"Total patches: {summary['total_patches']}")
    print(f"Low friction patches: {summary['low_friction_patches']}")
    print(f"Percent low friction: {summary['percent_low_friction']:.1%}")
    print(f"Max danger score: {summary['max_low_score']:.3f}")
    print(f"Overlay saved: {result['overlay_path']}")


if __name__ == "__main__":
    main()