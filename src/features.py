#!/usr/bin/env python3
"""
Feature Extraction Pipeline
Extract LBP + texture features from patches for RandomForest training.
"""

from pathlib import Path
from typing import Tuple, List
import warnings

import cv2
import numpy as np
import pandas as pd
from skimage import feature
from tqdm import tqdm

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract LBP and texture features from patches"""
    
    def __init__(self):
        self.patch_csv = Path("data/patches/patches.csv")
        self.output_file = Path("data/features/patch_features.npz")
        
    def extract_lbp_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract LBP histogram features"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
        return hist
    
    def extract_texture_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract additional texture features"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        sobel_features = [
            np.mean(sobel_mag),
            np.std(sobel_mag)
        ]
        
        # Grayscale statistics
        gray_features = [
            np.mean(gray),
            np.std(gray)
        ]
        
        # LAB color statistics
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        lab_features = [
            np.mean(lab[:,:,0]),  # L
            np.mean(lab[:,:,1]),  # A
            np.mean(lab[:,:,2]),  # B
            np.std(lab[:,:,0]),   # L std
            np.std(lab[:,:,1]),   # A std
            np.std(lab[:,:,2])    # B std
        ]
        
        return np.array(sobel_features + gray_features + lab_features)
    
    def extract_patch_features(self, patch_path: str) -> np.ndarray:
        """Extract all features from a single patch"""
        patch = cv2.imread(patch_path)
        if patch is None:
            return np.zeros(20)  # Return zeros if image can't be loaded
        
        # LBP features (10 dims)
        lbp_features = self.extract_lbp_features(patch)
        
        # Texture features (10 dims: 2 sobel + 2 gray + 6 lab)
        texture_features = self.extract_texture_features(patch)
        
        # Combine all features
        return np.concatenate([lbp_features, texture_features])
    
    def run(self) -> None:
        """Extract features from all patches"""
        # Load patch metadata
        patches_df = pd.read_csv(self.patch_csv)
        
        print(f"Extracting features from {len(patches_df)} patches...")
        
        # Extract features
        features_list = []
        labels_list = []
        ids_list = []
        
        for idx, row in tqdm(patches_df.iterrows(), total=len(patches_df), desc="Features"):
            features = self.extract_patch_features(row['path'])
            features_list.append(features)
            labels_list.append(row['label'])
            ids_list.append(row['patch_id'])
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        ids = np.array(ids_list)
        
        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save features
        np.savez_compressed(
            self.output_file,
            X=X,
            y=y,
            ids=ids,
            feature_names=np.array([
                'lbp_0', 'lbp_1', 'lbp_2', 'lbp_3', 'lbp_4',
                'lbp_5', 'lbp_6', 'lbp_7', 'lbp_8', 'lbp_9',
                'sobel_mean', 'sobel_std',
                'gray_mean', 'gray_std',
                'lab_l_mean', 'lab_a_mean', 'lab_b_mean',
                'lab_l_std', 'lab_a_std', 'lab_b_std'
            ])
        )
        
        print(f"Features saved to {self.output_file}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Print label distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")


def main():
    """Main entry point"""
    extractor = FeatureExtractor()
    extractor.run()


if __name__ == "__main__":
    main()