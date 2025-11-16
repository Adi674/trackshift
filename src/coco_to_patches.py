#!/usr/bin/env python3
"""
CVAT COCO to Training Patches Pipeline
Converts CVAT COCO 1.0 export to labeled 64x64 patches for RandomForest training.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')


@dataclass
class Config:
    """Configuration parameters"""
    patch_size: int = 64
    patch_stride: int = 32
    min_coverage: float = 0.4
    
    cvat_dir: Path = Path("data/cvat_coco")
    output_dir: Path = Path("data/patches")
    
    @property
    def images_dir(self) -> Path:
        return Path("data/segmented/segmented")  # Use segmented track images
    
    @property
    def annotations_file(self) -> Path:
        return self.cvat_dir / "annotations" / "instances_default.json"


class COCOProcessor:
    """Fast COCO to patches converter"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
    def load_coco_data(self) -> Tuple[Dict, List, Dict]:
        """Load COCO annotations efficiently"""
        with open(self.config.annotations_file, 'r') as f:
            data = json.load(f)
        
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        images = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        
        return images, annotations, categories
    
    def create_mask(self, shape: Tuple[int, int], annotation: Dict) -> np.ndarray:
        """Create binary mask from annotation"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        if 'segmentation' in annotation and annotation['segmentation']:
            seg = annotation['segmentation']
            # Handle different segmentation formats
            if isinstance(seg, list) and len(seg) > 0:
                if isinstance(seg[0], list) and len(seg[0]) >= 6:
                    # Polygon format: [[x1,y1,x2,y2,...]]
                    points = np.array(seg[0]).reshape(-1, 2).astype(np.int32)
                    points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
                    cv2.fillPoly(mask, [points], 255)
                elif isinstance(seg[0], (int, float)) and len(seg) >= 6:
                    # Flattened polygon: [x1,y1,x2,y2,...]
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
                    cv2.fillPoly(mask, [points], 255)
        
        # Fallback to bbox if segmentation failed or empty
        if np.sum(mask) == 0 and 'bbox' in annotation and annotation['bbox']:
            x, y, w, h = map(int, annotation['bbox'])
            x = max(0, min(x, shape[1]))
            y = max(0, min(y, shape[0]))
            w = max(0, min(w, shape[1] - x))
            h = max(0, min(h, shape[0] - y))
            if w > 0 and h > 0:
                mask[y:y+h, x:x+w] = 255
        
        return mask
    
    def extract_patches(self, image_path: Path, annotations: List[Dict], 
                       categories: Dict, image_id: int) -> List[Dict]:
        """Extract patches from single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            return []
        
        h, w = image.shape[:2]
        if h < self.config.patch_size or w < self.config.patch_size:
            return []
        
        # Create friction masks
        friction_masks = {}
        for ann in annotations:
            if ann['image_id'] == image_id:
                cat_name = categories[ann['category_id']]
                if cat_name not in friction_masks:
                    friction_masks[cat_name] = np.zeros((h, w), dtype=np.uint8)
                
                mask = self.create_mask((h, w), ann)
                friction_masks[cat_name] = cv2.bitwise_or(friction_masks[cat_name], mask)
        
        if not friction_masks:
            return []
        
        # Extract grid patches
        patches = []
        patch_id = 0
        patch_area = self.config.patch_size ** 2
        
        for y in range(0, h - self.config.patch_size + 1, self.config.patch_stride):
            for x in range(0, w - self.config.patch_size + 1, self.config.patch_stride):
                # Find dominant class
                best_label = None
                best_coverage = 0.0
                
                for friction_type, mask in friction_masks.items():
                    patch_mask = mask[y:y+self.config.patch_size, x:x+self.config.patch_size]
                    coverage = np.sum(patch_mask > 0) / patch_area
                    
                    if coverage > best_coverage and coverage >= self.config.min_coverage:
                        best_coverage = coverage
                        best_label = friction_type
                
                if best_label:
                    # Additional validation for segmented images
                    # Check if patch is mostly track surface (not black background)
                    patch_rgb = image[y:y+self.config.patch_size, x:x+self.config.patch_size]
                    
                    # Calculate non-black pixel ratio
                    gray_patch = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
                    non_black_pixels = np.sum(gray_patch > 10)  # Not pure black
                    total_pixels = gray_patch.size
                    track_ratio = non_black_pixels / total_pixels
                    
                    # Only save patches that are mostly track surface
                    if track_ratio >= 0.7:  # At least 70% track surface
                        # Save patch
                        patch = image[y:y+self.config.patch_size, x:x+self.config.patch_size]
                        patch_file = f"{image_path.stem}_patch_{patch_id:04d}.jpg"
                        patch_path = self.config.output_dir / "images" / patch_file
                        
                        cv2.imwrite(str(patch_path), patch)
                        
                        patches.append({
                            'patch_id': f"{image_path.stem}_patch_{patch_id:04d}",
                            'path': str(patch_path),
                            'image_id': image_path.stem,
                            'x1': x, 'y1': y,
                            'x2': x + self.config.patch_size,
                            'y2': y + self.config.patch_size,
                            'label': best_label,
                            'coverage': best_coverage,
                            'track_ratio': track_ratio
                        })
                        patch_id += 1
        
        return patches
    
    def run(self) -> pd.DataFrame:
        """Execute pipeline"""
        # Setup
        (self.config.output_dir / "images").mkdir(parents=True, exist_ok=True)
        
        # Load data
        images, annotations, categories = self.load_coco_data()
        
        # Process images
        all_patches = []
        for image_id, image_info in tqdm(images.items(), desc="Processing"):
            original_name = image_info['file_name']
            
            # Convert original COCO filename to segmented filename
            # Original: "WhatsApp Image 2025-11-15 at 22.31.02_84a17888.jpg"
            # Segmented: "WhatsApp Image 2025-11-15 at 22.31.02_84a17888_color_segmented.jpg"
            original_stem = Path(original_name).stem
            segmented_name = f"{original_stem}_color_segmented.jpg"
            image_path = self.config.images_dir / segmented_name
            
            # Fallback: try preprocessed color version if segmented doesn't exist
            if not image_path.exists():
                color_name = f"{original_stem}_color.jpg"
                image_path = Path("data/preprocessed") / color_name
                
            # Final fallback: try original name
            if not image_path.exists():
                image_path = Path("data/cvat_coco/images") / original_name
            
            if image_path.exists():
                image_annotations = [a for a in annotations if a['image_id'] == image_id]
                patches = self.extract_patches(image_path, image_annotations, categories, image_id)
                all_patches.extend(patches)
            else:
                print(f"Image not found: {original_name} (tried segmented: {segmented_name})")
        
        # Save results
        df = pd.DataFrame(all_patches)
        if len(df) > 0:
            df.to_csv(self.config.output_dir / "patches.csv", index=False)
            
            print(f"Generated {len(df)} patches")
            print("Label distribution:")
            for label, count in df['label'].value_counts().items():
                print(f"  {label}: {count}")
        else:
            print("No patches generated")
        
        return df


def main() -> pd.DataFrame:
    """Main entry point"""
    processor = COCOProcessor()
    return processor.run()


if __name__ == "__main__":
    patches_df = main()