#!/usr/bin/env python3
"""
Data Preprocessing Pipeline
Resize and enhance racing track images for optimal feature extraction.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ImagePreprocessor:
    """Preprocess racing track images"""
    
    def __init__(self):
        self.input_dir = Path("data/cvat_coco/images")
        self.output_dir = Path("data/preprocessed")
        self.target_size = 1280  # Max dimension
        
    def preprocess_image(self, image_path: Path) -> tuple:
        """Preprocess single image"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
            
        h, w = image.shape[:2]
        
        # Resize maintaining aspect ratio
        if max(h, w) > self.target_size:
            scale = self.target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            image_resized = image.copy()
        
        # Save color version
        color_path = self.output_dir / f"{image_path.stem}_color.jpg"
        cv2.imwrite(str(color_path), image_resized)
        
        # Create enhanced grayscale with CLAHE
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Save enhanced grayscale
        gray_path = self.output_dir / f"{image_path.stem}_gray_clahe.jpg"
        cv2.imwrite(str(gray_path), gray_enhanced)
        
        print(f"Processed: {image_path.name} ({w}x{h} → {image_resized.shape[1]}x{image_resized.shape[0]})")
        
        return color_path, gray_path
    
    def run(self):
        """Process all images"""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(self.input_dir.glob(ext))
        
        if not image_files:
            print("No images found in input directory!")
            return
        
        print(f"Found {len(image_files)} images to preprocess")
        
        # Process images
        processed_count = 0
        for image_path in tqdm(image_files, desc="Preprocessing"):
            color_path, gray_path = self.preprocess_image(image_path)
            if color_path and gray_path:
                processed_count += 1
        
        print(f"\nPreprocessing complete!")
        print(f"Successfully processed: {processed_count}/{len(image_files)} images")
        print(f"Output saved to: {self.output_dir}")


def main():
    """Main entry point"""
    preprocessor = ImagePreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()