#!/usr/bin/env python3
"""
Track Segmentation Pipeline
Isolate racing track surface from barriers, grass, sky, and other non-track elements.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class TrackSegmentator:
    """Segment racing track from background elements"""
    
    def __init__(self):
        self.input_dir = Path("data/preprocessed")
        self.output_dir = Path("data/segmented")
        
    def segment_track(self, image_path: Path) -> tuple:
        """Segment track surface from single image"""
        # Read preprocessed color image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Create track mask using multiple methods
        
        # Method 1: Color-based segmentation (asphalt is typically gray)
        # Asphalt HSV ranges
        lower_asphalt = np.array([0, 0, 20])     # Dark gray
        upper_asphalt = np.array([180, 50, 150])  # Light gray
        asphalt_mask = cv2.inRange(hsv, lower_asphalt, upper_asphalt)
        
        # Method 2: Texture-based (asphalt has specific texture patterns)
        # Use Sobel to detect texture
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Asphalt has moderate texture (not too smooth like sky, not too rough like grass)
        texture_mask = ((sobel_mag > 10) & (sobel_mag < 100)).astype(np.uint8) * 255
        
        # Method 3: Remove obvious non-track elements
        
        # Remove sky (typically top portion, blue/white)
        h, w = image.shape[:2]
        sky_mask = np.zeros_like(gray)
        sky_mask[0:h//4, :] = 255  # Top quarter likely sky
        
        # Remove very bright areas (likely sky/clouds)
        bright_mask = (gray > 200).astype(np.uint8) * 255
        
        # Remove very green areas (grass/vegetation)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        
        # Combine masks
        # Start with asphalt color mask
        track_mask = asphalt_mask.copy()
        
        # Add texture information
        track_mask = cv2.bitwise_or(track_mask, texture_mask)
        
        # Remove non-track elements
        track_mask = cv2.bitwise_and(track_mask, cv2.bitwise_not(sky_mask))
        track_mask = cv2.bitwise_and(track_mask, cv2.bitwise_not(bright_mask))
        track_mask = cv2.bitwise_and(track_mask, cv2.bitwise_not(green_mask))
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel)
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep largest contour (main track surface)
            largest_contour = max(contours, key=cv2.contourArea)
            track_mask = np.zeros_like(track_mask)
            cv2.fillPoly(track_mask, [largest_contour], 255)
        
        # Apply mask to original image
        segmented_image = image.copy()
        segmented_image[track_mask == 0] = [0, 0, 0]  # Black out non-track areas
        
        # Save outputs
        mask_path = self.output_dir / "masks" / f"{image_path.stem}_mask.jpg"
        segmented_path = self.output_dir / "segmented" / f"{image_path.stem}_segmented.jpg"
        
        cv2.imwrite(str(mask_path), track_mask)
        cv2.imwrite(str(segmented_path), segmented_image)
        
        # Calculate track coverage
        track_pixels = np.sum(track_mask > 0)
        total_pixels = track_mask.size
        coverage = track_pixels / total_pixels
        
        print(f"Segmented: {image_path.name} - Track coverage: {coverage:.1%}")
        
        return segmented_path, mask_path, coverage
    
    def run(self):
        """Process all preprocessed images"""
        # Create output directories
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "segmented").mkdir(parents=True, exist_ok=True)
        
        # Get all preprocessed color images
        image_files = list(self.input_dir.glob("*_color.jpg"))
        
        if not image_files:
            print("No preprocessed color images found!")
            return
        
        print(f"Found {len(image_files)} preprocessed images to segment")
        
        # Process images
        results = []
        for image_path in tqdm(image_files, desc="Segmenting"):
            segmented_path, mask_path, coverage = self.segment_track(image_path)
            if segmented_path and mask_path:
                results.append({
                    'original': image_path.name,
                    'segmented': segmented_path,
                    'mask': mask_path,
                    'coverage': coverage
                })
        
        print(f"\nSegmentation complete!")
        print(f"Successfully segmented: {len(results)}/{len(image_files)} images")
        
        # Summary statistics
        if results:
            coverages = [r['coverage'] for r in results]
            avg_coverage = np.mean(coverages)
            print(f"Average track coverage: {avg_coverage:.1%}")
            print(f"Coverage range: {min(coverages):.1%} - {max(coverages):.1%}")
        
        print(f"Segmented images saved to: {self.output_dir}")


def main():
    """Main entry point"""
    segmentator = TrackSegmentator()
    segmentator.run()


if __name__ == "__main__":
    main()