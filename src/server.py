#!/usr/bin/env python3
"""
Track Friction Analysis API Server
FastAPI server for real-time friction heatmap generation.
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from infer import FrictionInference

# Initialize FastAPI app
app = FastAPI(
    title="Track Friction Analyzer",
    description="Real-time racing track friction analysis API",
    version="1.0.0"
)

# Add CORS middleware for web demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
try:
    predictor = FrictionInference()
    print("✅ Friction inference model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    predictor = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        return {"status": "error", "message": "Model not loaded"}
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze-image")
async def analyze_track_image(
    image: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None)
):
    """
    Analyze track image and return friction heatmap
    
    Args:
        image: Racing track image file
        camera_id: Optional camera identifier for temporal tracking
        timestamp: Optional timestamp for the image
    
    Returns:
        JSON response with heatmap, grid data, and summary statistics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate image file
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await image.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Generate unique image ID
        current_time = datetime.now()
        if timestamp is None:
            timestamp = current_time.isoformat()
        
        image_id = f"{camera_id or 'unknown'}_{current_time.strftime('%Y%m%dT%H%M%S')}"
        
        # Save temporary image for processing
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / f"{image_id}.jpg"
        cv2.imwrite(str(temp_image_path), cv_image)
        
        # Run inference
        result = predictor.predict_image(str(temp_image_path), save_overlay=True)
        
        # Clean up temp file
        temp_image_path.unlink(missing_ok=True)
        
        # Convert heatmap to base64 for web transmission
        overlay_path = result['overlay_path']
        heatmap_base64 = None
        
        if overlay_path and Path(overlay_path).exists():
            with open(overlay_path, 'rb') as f:
                heatmap_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Build grid response
        positions = result['patch_positions']
        scores = result['patch_scores']
        
        grid = []
        for i, (pos, score) in enumerate(zip(positions, scores)):
            # Determine label based on score
            if score > 0.7:
                label = "low"
                confidence = float(score)
            elif score > 0.4:
                label = "medium" 
                confidence = float(1.0 - abs(score - 0.5) * 2)
            else:
                label = "high"
                confidence = float(1.0 - score)
            
            grid.append({
                "patch_id": f"{image_id}_r{i//20}_c{i%20}",  # Rough grid position
                "bbox": [
                    int(pos[0]) - 32, int(pos[1]) - 32,
                    int(pos[0]) + 32, int(pos[1]) + 32
                ],
                "score": float(score),
                "label": label,
                "confidence": confidence
            })
        
        # Generate summary
        summary = result['summary']
        
        # Find worst zone
        if len(scores) > 0:
            worst_idx = int(np.argmax(scores))
            worst_zone = f"r{worst_idx//20}_c{worst_idx%20}"
        else:
            worst_zone = "none"
        
        # Check for persistent zones (placeholder - would need temporal tracking)
        persistent_zones = []
        if summary['percent_low_friction'] > 0.5:  # If >50% low friction
            persistent_zones = [worst_zone]
        
        response = {
            "status": "ok",
            "image_id": image_id,
            "timestamp": timestamp,
            "heatmap_file": overlay_path,
            "heatmap_base64": heatmap_base64,
            "grid": grid,
            "summary": {
                "total_patches": int(summary['total_patches']),
                "low_friction_patches": int(summary['low_friction_patches']),
                "percent_low": float(summary['percent_low_friction']),
                "worst_zone": worst_zone,
                "max_danger_score": float(summary['max_low_score']),
                "persistent_zones": persistent_zones
            }
        }
        
        print(f"✅ Analyzed image: {image_id}")
        print(f"   Total patches: {summary['total_patches']}")
        print(f"   Dangerous areas: {summary['percent_low_friction']:.1%}")
        
        return response
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/fetch-history")
async def fetch_history(camera_id: str, limit: int = 10):
    """
    Fetch recent analysis history for a camera
    
    Args:
        camera_id: Camera identifier
        limit: Maximum number of recent results to return
    
    Returns:
        List of recent analysis results
    """
    # Placeholder implementation - would integrate with database
    return {
        "status": "ok",
        "camera_id": camera_id,
        "history": [],
        "message": "History tracking not implemented yet"
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Track Friction Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/analyze-image (POST)",
            "history": "/fetch-history (GET)",
            "health": "/health (GET)"
        },
        "model_loaded": predictor is not None
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Track Friction Analysis Server...")
    print("📊 Model loaded and ready for inference")
    print("🌐 Access API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)