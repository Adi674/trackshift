import base64
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Track Friction Analyzer",
    description="Racing track friction analysis API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = joblib.load("models/rf_model_balanced.joblib")
    encoder = joblib.load("models/label_encoder_balanced.joblib")
    print("Model loaded: Random Forest 71.6% accuracy")
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    encoder = None
    MODEL_LOADED = False


def extract_patch_features(patch: np.ndarray) -> np.ndarray:
    """Extract LBP + texture features from patch"""
    from skimage import feature
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    sobel_features = [np.mean(sobel_mag), np.std(sobel_mag)]
    gray_features = [np.mean(gray), np.std(gray)]
    
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    lab_features = [
        np.mean(lab[:,:,0]), np.mean(lab[:,:,1]), np.mean(lab[:,:,2]),
        np.std(lab[:,:,0]), np.std(lab[:,:,1]), np.std(lab[:,:,2])
    ]
    
    return np.concatenate([lbp_hist, sobel_features, gray_features, lab_features])


def analyze_racing_image(image: np.ndarray):
    """Analyze racing image and generate friction heatmap"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not available")
    
    h, w = image.shape[:2]
    patch_size = 64
    patch_stride = 32
    
    patch_predictions = []
    patch_positions = []
    
    for y in range(0, h - patch_size + 1, patch_stride):
        for x in range(0, w - patch_size + 1, patch_stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            features = extract_patch_features(patch).reshape(1, -1)
            prob = model.predict_proba(features)[0]
            
            patch_predictions.append(prob)
            patch_positions.append([x + patch_size//2, y + patch_size//2])
    
    patch_predictions = np.array(patch_predictions)
    patch_positions = np.array(patch_positions)
    
    from scipy.interpolate import griddata
    
    low_idx = list(encoder.classes_).index('Low')
    low_scores = patch_predictions[:, low_idx]
    
    xi = np.arange(0, w, 1)
    yi = np.arange(0, h, 1)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    heatmap = griddata(patch_positions, low_scores, (xi_grid, yi_grid), method='cubic', fill_value=0)
    heatmap = np.clip(heatmap, 0, 1)
    
    # Generate colored overlay with reduced brightness
    colored_heatmap = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    
    # Softer, less bright colors
    colored_heatmap[heatmap <= 0.3] = [0, 0.7, 0]      # Darker green (safe)
    colored_heatmap[(heatmap > 0.3) & (heatmap <= 0.7)] = [0.8, 0.8, 0]  # Muted yellow (medium)
    colored_heatmap[heatmap > 0.7] = [0.9, 0, 0]       # Darker red (dangerous)
    
    colored_bgr = (colored_heatmap[:,:,[2,1,0]] * 255).astype(np.uint8)
    
    mask = heatmap > 0.1
    overlay = image.copy()
    alpha = 0.4  # Reduced opacity from 0.6 to 0.4 for subtler effect
    overlay[mask] = cv2.addWeighted(image[mask], 1-alpha, colored_bgr[mask], alpha, 0)
    
    dangerous_patches = np.sum(low_scores > 0.7)
    total_patches = len(low_scores)
    danger_percentage = dangerous_patches / total_patches if total_patches > 0 else 0
    
    return {
        'overlay': overlay,
        'stats': {
            'total_patches': total_patches,
            'dangerous_patches': dangerous_patches,
            'danger_percentage': danger_percentage,
            'max_danger_score': float(np.max(low_scores)) if len(low_scores) > 0 else 0
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_accuracy": "71.6%" if MODEL_LOADED else "N/A",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """Analyze racing track image and return friction heatmap"""
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        h, w = cv_image.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            cv_image = cv2.resize(cv_image, (new_w, new_h))
        
        result = analyze_racing_image(cv_image)
        
        # Save heatmap as actual image file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_filename = f"heatmap_{timestamp}.jpg"
        heatmap_path = Path("results/heatmaps") / heatmap_filename
        
        # Create directory if it doesn't exist
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the heatmap overlay as image file
        cv2.imwrite(str(heatmap_path), result['overlay'])
        
        # Convert overlay to base64 (for web display)
        _, buffer = cv2.imencode('.jpg', result['overlay'])
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"Heatmap saved to: {heatmap_path}")
        
        # Build response
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "image_size": [cv_image.shape[1], cv_image.shape[0]],
            "heatmap_base64": heatmap_base64,
            "heatmap_file_path": str(heatmap_path),
            "statistics": {
                "total_patches_analyzed": result['stats']['total_patches'],
                "dangerous_area_percentage": f"{result['stats']['danger_percentage']:.1%}",
                "max_danger_score": f"{result['stats']['max_danger_score']:.3f}",
                "safety_status": "SAFE" if result['stats']['danger_percentage'] < 0.2 else "CAUTION"
            },
            "model_info": {
                "accuracy": "71.6%",
                "algorithm": "Random Forest"
            }
        }
        
        print(f"Analyzed image: {cv_image.shape} - {result['stats']['danger_percentage']:.1%} dangerous")
        return response
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Track Friction Analysis API")
    print("API documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)