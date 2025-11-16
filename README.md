# 🏁 TrackShift - Racing Track Monitoring & Analysis System

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-trackshift--frontend.vercel.app-brightgreen)](https://trackshift-frontend.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org)

**Comprehensive racing track monitoring and analysis using AI-powered computer vision to provide real-time surface analysis, friction prediction, and safety insights for enhanced race performance.**

🌐 **[Live Application](https://trackshift-frontend.vercel.app/)** - Experience the full system in action!

## 📋 Overview

TrackShift is a comprehensive racing track monitoring and analysis platform that revolutionizes how racing teams and track operators understand track conditions. Using advanced computer vision and machine learning, the system continuously monitors track surfaces, analyzes friction patterns, identifies potential hazards, and provides actionable insights for optimal racing performance and safety.

### Key Features

- **🔍 Real-time Track Monitoring**: Continuous analysis of track conditions with instant alerts
- **📊 Surface Analysis**: Advanced friction detection and surface condition assessment
- **🎯 Precision Mapping**: Grid-based 64x64 pixel patch analysis for detailed track coverage
- **📈 Performance Insights**: Data-driven recommendations for racing line optimization
- **🎨 Visual Dashboard**: Interactive heatmaps and analytics for track operators and teams
- **⚠️ Safety Alerts**: Automated detection of dangerous zones and weather impact analysis
- **🤖 ML Pipeline**: Complete training system with COCO dataset integration
- **🌐 Web Platform**: Modern React interface with real-time data visualization
- **🚀 API Integration**: RESTful backend for seamless integration with existing race systems

## 🛠️ Technology Stack

### Backend
- **Python 3.12+** - Core development language
- **FastAPI** - High-performance API framework
- **OpenCV** - Computer vision and image processing
- **scikit-learn** - Machine learning algorithms (RandomForest)
- **scikit-image** - Advanced image processing features
- **SMOTE** - Imbalanced dataset handling

### Frontend
- **React 18+** - Modern UI framework
- **Vite** - Fast development and build tool
- **Responsive Design** - Works on desktop and mobile

### Machine Learning
- **RandomForest Classifier** - Robust friction prediction
- **Local Binary Pattern (LBP)** - Texture feature extraction
- **LAB Color Space** - Advanced color analysis
- **Sobel Edge Detection** - Surface texture analysis

### Deployment
- **Vercel** - Frontend hosting and deployment
- **CORS Enabled** - Cross-origin resource sharing for web integration

## 🚀 Live Demo

Visit **[trackshift-frontend.vercel.app](https://trackshift-frontend.vercel.app/)** to experience the monitoring platform:

1. Upload racing track images or use sample data
2. Watch real-time AI analysis of track conditions  
3. Explore interactive heatmaps and safety zones
4. Access detailed monitoring reports and insights
5. View historical track condition trends

## 📁 Project Structure

```
trackgrip-live/
├── src/                          # Core Python modules
│   ├── coco_to_patches.py        # CVAT COCO dataset → training patches
│   ├── features.py               # LBP + texture feature extraction  
│   ├── train_rf.py               # RandomForest training with validation
│   ├── train_regularized.py      # Overfitting prevention training
│   ├── infer.py                  # Single image friction prediction
│   └── server.py                 # FastAPI REST API server
├── data/                         # Training data pipeline
│   ├── cvat_coco/               # COCO annotated track images
│   ├── patches/                 # 64x64 training patches
│   └── features/                # Extracted feature vectors
├── models/                      # Trained models and metrics
│   ├── rf_model_regularized.joblib
│   ├── label_encoder_regularized.joblib
│   └── validation_metrics.json
├── results/                     # Generated outputs
│   ├── overlays/               # Friction heatmap overlays
│   └── validation_plots/       # Training analysis plots
├── requirements.txt            # Python dependencies
├── config.yaml                # Configuration settings
└── .gitignore                 # Git exclusions
```

## 🏗️ Installation & Setup

### Prerequisites
- Python 3.12+
- Node.js 18+ (for frontend development)
- Git

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/manasjh1/trackgrip-live.git
cd trackgrip-live
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare training data** (if training new models)
```bash
# Convert COCO annotations to patches
python src/coco_to_patches.py

# Extract features from patches
python src/features.py

# Train regularized model
python src/train_regularized.py
```

4. **Start the API server**
```bash
python src/server.py
```
API will be available at `http://localhost:8000`

### Frontend Development

1. **Install dependencies**
```bash
npm install
```

2. **Start development server**
```bash
npm run dev
```
Frontend will be available at `http://localhost:8080`

## 🎯 Usage Examples

### API Usage

**Analyze track image:**
```bash
curl -X POST "http://localhost:8000/analyze-image" \
     -F "image=@track_image.jpg" \
     -F "camera_id=track_cam_01"
```

**Response:**
```json
{
  "status": "ok",
  "image_id": "track_cam_01_20241116T143022",
  "heatmap_base64": "data:image/jpeg;base64,...",
  "grid": [
    {
      "patch_id": "track_001_patch_0001",
      "bbox": [32, 32, 96, 96],
      "score": 0.85,
      "label": "low",
      "confidence": 0.85
    }
  ],
  "summary": {
    "total_patches": 150,
    "low_friction_patches": 12,
    "percent_low": 0.08,
    "worst_zone": "r5_c8",
    "max_danger_score": 0.92
  }
}
```

### Python Usage

```python
from src.infer import FrictionInference

# Initialize TrackShift monitoring system
track_monitor = FrictionInference()

# Analyze track conditions
analysis_result = track_monitor.predict_image("track_sector_1.jpg")

print(f"Track condition alert: {analysis_result['summary']['percent_low_friction']:.1%}")
print(f"Critical zone detected: {analysis_result['summary']['max_low_score']:.3f}")
print(f"Monitoring status: Track safety assessment complete")
```

## 🧪 Model Performance

Our regularized RandomForest model achieves:

- **Holdout Accuracy**: 87.4%
- **Validation Accuracy**: 92.3%
- **Overfitting Gap**: <8% (well-controlled)
- **Generalization Gap**: <5% (excellent)

### Model Features
- **20 Feature Dimensions**: LBP histograms + texture statistics
- **Balanced Training**: SMOTE oversampling for class balance
- **Regularization**: Conservative hyperparameters prevent overfitting
- **Cross-Validation**: 5-fold stratified validation

## 🔧 Configuration

Edit `config.yaml` for custom settings:

```yaml
patch_size: 64          # Analysis patch dimensions
patch_stride: 32        # Overlap between patches
min_coverage: 0.4       # Minimum annotation coverage
target_size: 1280       # Maximum input image dimension
```

## 📊 Data Pipeline

1. **COCO Annotation Import** (`coco_to_patches.py`)
   - Converts CVAT COCO 1.0 exports to training patches
   - Handles polygon and bounding box annotations
   - Creates 64x64 labeled patches with friction classes

2. **Feature Extraction** (`features.py`)
   - Local Binary Pattern (LBP) texture analysis
   - Sobel gradient edge detection
   - LAB color space statistics
   - 20-dimensional feature vectors

3. **Model Training** (`train_regularized.py`)
   - SMOTE balancing for class imbalance
   - RandomForest with overfitting prevention
   - Comprehensive validation with holdout testing
   - Learning curve analysis

4. **Inference Pipeline** (`infer.py`)
   - Grid-based patch extraction
   - Real-time friction prediction
   - Smooth heatmap interpolation
   - Color-coded overlay generation

## 🌐 API Documentation

When running locally, visit `http://localhost:8000/docs` for interactive API documentation.

### Main Endpoints

- `POST /analyze-image` - Upload image for friction analysis
- `GET /health` - System health check
- `GET /fetch-history` - Retrieve analysis history
- `GET /` - API information and status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Achievements

- ✅ **Production-Ready**: Deployed and accessible at [trackshift-frontend.vercel.app](https://trackshift-frontend.vercel.app/)
- ✅ **Robust ML Pipeline**: Comprehensive training with overfitting prevention
- ✅ **Real-time Processing**: Fast inference suitable for live track monitoring
- ✅ **Web Integration**: Modern React frontend with responsive design
- ✅ **API Ready**: RESTful backend for system integration

## 🔬 Technical Highlights

- **Advanced Computer Vision**: Multi-scale texture analysis using LBP and edge detection
- **Imbalanced Learning**: SMOTE oversampling with careful validation
- **Overfitting Prevention**: Regularized training with learning curve analysis
- **Production Validation**: Holdout testing ensures real-world performance
- **Scalable Architecture**: Modular design for easy extension and maintenance

## 📞 Contact & Support

For questions, issues, or collaboration opportunities:

- **Live Demo**: [trackshift-frontend.vercel.app](https://trackshift-frontend.vercel.app/)
- **GitHub**: [github.com/manasjh1/trackgrip-live](https://github.com/manasjh1/trackgrip-live)
- **Issues**: Use GitHub Issues for bug reports and feature requests

---

**🏁 Revolutionizing Racing Safety Through Intelligent Track Monitoring**

*Built with ❤️ for enhanced racing performance and safety*
