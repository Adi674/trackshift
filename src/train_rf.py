#!/usr/bin/env python3
"""
Balanced RandomForest Training Pipeline with Overfitting Detection
Train friction classifier with SMOTE balancing and comprehensive validation.
"""

from pathlib import Path
import warnings

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    validation_curve, learning_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


class RobustRandomForestTrainer:
    """Train and validate RandomForest with overfitting detection"""
    
    def __init__(self):
        self.features_file = Path("data/features/patch_features.npz")
        self.model_file = Path("models/rf_model.joblib")
        self.encoder_file = Path("models/label_encoder.joblib")
        self.plots_dir = Path("results/validation_plots")
        
    def load_features(self):
        """Load feature data"""
        print("Loading features...")
        data = np.load(self.features_file)
        X = data['X']
        y = data['y']
        ids = data['ids']
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print("Original class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
        
        return X, y, ids
    
    def create_holdout_split(self, X, y, test_size=0.15):
        """Create holdout test set BEFORE any processing"""
        print(f"\nCreating {test_size:.0%} holdout test set...")
        
        # Encode labels for stratification
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split into dev and holdout
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )
        
        print(f"Development set: {len(X_dev)} samples")
        print(f"Holdout test set: {len(X_holdout)} samples")
        
        return X_dev, X_holdout, y_dev, y_holdout, label_encoder
    
    def apply_smote(self, X, y):
        """Apply SMOTE to balance the dataset"""
        print("\nApplying SMOTE balancing...")
        
        smote = SMOTE(
            random_state=42,
            k_neighbors=5,
            sampling_strategy='auto'
        )
        
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"Dataset size: {len(X)} → {len(X_balanced)} samples")
        return X_balanced, y_balanced
    
    def plot_learning_curves(self, rf, X, y, label_encoder):
        """Plot learning curves to detect overfitting"""
        print("\nGenerating learning curves...")
        
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        train_sizes, train_scores, val_scores = learning_curve(
            rf, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves - Overfitting Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.plots_dir / "learning_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Analyze overfitting
        final_gap = train_mean[-1] - val_mean[-1]
        print(f"Final training-validation gap: {final_gap:.3f}")
        
        if final_gap > 0.05:
            print("⚠️  WARNING: Possible overfitting detected (gap > 5%)")
        else:
            print("✅ No significant overfitting detected")
        
        return final_gap
    
    def plot_validation_curves(self, X, y):
        """Plot validation curves for different n_estimators"""
        print("Generating validation curves...")
        
        # Ensure plots directory exists
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        param_range = [50, 100, 200, 300, 400, 500, 600]
        
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(
                max_depth=20, 
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            X, y, 
            param_name='n_estimators',
            param_range=param_range,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy Score')
        plt.title('Validation Curves - Model Complexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.plots_dir / "validation_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Find optimal n_estimators
        optimal_idx = np.argmax(val_mean)
        optimal_n = param_range[optimal_idx]
        print(f"Optimal n_estimators: {optimal_n}")
        
        return optimal_n
    
    def train_model(self, X, y):
        """Train with comprehensive validation"""
        print("\nTraining with Overfitting Detection...")
        
        # Create holdout set FIRST
        X_dev, X_holdout, y_dev, y_holdout, label_encoder = self.create_holdout_split(X, y)
        
        # Apply SMOTE only to development set
        X_balanced, y_balanced = self.apply_smote(X_dev, y_dev)
        
        # Find optimal hyperparameters
        optimal_n_estimators = self.plot_validation_curves(X_balanced, y_balanced)
        
        # Train with optimal parameters
        rf = RandomForestClassifier(
            n_estimators=optimal_n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        # Split balanced data for training
        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_balanced
        )
        
        rf.fit(X_train, y_train)
        
        # Generate learning curves
        overfitting_gap = self.plot_learning_curves(rf, X_balanced, y_balanced, label_encoder)
        
        # Evaluate on validation set
        val_score = rf.score(X_val, y_val)
        train_score = rf.score(X_train, y_train)
        
        print(f"\nDevelopment Set Performance:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Validation accuracy: {val_score:.3f}")
        print(f"Train-Val gap: {train_score - val_score:.3f}")
        
        # CRUCIAL: Test on completely unseen holdout set
        print("\n" + "="*60)
        print("HOLDOUT TEST SET PERFORMANCE (Unseen Data)")
        print("="*60)
        
        holdout_score = rf.score(X_holdout, y_holdout)
        holdout_pred = rf.predict(X_holdout)
        
        print(f"Holdout test accuracy: {holdout_score:.3f}")
        print(f"Generalization gap: {val_score - holdout_score:.3f}")
        
        if abs(val_score - holdout_score) > 0.03:
            print("⚠️  WARNING: Large generalization gap detected!")
        else:
            print("✅ Good generalization performance")
        
        print("\nHoldout Classification Report:")
        print(classification_report(
            y_holdout, holdout_pred, 
            target_names=label_encoder.classes_
        ))
        
        return rf, label_encoder, {
            'overfitting_gap': overfitting_gap,
            'generalization_gap': val_score - holdout_score,
            'holdout_accuracy': holdout_score,
            'validation_accuracy': val_score
        }
    
    def save_model(self, rf, label_encoder, validation_metrics):
        """Save model with validation metrics"""
        # Create models directory
        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and encoder
        joblib.dump(rf, self.model_file)
        joblib.dump(label_encoder, self.encoder_file)
        
        # Save validation metrics
        import json
        metrics_file = self.model_file.parent / "validation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(validation_metrics, f, indent=2)
        
        print(f"\nModel saved to {self.model_file}")
        print(f"Validation metrics saved to {metrics_file}")
        print(f"Plots saved to {self.plots_dir}")
        
        # Feature importance
        feature_names = [
            'lbp_0', 'lbp_1', 'lbp_2', 'lbp_3', 'lbp_4',
            'lbp_5', 'lbp_6', 'lbp_7', 'lbp_8', 'lbp_9',
            'sobel_mean', 'sobel_std', 'gray_mean', 'gray_std',
            'lab_l_mean', 'lab_a_mean', 'lab_b_mean',
            'lab_l_std', 'lab_a_std', 'lab_b_std'
        ]
        
        importance = rf.feature_importances_
        top_features = sorted(
            zip(feature_names, importance), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print("\nTop 10 Feature Importance:")
        for name, imp in top_features:
            print(f"  {name}: {imp:.3f}")
    
    def run(self):
        """Execute robust training with overfitting detection"""
        # Load data
        X, y, ids = self.load_features()
        
        # Train with validation
        rf, label_encoder, metrics = self.train_model(X, y)
        
        # Save model and metrics
        self.save_model(rf, label_encoder, metrics)
        
        print("\n" + "="*60)
        print("ROBUST TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Overfitting gap: {metrics['overfitting_gap']:.3f}")
        print(f"✅ Generalization gap: {metrics['generalization_gap']:.3f}")
        print(f"✅ Holdout accuracy: {metrics['holdout_accuracy']:.3f}")
        print("✅ Learning curves generated")
        print("✅ Model ready for production!")


def main():
    """Main entry point"""
    trainer = RobustRandomForestTrainer()
    trainer.run()


if __name__ == "__main__":
    main()