#!/usr/bin/env python3
"""
Regularized Training Pipeline
Reduce overfitting with conservative hyperparameters.
"""

from pathlib import Path
import warnings

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


class RegularizedTrainer:
    """Train RandomForest with overfitting prevention"""
    
    def __init__(self):
        self.features_file = Path("data/features/patch_features.npz")
        self.model_file = Path("models/rf_model_regularized.joblib")
        self.encoder_file = Path("models/label_encoder_regularized.joblib")
        
    def train_regularized_model(self):
        """Train with overfitting prevention"""
        print("Loading features...")
        data = np.load(self.features_file)
        X, y = data['X'], data['y']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Holdout split
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
        )
        
        # Conservative SMOTE (less aggressive)
        smote = SMOTE(random_state=42, k_neighbors=3)  # Fewer neighbors
        X_balanced, y_balanced = smote.fit_resample(X_dev, y_dev)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_balanced, test_size=0.25, random_state=42, stratify=y_balanced
        )
        
        print("Training regularized RandomForest...")
        
        # REGULARIZED HYPERPARAMETERS
        rf = RandomForestClassifier(
            n_estimators=200,        # Fewer trees
            max_depth=12,           # Shallower trees
            min_samples_split=10,   # More samples required to split
            min_samples_leaf=5,     # More samples required in leaf
            max_features='sqrt',    # Fewer features per tree
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,         # Out-of-bag validation
            n_jobs=-1,
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate
        train_score = rf.score(X_train, y_train)
        val_score = rf.score(X_val, y_val)
        holdout_score = rf.score(X_holdout, y_holdout)
        oob_score = rf.oob_score_
        
        overfitting_gap = train_score - val_score
        generalization_gap = val_score - holdout_score
        
        print(f"\nRegularized Model Performance:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Validation accuracy: {val_score:.3f}")
        print(f"OOB accuracy: {oob_score:.3f}")
        print(f"Holdout accuracy: {holdout_score:.3f}")
        print(f"Overfitting gap: {overfitting_gap:.3f}")
        print(f"Generalization gap: {generalization_gap:.3f}")
        
        # Check improvements
        if overfitting_gap <= 0.05:
            print("✅ Overfitting RESOLVED!")
        else:
            print("⚠️  Still some overfitting")
            
        if abs(generalization_gap) <= 0.03:
            print("✅ Good generalization!")
        else:
            print("⚠️  Generalization could be better")
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_balanced, y_balanced, cv=5)
        print(f"CV accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Holdout evaluation
        holdout_pred = rf.predict(X_holdout)
        print(f"\nHoldout Test Performance:")
        print(classification_report(y_holdout, holdout_pred, 
                                   target_names=label_encoder.classes_))
        
        return rf, label_encoder, holdout_score
    
    def run(self):
        """Execute regularized training"""
        rf, label_encoder, holdout_score = self.train_regularized_model()
        
        # Save model
        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, self.model_file)
        joblib.dump(label_encoder, self.encoder_file)
        
        print(f"\nRegularized model saved to {self.model_file}")
        
        if holdout_score >= 0.85:
            print("🎉 REGULARIZED MODEL READY FOR PRODUCTION!")
        else:
            print("⚠️  Model performance below production threshold")
        
        return rf, label_encoder


def main():
    trainer = RegularizedTrainer()
    trainer.run()


if __name__ == "__main__":
    main()