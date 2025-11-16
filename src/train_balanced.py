from pathlib import Path
import warnings

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

warnings.filterwarnings('ignore')


class BalancedCutTrainer:
    """Train RandomForest with undersampling balance"""
    
    def __init__(self):
        self.features_file = Path("data/features/patch_features.npz")
        self.model_file = Path("models/rf_model_balanced.joblib")
        self.encoder_file = Path("models/label_encoder_balanced.joblib")
        
    def balance_by_cutting(self, X, y):
        """Balance dataset by undersampling majority classes"""
        print("\nBalancing dataset by cutting...")
        
        # Get class counts
        unique_classes, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)
        
        print("Original distribution:")
        for cls, count in zip(unique_classes, counts):
            print(f"  {cls}: {count}")
        
        print(f"\nCutting all classes to {min_count} samples each")
        
        # Undersample each class
        X_balanced = []
        y_balanced = []
        
        for cls in unique_classes:
            # Get all samples for this class
            class_indices = np.where(y == cls)[0]
            
            # Randomly sample min_count samples
            if len(class_indices) > min_count:
                sampled_indices = resample(class_indices, 
                                         n_samples=min_count, 
                                         random_state=42, 
                                         replace=False)
            else:
                sampled_indices = class_indices
            
            X_balanced.append(X[sampled_indices])
            y_balanced.append(y[sampled_indices])
        
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.hstack(y_balanced)
        
        print("Balanced distribution:")
        unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
        for cls, count in zip(unique_balanced, counts_balanced):
            print(f"  {cls}: {count}")
        
        print(f"Total samples: {len(X)} → {len(X_balanced)}")
        
        # Verify perfect balance
        assert len(np.unique(counts_balanced)) == 1, "Classes not perfectly balanced!"
        print(f"Perfect balance achieved: {counts_balanced[0]} samples per class")
        
        return X_balanced, y_balanced
    
    def train_balanced_model(self):
        """Train with balanced data"""
        print("Loading features...")
        data = np.load(self.features_file)
        X, y = data['X'], data['y']
        
        # Balance by cutting
        X_balanced, y_balanced = self.balance_by_cutting(X, y)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_balanced)
        
        # Holdout split (15% for final test)
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X_balanced, y_encoded, 
            test_size=0.15, 
            random_state=42, 
            stratify=y_encoded
        )
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, 
            test_size=0.25, 
            random_state=42, 
            stratify=y_dev
        )
        
        print(f"\nTraining on balanced dataset...")
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Holdout set: {len(X_holdout)} samples")
        
        # Train RandomForest with heavily regularized parameters
        rf = RandomForestClassifier(
            n_estimators=100,        # Fewer trees to reduce overfitting
            max_depth=8,            # Much shallower trees
            min_samples_split=20,   # Require more samples to split
            min_samples_leaf=10,    # Larger leaf nodes
            max_features=0.5,       # Use only half the features
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,
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
        
        print(f"\nBalanced Model Performance:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Validation accuracy: {val_score:.3f}")
        print(f"OOB accuracy: {oob_score:.3f}")
        print(f"Holdout accuracy: {holdout_score:.3f}")
        print(f"Overfitting gap: {overfitting_gap:.3f}")
        print(f"Generalization gap: {generalization_gap:.3f}")
        
        # Check improvements
        if overfitting_gap <= 0.05:
            print("Overfitting under control!")
        else:
            print(f"Some overfitting (gap: {overfitting_gap:.3f})")
            
        if abs(generalization_gap) <= 0.03:
            print("Good generalization!")
        else:
            print(f"Generalization gap: {generalization_gap:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_dev, y_dev, cv=5, scoring='accuracy')
        print(f"CV accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Detailed holdout evaluation
        holdout_pred = rf.predict(X_holdout)
        print(f"\nHoldout Test Performance:")
        print(classification_report(y_holdout, holdout_pred, 
                                   target_names=label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_holdout, holdout_pred)
        print("\nConfusion Matrix:")
        print("     ", " ".join(f"{cls:>8}" for cls in label_encoder.classes_))
        for i, row in enumerate(cm):
            print(f"{label_encoder.classes_[i]:>8}", " ".join(f"{val:>8}" for val in row))
        
        return rf, label_encoder, holdout_score
    
    def run(self):
        """Execute balanced training"""
        rf, label_encoder, holdout_score = self.train_balanced_model()
        
        # Save model
        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, self.model_file)
        joblib.dump(label_encoder, self.encoder_file)
        
        print(f"\nBalanced model saved to {self.model_file}")
        
        if holdout_score >= 0.75:
            print("BALANCED MODEL READY FOR PRODUCTION!")
        else:
            print(f"Model accuracy: {holdout_score:.3f}")
        
        return rf, label_encoder


def main():
    trainer = BalancedCutTrainer()
    trainer.run()


if __name__ == "__main__":
    main()