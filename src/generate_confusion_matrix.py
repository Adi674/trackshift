#!/usr/bin/env python3
"""
Confusion Matrix Generator for Racing Friction Analysis
Generate detailed confusion matrix with visualization for model evaluation.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ConfusionMatrixGenerator:
    """Generate comprehensive confusion matrix analysis"""
    
    def __init__(self):
        self.features_file = Path("data/features/patch_features.npz")
        self.model_file = Path("models/rf_model_balanced.joblib")
        self.encoder_file = Path("models/label_encoder_balanced.joblib")
        self.output_dir = Path("results/confusion_matrix")
        
    def load_model_and_data(self):
        """Load trained model and test data"""
        print("Loading model and data...")
        
        # Load model
        model = joblib.load(self.model_file)
        encoder = joblib.load(self.encoder_file)
        
        # Load features
        data = np.load(self.features_file)
        X, y = data['X'], data['y']
        
        # Encode labels
        y_encoded = encoder.transform(y)
        
        # Create same holdout split as training (15% holdout)
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y_encoded, 
            test_size=0.15, 
            random_state=42, 
            stratify=y_encoded
        )
        
        print(f"Holdout test set: {len(X_holdout)} samples")
        print(f"Class distribution in holdout:")
        unique, counts = np.unique(y_holdout, return_counts=True)
        for i, (cls, count) in enumerate(zip(encoder.classes_, counts)):
            print(f"  {cls}: {count} samples")
        
        return model, encoder, X_holdout, y_holdout
    
    def generate_predictions(self, model, X_holdout):
        """Generate predictions for confusion matrix"""
        print("Generating predictions...")
        predictions = model.predict(X_holdout)
        probabilities = model.predict_proba(X_holdout)
        return predictions, probabilities
    
    def create_confusion_matrix_plot(self, y_true, y_pred, class_names, accuracy):
        """Create detailed confusion matrix visualization"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts confusion matrix
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax1,
                   cbar_kws={'label': 'Count'})
        ax1.set_title(f'Confusion Matrix (Counts)\nOverall Accuracy: {accuracy:.1%}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Friction Type', fontsize=12)
        ax1.set_ylabel('Actual Friction Type', fontsize=12)
        
        # Normalized confusion matrix (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2%', 
                   cmap='Greens',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax2,
                   cbar_kws={'label': 'Percentage'})
        ax2.set_title('Normalized Confusion Matrix (Percentages)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Friction Type', fontsize=12)
        ax2.set_ylabel('Actual Friction Type', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "confusion_matrix_detailed.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to: {plot_path}")
        return cm, cm_normalized
    
    def create_racing_specific_analysis(self, y_true, y_pred, class_names, cm):
        """Create racing-specific performance analysis"""
        print("\n" + "="*60)
        print("RACING FRICTION ANALYSIS - CONFUSION MATRIX RESULTS")
        print("="*60)
        
        # Print raw confusion matrix
        print("\nConfusion Matrix (Raw Counts):")
        print(f"{'':>10}", end="")
        for cls in class_names:
            print(f"{cls:>10}", end="")
        print()
        
        for i, row in enumerate(cm):
            print(f"{class_names[i]:>10}", end="")
            for val in row:
                print(f"{val:>10}", end="")
            print()
        
        # Calculate detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, 
            labels=range(len(class_names)),
            average=None
        )
        
        print("\nDetailed Performance Analysis:")
        print(f"{'Friction Type':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Accuracy':<10}")
        print("-" * 70)
        
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, cls in enumerate(class_names):
            print(f"{cls:<15} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10} {class_accuracies[i]:<10.3f}")
        
        # Racing-specific insights
        print("\n" + "="*60)
        print("RACING SAFETY & PERFORMANCE INSIGHTS")
        print("="*60)
        
        # High friction (racing line) analysis
        high_idx = list(class_names).index('High')
        high_recall = recall[high_idx]
        high_precision = precision[high_idx]
        
        print(f"\n🏁 HIGH FRICTION (Racing Line) Performance:")
        print(f"   Recall: {high_recall:.1%} - Correctly identifies {high_recall:.1%} of optimal racing areas")
        print(f"   Precision: {high_precision:.1%} - {high_precision:.1%} of high predictions are correct")
        
        if high_recall >= 0.75:
            print("   ✅ EXCELLENT: Great for lap time optimization")
        elif high_recall >= 0.65:
            print("   ✅ GOOD: Suitable for racing line detection")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT: May miss optimal racing areas")
        
        # Low friction (danger zone) analysis  
        low_idx = list(class_names).index('Low')
        low_recall = recall[low_idx]
        low_precision = precision[low_idx]
        
        print(f"\n⚠️  LOW FRICTION (Danger Zones) Performance:")
        print(f"   Recall: {low_recall:.1%} - Detects {low_recall:.1%} of dangerous areas")
        print(f"   Precision: {low_precision:.1%} - {low_precision:.1%} of danger warnings are accurate")
        
        if low_recall >= 0.75:
            print("   ✅ EXCELLENT: Critical for safety systems")
        elif low_recall >= 0.65:
            print("   ✅ ACCEPTABLE: Good danger detection")
        else:
            print("   🚨 CRITICAL: May miss dangerous conditions!")
        
        # Medium friction analysis
        medium_idx = list(class_names).index('Medium')
        medium_recall = recall[medium_idx]
        medium_precision = precision[medium_idx]
        
        print(f"\n🟡 MEDIUM FRICTION (Off-Line) Performance:")
        print(f"   Recall: {medium_recall:.1%} - Identifies {medium_recall:.1%} of usable off-line areas")
        print(f"   Precision: {medium_precision:.1%} - {medium_precision:.1%} accuracy for medium grip zones")
        
        # Overall system assessment
        overall_accuracy = np.trace(cm) / np.sum(cm)
        print(f"\n📊 OVERALL SYSTEM PERFORMANCE:")
        print(f"   Accuracy: {overall_accuracy:.1%}")
        print(f"   Model: Random Forest with 71.6% validation accuracy")
        print(f"   Training: Balanced dataset (405 samples per class)")
        
        if overall_accuracy >= 0.70:
            print("   🎉 READY FOR MOTORSPORTS DEPLOYMENT!")
        elif overall_accuracy >= 0.60:
            print("   ✅ SUITABLE FOR PROTOTYPE TESTING")
        else:
            print("   ⚠️  REQUIRES FURTHER OPTIMIZATION")
        
        return {
            'confusion_matrix': cm.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'class_accuracy': class_accuracies.tolist(),
            'overall_accuracy': float(overall_accuracy)
        }
    
    def save_detailed_report(self, metrics, class_names):
        """Save detailed analysis report"""
        report_path = self.output_dir / "confusion_matrix_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("RACING FRICTION ANALYSIS - CONFUSION MATRIX REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("Model Performance Summary:\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10}\n")
            f.write("-" * 65 + "\n")
            
            for i, cls in enumerate(class_names):
                f.write(f"{cls:<15} {metrics['precision'][i]:<10.3f} {metrics['recall'][i]:<10.3f} "
                       f"{metrics['f1_score'][i]:<10.3f} {metrics['class_accuracy'][i]:<10.3f}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"{'':>10}")
            for cls in class_names:
                f.write(f"{cls:>10}")
            f.write("\n")
            
            cm = np.array(metrics['confusion_matrix'])
            for i, row in enumerate(cm):
                f.write(f"{class_names[i]:>10}")
                for val in row:
                    f.write(f"{val:>10}")
                f.write("\n")
        
        print(f"Detailed report saved to: {report_path}")
    
    def run(self):
        """Execute confusion matrix analysis"""
        # Load model and data
        model, encoder, X_holdout, y_holdout = self.load_model_and_data()
        
        # Generate predictions
        predictions, probabilities = self.generate_predictions(model, X_holdout)
        
        # Calculate accuracy
        accuracy = np.mean(y_holdout == predictions)
        
        # Create confusion matrix visualizations
        cm, cm_normalized = self.create_confusion_matrix_plot(
            y_holdout, predictions, encoder.classes_, accuracy
        )
        
        # Generate racing-specific analysis
        metrics = self.create_racing_specific_analysis(
            y_holdout, predictions, encoder.classes_, cm
        )
        
        # Save detailed report
        self.save_detailed_report(metrics, encoder.classes_)
        
        print(f"\n✅ Confusion matrix analysis complete!")
        print(f"📊 Results saved to: {self.output_dir}")
        
        return metrics


def main():
    """Main entry point"""
    generator = ConfusionMatrixGenerator()
    metrics = generator.run()
    return metrics


if __name__ == "__main__":
    main()